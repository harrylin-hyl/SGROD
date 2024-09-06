# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# -----------------------------------------------------------------------
"""
Deformable DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn
import math

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss)
from .segmentation import sigmoid_focal_loss as seg_sigmoid_focal_loss
from .deformable_transformer import build_deforamble_transformer
import copy
from scipy.optimize import linear_sum_assignment


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2, num_classes: int = 81, empty_weight: float = 0.1):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    W = torch.ones(num_classes, dtype=prob.dtype, layout=prob.layout, device=prob.device)
    W[-1] = empty_weight
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none", weight=W)
    
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


class ProbObjectnessHead(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.flatten = nn.Flatten(0,1)
        self.objectness_bn = nn.BatchNorm1d(hidden_dim, affine=False)

    def freeze_prob_model(self):
        self.objectness_bn.eval()
    
    def unfreeze_prob_model(self):
        self.objectness_bn.train()
        
    def forward(self, x):
        out=self.flatten(x)
        out=self.objectness_bn(out).unflatten(0, x.shape[:2])
        return out.norm(dim=-1)**2



    
class FullProbObjectnessHead(nn.Module):
    def __init__(self, hidden_dim=256, device='cpu'):
        super().__init__()
        self.flatten = nn.Flatten(0, 1)
        self.momentum = 0.1
        self.obj_mean=nn.Parameter(torch.ones(hidden_dim, device=device), requires_grad=False)
        self.obj_cov=nn.Parameter(torch.eye(hidden_dim, device=device), requires_grad=False)
        self.inv_obj_cov=nn.Parameter(torch.eye(hidden_dim, device=device), requires_grad=False)
        self.device=device
        self.hidden_dim=hidden_dim
            
    def update_params(self,x):
        out=self.flatten(x).detach()
        obj_mean=out.mean(dim=0)
        obj_cov=torch.cov(out.T)
        self.obj_mean.data = self.obj_mean*(1-self.momentum) + self.momentum*obj_mean
        self.obj_cov.data = self.obj_cov*(1-self.momentum) + self.momentum*obj_cov
        return
    
    def update_icov(self):
        self.inv_obj_cov.data = torch.pinverse(self.obj_cov.detach().cpu(), rcond=1e-6).to(self.device)
        return
        
    def mahalanobis(self, x):
        out=self.flatten(x)
        delta = out - self.obj_mean
        m = (delta * torch.matmul(self.inv_obj_cov, delta.T).T).sum(dim=-1)
        return m.unflatten(0, x.shape[:2])
    
    def set_momentum(self, m):
        self.momentum=m
        return
    
    def forward(self, x):
        if self.training:
            self.update_params(x)
        return self.mahalanobis(x)


class DeformableDETR(nn.Module):
    """ This is the Deformable DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries, num_feature_levels,
                 aux_loss=True, with_box_refine=False, two_stage=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage Deformable DETR
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.prob_obj_head = ProbObjectnessHead(hidden_dim)

        self.num_feature_levels = num_feature_levels
        if not two_stage:
            self.query_embed = nn.Embedding(num_queries, hidden_dim*2)
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = (transformer.decoder.num_layers + 1) if two_stage else transformer.decoder.num_layers
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            self.prob_obj_head =  _get_clones(self.prob_obj_head, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.prob_obj_head = nn.ModuleList([self.prob_obj_head for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None
        if two_stage:
            # hack implementation for two-stage
            self.transformer.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)

    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        query_embeds = None
        if not self.two_stage:
            query_embeds = self.query_embed.weight
        hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact = self.transformer(srcs, masks, pos, query_embeds)

        outputs_classes = []
        outputs_coords = []
        outputs_objectnesses = []
        eval_outputs_objectnesses = []

        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            outputs_objectness = self.prob_obj_head[lvl](hs[lvl])
            self.prob_obj_head[lvl].freeze_prob_model()
            eval_outputs_objectness = self.prob_obj_head[lvl](hs[lvl])
            self.prob_obj_head[lvl].unfreeze_prob_model()

            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
                
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
            outputs_objectnesses.append(outputs_objectness)
            eval_outputs_objectnesses.append(eval_outputs_objectness)
            
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)
        outputs_objectness = torch.stack(outputs_objectnesses)
        eval_outputs_objectness = torch.stack(eval_outputs_objectnesses)


        # cross-layer pair
        outputs_objectnesses[0], outputs_objectnesses[1], outputs_objectnesses[2], outputs_objectnesses[3], outputs_objectnesses[4], outputs_objectnesses[5] = \
                outputs_objectnesses[5], outputs_objectnesses[4], outputs_objectnesses[3], outputs_objectnesses[2], outputs_objectnesses[1], outputs_objectnesses[0]
        eval_outputs_objectnesses[0], eval_outputs_objectnesses[1], eval_outputs_objectnesses[2], eval_outputs_objectnesses[3], eval_outputs_objectnesses[4], eval_outputs_objectnesses[5] = \
                eval_outputs_objectnesses[5], eval_outputs_objectnesses[4], eval_outputs_objectnesses[3], eval_outputs_objectnesses[2], eval_outputs_objectnesses[1], eval_outputs_objectnesses[0]


        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1], 'pred_obj':outputs_objectness[-1], 'eval_obj': eval_outputs_objectness[-1]} 
        
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord, outputs_objectnesses, eval_outputs_objectness)

        if self.two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            out['enc_outputs'] = {'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord}
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, objectness, eval_objectness):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_obj': b, 'pred_boxes': c, 'eval_obj': d}
                for a, b, c, d in zip(outputs_class[:-1], objectness[:-1], outputs_coord[:-1], eval_objectness[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, losses, invalid_cls_logits, hidden_dim, focal_alpha=0.25, \
                empty_weight=0.1, temperature=1, geometirc_mean_alpha=0.5, iou_thre = 0.5, obj_thre=0.7, sns_thre = 0.3, unmatched_boxes=True, lr_decline_p=0.5):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.unmatched_boxes = unmatched_boxes
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.geometirc_mean_alpha = geometirc_mean_alpha
        self.obj_thre = obj_thre
        self.iou_thre = iou_thre 
        self.sns_thre = sns_thre
        self.empty_weight=empty_weight
        self.invalid_cls_logits = invalid_cls_logits
        self.max_prob=0.9
        self.lr_decline_p=lr_decline_p
        self.temperature = temperature
        self.hidden_dim = hidden_dim
        
    def sum_geometric_sequence(self, p):
        return  (1 - p**6) / (1 - p)

    def loss_labels(self, outputs, targets, indices, num_boxes, num_pseudo_boxes, lvl,  owod_targets, owod_indices, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        temp_src_logits = outputs['pred_logits'].clone()
        temp_src_logits[:,:, self.invalid_cls_logits] = -10e10
        src_logits = temp_src_logits
        
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        
        target_classes = torch.full(src_logits.shape[:2], self.num_classes-1, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2]],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)

        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        eval_pseudo_pred_obj = outputs["eval_obj"]
        eval_pseudo_obj_prob = torch.exp(-self.temperature * eval_pseudo_pred_obj)
        eval_pseudo_obj_prob = eval_pseudo_obj_prob.detach() 

        loss_ce =  sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha,
                                     num_classes=self.num_classes, empty_weight=self.empty_weight) * src_logits.shape[1]
        
        if self.lr_decline_p == 0:
            losses = {'loss_ce': loss_ce}
        else:
            sum_geo = self.sum_geometric_sequence(self.lr_decline_p)
            lr_decay = (6 / sum_geo) * self.lr_decline_p **(5-lvl)
            losses = {'loss_ce': loss_ce * lr_decay}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes, num_pseudo_boxes, lvl, owod_targets, owod_indices):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses
    

    def loss_boxes(self, outputs, targets, indices, num_boxes, num_pseudo_boxes, lvl, owod_targets, owod_indices):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes

        owod_idx = self._get_src_permutation_idx(owod_indices)
        owod_src_boxes = outputs['pred_boxes'][owod_idx]
        pseudo_target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(owod_targets, owod_indices)], dim=0)
        
        if len(pseudo_target_boxes) == 0:
            losses['pseudo_loss_bbox'] = losses['loss_bbox'] * 0.
            losses['pseudo_loss_giou'] = losses['loss_giou'] * 0.
        else:
            temperature = 1 / self.hidden_dim
            pseudo_eval_obj = outputs["eval_obj"][owod_idx]
            obj_weights = torch.exp(-temperature * pseudo_eval_obj).detach()
            pseudo_loss_bbox = obj_weights[:, None] * F.l1_loss(owod_src_boxes, pseudo_target_boxes, reduction='none')
            losses['pseudo_loss_bbox'] = pseudo_loss_bbox.sum() / num_pseudo_boxes

            pseudo_loss_giou =  (1 - torch.diag(box_ops.generalized_box_iou(
                box_ops.box_cxcywh_to_xyxy(owod_src_boxes),
                box_ops.box_cxcywh_to_xyxy(pseudo_target_boxes)))) 
            losses['pseudo_loss_giou'] = pseudo_loss_giou.sum() / num_pseudo_boxes

        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes, num_pseudo_boxes, lvl,  owod_targets, owod_indices):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        src_masks = outputs["pred_masks"]

        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list([t["masks"] for t in targets]).decompose()
        target_masks = target_masks.to(src_masks)

        src_masks = src_masks[src_idx]
        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks[tgt_idx].flatten(1)

        losses = {
            "loss_mask": seg_sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses
    

    def loss_obj_likelihood(self, outputs, targets, indices, num_boxes, num_pseudo_boxes, lvl, owod_targets, owod_indices):
        assert "pred_obj" in outputs
        temperature = 1 / self.hidden_dim
        idx = self._get_src_permutation_idx(indices)
        owod_idx = self._get_src_permutation_idx(owod_indices)
        # unmatch indices
        queries = torch.arange(outputs['pred_obj'].shape[1])
        unmatched_indices = []
        for i in range(len(indices)):
            combined = torch.cat((queries, self._get_src_single_permutation_idx(indices[i], i)[-1])) ## need to fix the indexing
            uniques, counts = combined.unique(return_counts=True)
            unmatched_indices.append(uniques[counts == 1])

        # positive samples
        pred_obj = outputs["pred_obj"][idx]
        
        # negative samples
        region_boxes_list = [t['segment_region'][self._filter_invalid(t['segment_region'])] for t in targets]
        neg_boxes_list = [t[i] for t, i in zip(outputs['pred_boxes'], unmatched_indices)]
        neg_obj_list = [t[i] for t, i in zip(outputs['pred_obj'], unmatched_indices)]
        neg_mask_list = []
        for i, (region_boxes, neg_boxes) in enumerate(zip(region_boxes_list, neg_boxes_list)):
            img_h, img_w = targets[i]['size']
            region_boxes = box_ops.box_cxcywh_to_xyxy(region_boxes)
            region_boxes = region_boxes * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32).to(region_boxes.device)
            neg_boxes = box_ops.box_cxcywh_to_xyxy(neg_boxes)
            neg_boxes = neg_boxes * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32).to(neg_boxes.device)
            if len(region_boxes)!=0:
                neg_ious, _ = box_ops.jaccard(neg_boxes, region_boxes).max(-1)
                neg_mask = neg_ious<self.sns_thre
            else:
                neg_mask = torch.ones((len(neg_boxes)), dtype=bool)
            neg_mask_list.append(neg_mask)
        neg_obj = torch.cat([b[m] for b, m in zip(neg_obj_list, neg_mask_list)], dim=0)

        # positve loss
        pred_obj_prob = torch.exp(-temperature * pred_obj)
        pred_obj_prob = torch.clamp(pred_obj_prob, max=self.max_prob)
        loss_pos_obj_ll = (- torch.log(pred_obj_prob)).sum() / num_boxes
        # negative loss
        neg_obj_prob = torch.exp(-temperature * neg_obj)
        # neg_obj_prob = torch.clamp(neg_obj_prob, min=self.min_prob)
        if len(neg_obj_prob) != 0:
            loss_neg_obj_ll = (- torch.log(1 - neg_obj_prob)).sum() / len(neg_obj_prob)
        else:
            loss_neg_obj_ll = loss_pos_obj_ll * 0.

        # loss_neg_obj_ll = loss_pos_obj_ll * 0. # w/o sns

        if owod_targets != None:
            eval_pseudo_pred_obj = outputs["eval_obj"][owod_idx]
            eval_pseudo_obj_prob = torch.exp(-temperature * eval_pseudo_pred_obj)
            obj_weights = eval_pseudo_obj_prob.detach()
            valid_mask = obj_weights > self.obj_thre
            owod_idx = [owod_idx[0][valid_mask], owod_idx[1][valid_mask]]
            pseudo_pred_obj = outputs["pred_obj"][owod_idx]
            pseudo_pred_obj_prob = torch.exp(-temperature * pseudo_pred_obj)
            pseudo_pred_obj_prob = torch.clamp(pseudo_pred_obj_prob, max=self.max_prob)
            loss_pseudo_obj_ll = (- torch.log(pseudo_pred_obj_prob)).sum() / num_pseudo_boxes
        else:
            loss_pseudo_obj_ll = loss_pos_obj_ll * 0.

        if self.lr_decline_p == 0:
            return  {'loss_pos_obj_ll': loss_pos_obj_ll, 'loss_neg_obj_ll': loss_neg_obj_ll, 'loss_pseudo_obj_ll': loss_pseudo_obj_ll}
        else:
            sum_geo = self.sum_geometric_sequence(self.lr_decline_p)
            lr_decay = (6 / sum_geo) * self.lr_decline_p **(5-lvl)
            return  {'loss_pos_obj_ll': loss_pos_obj_ll* lr_decay, 'loss_neg_obj_ll': loss_neg_obj_ll* lr_decay, 'loss_pseudo_obj_ll': loss_pseudo_obj_ll*lr_decay}


    def _get_src_single_permutation_idx(self, indices, index):
        ## Only need the src query index selection from this function for attention feature selection
        batch_idx = [torch.full_like(src, i) for i, src in enumerate(indices)][0]
        src_idx = indices[0]
        return batch_idx, src_idx

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, num_pseudo_boxes, lvl,  owod_targets, owod_indices, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'obj_likelihood': self.loss_obj_likelihood,
            'masks': self.loss_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, num_pseudo_boxes, lvl, owod_targets, owod_indices, **kwargs)

    def _filter_invalid(self, boxes):
        return (boxes[:, 2] > 0) & (boxes[:, 3] > 0)

    def forward(self, samples, outputs, targets, epoch):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)
        owod_targets = copy.deepcopy(targets)
        owod_indices = copy.deepcopy(indices)
        owod_outputs = outputs_without_aux.copy()
        owod_device = owod_outputs["pred_boxes"].device

        if epoch > 10 and self.unmatched_boxes:
            ## get pseudo unmatched boxes from this section
            eval_objectness = torch.exp(-self.temperature * owod_outputs["eval_obj"])
            queries = torch.arange(owod_outputs['pred_logits'].shape[1])
            for i in range(len(indices)):
                combined = torch.cat((queries, self._get_src_single_permutation_idx(indices[i], i)[-1])) ## need to fix the indexing
                uniques, counts = combined.unique(return_counts=True)
                unmatched_indices = uniques[counts == 1]
                boxes = owod_outputs['pred_boxes'][i] #[unmatched_indices,:]

                img_h, img_w = targets[i]['size']
                boxes = box_ops.box_cxcywh_to_xyxy(boxes)
                boxes = boxes * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32).to(owod_device)
                target_bboxes = box_ops.box_cxcywh_to_xyxy(targets[i]["boxes"])
                target_bboxes = target_bboxes * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32).to(owod_device)

                objectnesses = torch.zeros(queries.shape[0]).to(boxes)
                objectnesses[unmatched_indices] = eval_objectness[i][unmatched_indices]
                if "segment_region" in targets[i]:
                    segment_region = copy.deepcopy(targets[i]['segment_region'])
                    segment_region = segment_region[self._filter_invalid(segment_region)]
                    segment_region = box_ops.box_cxcywh_to_xyxy(segment_region)
                    segment_region = segment_region * torch.tensor([img_w, img_h, img_w, img_h],dtype=torch.float32).to(owod_device)
                else:
                    segment_region = torch.tensor([])
                if len(segment_region) != 0 and len(target_bboxes)!=0:
                    iou, _ = box_ops.jaccard(segment_region, target_bboxes).max(dim=1)
                    segment_region = segment_region[iou < 0.7] # no overlap with GT
                if len(segment_region) != 0:
                    ss_iou, ss_ind = box_ops.jaccard(boxes, segment_region).max(dim=1) # compute iou with segment_region
                    criterion = ss_iou  
                    # criterion = objectnesses**self.geometirc_mean_alpha * ss_iou**(1-self.geometirc_mean_alpha) # geometirc_mean
                    sorted_criterion, sorted_indices = torch.sort(criterion, descending=True, dim=-1)
                    thre_ind = torch.where(sorted_criterion > self.iou_thre)[0]
                    if len(thre_ind) != 0:
                        # remove depulicate segment_region
                        match_ind = sorted_indices[thre_ind].to(owod_device)
                        select_ss_ind = ss_ind[sorted_indices][thre_ind].to(owod_device)
                        unique_ss_ind = torch.unique(select_ss_ind)
                        filter_select_inds = torch.tensor([match_ind[torch.where(select_ss_ind==ind)[0][0].unsqueeze(0)] for ind in unique_ss_ind])
                        filter_select_ss_inds = torch.tensor([select_ss_ind[torch.where(select_ss_ind==ind)[0][0].unsqueeze(0)] for ind in unique_ss_ind])
                        topk_segment_region = segment_region[filter_select_ss_inds] / torch.tensor([img_w, img_h, img_w, img_h],dtype=torch.float32).to(owod_device)
                        topk_segment_region = box_ops.box_xyxy_to_cxcywh(topk_segment_region)
                        unk_label = torch.as_tensor([self.num_classes-1], device=owod_device)
                        owod_targets[i]['labels'] = unk_label.repeat_interleave(len(filter_select_inds))
                        owod_targets[i]['boxes'] = topk_segment_region
                        owod_indices[i] = (filter_select_inds, (owod_targets[i]['labels'] == unk_label).nonzero(as_tuple=True)[0].cpu())
                    else:
                        owod_targets[i]['labels'] = torch.tensor([], dtype=torch.int).to(owod_device)
                        owod_targets[i]['boxes'] = torch.tensor([]).to(owod_device)
                        owod_indices[i] = (torch.tensor([], dtype=torch.long).cpu(), torch.tensor([], dtype=torch.long).cpu())
                else:
                    owod_targets[i]['labels'] = torch.tensor([], dtype=torch.int).to(owod_device)
                    owod_targets[i]['boxes'] = torch.tensor([]).to(owod_device)
                    owod_indices[i] = (torch.tensor([], dtype=torch.long).cpu(), torch.tensor([], dtype=torch.long).cpu())
        else:
            for i in range(len(owod_targets)):
                owod_targets[i]['labels'] = torch.tensor([], dtype=torch.int).to(owod_device)
                owod_targets[i]['boxes'] = torch.tensor([]).to(owod_device)
                owod_indices[i] = (torch.tensor([], dtype=torch.long).cpu(), torch.tensor([], dtype=torch.long).cpu())

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        num_pseudo_boxes = sum(len(t["labels"]) for t in owod_targets)
        num_pseudo_boxes = torch.as_tensor([num_pseudo_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_pseudo_boxes)
        num_pseudo_boxes = torch.clamp(num_pseudo_boxes / get_world_size(), min=1).item()
        

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, num_pseudo_boxes, 5,  owod_targets, owod_indices, **kwargs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for p, aux_outputs in enumerate(outputs['aux_outputs']):
                # if p != 0: # keep the indices are the same for the first and the last decoder layer
                #     indices = self.matcher(aux_outputs, targets)
                indices = self.matcher(aux_outputs, targets)
                owod_targets = copy.deepcopy(targets)
                owod_indices = copy.deepcopy(indices)
                
                owod_outputs = outputs_without_aux.copy()
                owod_device = owod_outputs["pred_boxes"].device

                if epoch > 10 and self.unmatched_boxes:
                    ## get pseudo unmatched boxes from this section
                    eval_objectness = torch.exp(-self.temperature * owod_outputs["eval_obj"])
                    queries = torch.arange(owod_outputs['pred_logits'].shape[1])
                    for i in range(len(indices)):
                        combined = torch.cat((queries, self._get_src_single_permutation_idx(indices[i], i)[-1])) ## need to fix the indexing
                        uniques, counts = combined.unique(return_counts=True)
                        unmatched_indices = uniques[counts == 1]
                        boxes = owod_outputs['pred_boxes'][i] #[unmatched_indices,:]

                        img_h, img_w = targets[i]['size']
                        boxes = box_ops.box_cxcywh_to_xyxy(boxes)
                        boxes = boxes * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32).to(owod_device)
                        target_bboxes = box_ops.box_cxcywh_to_xyxy(targets[i]["boxes"])
                        target_bboxes = target_bboxes * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32).to(owod_device)

                        objectnesses = torch.zeros(queries.shape[0]).to(boxes)
                        objectnesses[unmatched_indices] = eval_objectness[i][unmatched_indices]
                        if "segment_region" in targets[i]:
                            segment_region = copy.deepcopy(targets[i]['segment_region'])
                            segment_region = segment_region[self._filter_invalid(segment_region)]
                            segment_region = box_ops.box_cxcywh_to_xyxy(segment_region)
                            segment_region = segment_region * torch.tensor([img_w, img_h, img_w, img_h],dtype=torch.float32).to(owod_device)
                        else:
                            segment_region = torch.tensor([])
                        if len(segment_region) != 0 and len(target_bboxes)!=0:
                            iou, _ = box_ops.jaccard(segment_region, target_bboxes).max(dim=1)
                            segment_region = segment_region[iou < 0.7] # no overlap with GT
                        if len(segment_region) != 0:
                            ss_iou, ss_ind = box_ops.jaccard(boxes, segment_region).max(dim=1) # compute iou with segment_region
                            criterion = ss_iou  
                            # criterion = objectnesses**self.geometirc_mean_alpha * ss_iou**(1-self.geometirc_mean_alpha) # geometirc_mean
                            sorted_criterion, sorted_indices = torch.sort(criterion, descending=True, dim=-1)
                            thre_ind = torch.where(sorted_criterion > self.iou_thre)[0]
                            if len(thre_ind) != 0:
                                # remove depulicate segment_region
                                match_ind = sorted_indices[thre_ind].to(owod_device)
                                select_ss_ind = ss_ind[sorted_indices][thre_ind].to(owod_device)
                                unique_ss_ind = torch.unique(select_ss_ind)
                                filter_select_inds = torch.tensor([match_ind[torch.where(select_ss_ind==ind)[0][0].unsqueeze(0)] for ind in unique_ss_ind])
                                filter_select_ss_inds = torch.tensor([select_ss_ind[torch.where(select_ss_ind==ind)[0][0].unsqueeze(0)] for ind in unique_ss_ind])
                                topk_segment_region = segment_region[filter_select_ss_inds] / torch.tensor([img_w, img_h, img_w, img_h],dtype=torch.float32).to(owod_device)
                                topk_segment_region = box_ops.box_xyxy_to_cxcywh(topk_segment_region)
                                unk_label = torch.as_tensor([self.num_classes-1], device=owod_device)
                                owod_targets[i]['labels'] = unk_label.repeat_interleave(len(filter_select_inds))
                                owod_targets[i]['boxes'] = topk_segment_region
                                owod_indices[i] = (filter_select_inds, (owod_targets[i]['labels'] == unk_label).nonzero(as_tuple=True)[0].cpu())
                            else:
                                owod_targets[i]['labels'] = torch.tensor([], dtype=torch.int).to(owod_device)
                                owod_targets[i]['boxes'] = torch.tensor([]).to(owod_device)
                                owod_indices[i] = (torch.tensor([], dtype=torch.long).cpu(), torch.tensor([], dtype=torch.long).cpu())
                        else:
                            owod_targets[i]['labels'] = torch.tensor([], dtype=torch.int).to(owod_device)
                            owod_targets[i]['boxes'] = torch.tensor([]).to(owod_device)
                            owod_indices[i] = (torch.tensor([], dtype=torch.long).cpu(), torch.tensor([], dtype=torch.long).cpu())
                else:
                    for i in range(len(owod_targets)):
                        owod_targets[i]['labels'] = torch.tensor([], dtype=torch.int).to(owod_device)
                        owod_targets[i]['boxes'] = torch.tensor([]).to(owod_device)
                        owod_indices[i] = (torch.tensor([], dtype=torch.long).cpu(), torch.tensor([], dtype=torch.long).cpu())
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs['log'] = False
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, num_pseudo_boxes, p,  owod_targets, owod_indices, **kwargs)
                    l_dict = {k + f'_{p}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets:
                bt['labels'] = torch.zeros_like(bt['labels'])
            indices = self.matcher(enc_outputs, bin_targets)
            for loss in self.losses:
                if loss == 'masks':
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs['log'] = False
                l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices, num_boxes, **kwargs)
                l_dict = {k + f'_enc': v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    def __init__(self, invalid_cls_logits, temperature=1, pred_per_im=100, post_geometirc_mean_alpha=1.):
        super().__init__()
        self.temperature=temperature
        self.invalid_cls_logits=invalid_cls_logits
        self.pred_per_im=pred_per_im
        self.post_geometirc_mean_alpha=post_geometirc_mean_alpha

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """        
        out_logits, pred_obj, out_bbox = outputs['pred_logits'], outputs['pred_obj'], outputs['pred_boxes']
        out_logits[:,:, self.invalid_cls_logits] = -10e10
        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2
        obj_prob = torch.exp(-self.temperature*pred_obj).unsqueeze(-1)
        if self.post_geometirc_mean_alpha == 1.:
            prob = obj_prob * out_logits.sigmoid()
        else:
            prob = obj_prob**self.post_geometirc_mean_alpha * out_logits.sigmoid()**(1-self.post_geometirc_mean_alpha)
        
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), self.pred_per_im, dim=1)
        scores = topk_values
        
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]
        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]
        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
    
    
class ExemplarSelection(nn.Module):
    def __init__(self, args, num_classes, matcher, invalid_cls_logits, temperature=1):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.num_seen_classes = args.PREV_INTRODUCED_CLS + args.CUR_INTRODUCED_CLS
        self.invalid_cls_logits=invalid_cls_logits
        self.temperature=temperature
        print(f'running with exemplar_replay_selection')   
              
            
    def calc_energy_per_image(self, outputs, targets, indices):
        out_logits, pred_obj = outputs['pred_logits'], outputs['pred_obj']
        out_logits[:,:, self.invalid_cls_logits] = -10e10

        obj_prob = torch.exp(-self.temperature*pred_obj).unsqueeze(-1)
        prob = obj_prob * out_logits.sigmoid()
        image_sorted_scores={}
        for i in range(len(targets)):
            image_sorted_scores[''.join([chr(int(c)) for c in targets[i]['org_image_id']])] = {'labels':targets[i]['labels'].cpu().numpy(),"scores": prob[i,indices[i][0],targets[i]['labels']].detach().cpu().numpy()}
        return [image_sorted_scores]

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def forward(self, samples, outputs, targets):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs' and k !='pred_obj'}
        indices = self.matcher(outputs_without_aux, targets)       
        return self.calc_energy_per_image(outputs, targets, indices)


def build(args):
    num_classes = args.num_classes
    invalid_cls_logits = list(range(args.PREV_INTRODUCED_CLS+args.CUR_INTRODUCED_CLS, num_classes-1))
    print("Invalid class range: " + str(invalid_cls_logits))
    
    device = torch.device(args.device)
    
    backbone = build_backbone(args)
    transformer = build_deforamble_transformer(args)
    
    model = DeformableDETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels,
        aux_loss=args.aux_loss,
        with_box_refine=args.with_box_refine,
        two_stage=args.two_stage,
    )
    
    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
        
    matcher = build_matcher(args)
    weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_bbox': args.bbox_loss_coef, 'loss_giou': args.giou_loss_coef, 'loss_pos_obj_ll': args.obj_loss_coef, 'loss_neg_obj_ll': args.obj_loss_coef,\
                 'loss_pseudo_obj_ll': args.pseudo_obj_loss_coef, 'pseudo_loss_bbox': args.pseudo_bbox_loss_coef, 'pseudo_loss_giou': args.pseudo_giou_loss_coef}
    
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality','obj_likelihood']
    if args.masks:
        losses += ["masks"]

        
    criterion = SetCriterion(num_classes, matcher, weight_dict, losses, invalid_cls_logits, args.hidden_dim, focal_alpha=args.focal_alpha, \
                            temperature=args.pseudo_obj_temp/args.hidden_dim, geometirc_mean_alpha=args.geometirc_mean_alpha, \
                            iou_thre = args.iou_thre, obj_thre=args.obj_thre, sns_thre=args.sns_thre, unmatched_boxes=args.unmatched_boxes, lr_decline_p=args.lr_decline_p)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess(invalid_cls_logits, temperature=args.obj_temp/args.hidden_dim, post_geometirc_mean_alpha=args.post_geometirc_mean_alpha)}
    exemplar_selection = ExemplarSelection(args, num_classes, matcher, invalid_cls_logits, temperature=args.obj_temp/args.hidden_dim)
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, postprocessors, exemplar_selection
