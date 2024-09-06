# ------------------------------------------------------------------------
# Modified from Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# -----------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
 
"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable
 
import torch
import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.open_world_eval import OWEvaluator
from datasets.panoptic_eval import PanopticEvaluator
from datasets.data_prefetcher import data_prefetcher
from util.box_ops import box_xyxy_to_cxcywh, box_cxcywh_to_xyxy, jaccard, nms
from util.plot_utils import plot_prediction, plot_prediction_GT, rescale_bboxes
import matplotlib.pyplot as plt
from copy import deepcopy


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, nc_epoch: int, max_norm: float = 0, wandb: object = None):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    prefetcher = data_prefetcher(data_loader, device, prefetch=True)
    samples, targets = prefetcher.next()

    for _ in metric_logger.log_every(range(len(data_loader)), print_freq, header):
        outputs = model(samples)
        # loss_dict = criterion(outputs, targets) # prob
        loss_dict = criterion(samples, outputs, targets, epoch)  # SGOD
        weight_dict = deepcopy(criterion.weight_dict)
        
        ## condition for starting nc loss computation after certain epoch so that the F_cls branch has the time
        ## to learn the within classes seperation.
        if epoch < nc_epoch: 
            for k,v in weight_dict.items():
                if 'NC' in k:
                    weight_dict[k] = 0
         
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        # reduce losses over all GPUs for logging purposes

        loss_dict_reduced = utils.reduce_dict(loss_dict)
        ## Just printing NOt affectin gin loss function
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
 
        loss_value = losses_reduced_scaled.item()
 
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)
 
        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        optimizer.step()
        
        if wandb is not None:
            wandb.log({"total_loss":loss_value})
            wandb.log(loss_dict_reduced_scaled)
            wandb.log(loss_dict_reduced_unscaled)
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)
        
        samples, targets = prefetcher.next()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

## ORIGINAL FUNCTION
@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, args):
    model.eval()   
            
    criterion.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = OWEvaluator(base_ds, iou_types, args=args)
 
    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )
    # from util import box_ops
    # import torch
    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs = model(samples)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        # test U-Recall of SAM
        # for i in range(len(targets)):
        #     # boxes = box_ops.box_cxcywh_to_xyxy(targets[i]['segment_region'])
        #     # img_h, img_w = orig_target_sizes[i][None].unbind(1)
        #     # scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        #     # boxes = boxes * scale_fct[:, None, :]
        #     # nq = boxes.shape[1]
        #     # results[i]["boxes"] = boxes[0]
        #     # results[i]["labels"] = (torch.ones((nq), dtype=torch.int64) * 80).to(boxes[0].device)
        #     results[i]["labels"] = (torch.ones((results[i]["boxes"].shape[0]), dtype=torch.int64) * 80).to(results[i]["boxes"].device)
        #     # results[i]["scores"] = (torch.ones((nq), dtype=torch.float32) * 0.5).to(boxes[0].device)
        #     # print("results:", results)
 
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)
 
        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name
 
            panoptic_evaluator.update(res_pano)
 
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    # print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()
 
    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        res = coco_evaluator.summarize()
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    stats['metrics']=res
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]
    return stats, coco_evaluator
 
    
@torch.no_grad()
def get_exemplar_replay(model, exemplar_selection, device, data_loader):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = '[ExempReplay]'
    print_freq = 10
    prefetcher = data_prefetcher(data_loader, device, prefetch=True)
    samples, targets = prefetcher.next()
    image_sorted_scores_reduced={}
    for _ in metric_logger.log_every(range(len(data_loader)), print_freq, header):
        outputs = model(samples)
        image_sorted_scores = exemplar_selection(samples, outputs, targets)
        for i in utils.combine_dict(image_sorted_scores):
            image_sorted_scores_reduced.update(i[0])
            
        metric_logger.update(loss=len(image_sorted_scores_reduced.keys()))
        samples, targets = prefetcher.next()
        
    print(f'found a total of {len(image_sorted_scores_reduced.keys())} images')
    return image_sorted_scores_reduced

@torch.no_grad()
def viz(model, criterion, postprocessors, data_loader, base_ds, device, output_dir):
    known_viz_thre = 0.3
    unknown_viz_thre = 0.5
    import numpy as np
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    criterion.eval()
 
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
 
    for samples, targets in data_loader:
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        top_k = len(targets[0]['boxes'])
        outputs = model(samples)
        image_size = samples.tensors.shape
        w, h = image_size[-2], image_size[-1]
        target_sizes = torch.zeros(1, 2).to(outputs['pred_logits'].device)
        target_sizes[0][0] = w
        target_sizes[0][1] = h

        results = postprocessors['bbox'](outputs, target_sizes)[0]

        predictied_scores = results['scores']
        predictied_boxes = results['boxes']
        predictied_labels = results['labels']
        known_mask = predictied_labels != 80
  
        known_viz_mask = predictied_scores[known_mask] > known_viz_thre
        k_predictied_scores = predictied_scores[known_mask][known_viz_mask]
        k_predictied_boxes = predictied_boxes[known_mask][known_viz_mask]
        k_predictied_labels = predictied_labels[known_mask][known_viz_mask]

        unknown_viz_mask = predictied_scores[~known_mask] > unknown_viz_thre
        uk_predictied_scores = predictied_scores[~known_mask][unknown_viz_mask]
        uk_predictied_boxes = predictied_boxes[~known_mask][unknown_viz_mask]
        uk_predictied_labels = predictied_labels[~known_mask][unknown_viz_mask]

        # remove unknown boxes overlapping with known boxes
        if len(k_predictied_boxes)!=0:
            ss_iou, _ = jaccard(uk_predictied_boxes, k_predictied_boxes).max(dim=1) # compute iou with groundtruths
            uk_predictied_boxes = uk_predictied_boxes[ss_iou < 0.3] # no overlap with known predictions
            uk_predictied_scores = uk_predictied_scores[ss_iou < 0.3]
            uk_predictied_labels = uk_predictied_labels[ss_iou < 0.3]
        
        # remove unknown boxes overlapping with themselves
        nms_pick = nms(uk_predictied_boxes, uk_predictied_scores) # compute iou with groundtruths

        uk_predictied_boxes = uk_predictied_boxes[nms_pick,:] # nms
        uk_predictied_scores = uk_predictied_scores[nms_pick]
        uk_predictied_labels = uk_predictied_labels[nms_pick]

        # target_boxes = rescale_bboxes(targets[0]['boxes'], list(samples.tensors[0:1].shape[2:])[::-1])
        # uk_iou, uk_ind = jaccard(uk_predictied_boxes, target_boxes).max(dim=1) # compute iou with groundtruths
        # uk_predictied_boxes = uk_predictied_boxes[uk_iou > 0.3] # no overlap with GT
        # uk_predictied_scores = uk_predictied_scores[uk_iou > 0.3]
        # uk_predictied_labels = uk_predictied_labels[uk_iou > 0.3]

        predictied_scores = torch.cat([k_predictied_scores, uk_predictied_scores])
        predictied_boxes = torch.cat([k_predictied_boxes, uk_predictied_boxes])
        predictied_labels = torch.cat([k_predictied_labels, uk_predictied_labels])

        fig, ax = plt.subplots(1, 1, figsize=(10,3), dpi=200)

        # # Known pred results
        # plot_prediction(samples.tensors[0:1], k_predictied_scores, k_predictied_boxes, k_predictied_labels, ax[0], plot_prob=False)
        # ax[0].set_title('Known prediction (Ours)')
        # # Unknown pred results
        # plot_prediction(samples.tensors[0:1], uk_predictied_scores, uk_predictied_boxes, uk_predictied_labels, ax[1], plot_prob=False)
        # ax[1].set_title('Unknown prediction (Ours)')
        # # GT Results
        # gt_boxes = targets[0]['boxes']
        # gt_labels = targets[0]['labels']
        # gt_boxes = targets[0]['segment_region']
        # gt_labels = torch.ones((len(gt_boxes)), dtype=torch.int) * 80
        # gt_known_mask = gt_labels != 80
        # gt_known_mask[gt_known_mask==True]=False
        # plot_prediction_GT(samples.tensors[0:1], gt_boxes[gt_known_mask], gt_labels[gt_known_mask], ax, plot_prob=False)
        # plot_prediction_GT(samples.tensors[0:1], gt_boxes, gt_labels, ax, plot_prob=False)
        # ax[2].set_title('GT')

        # draw results
        plot_prediction(samples.tensors[0:1], predictied_scores, predictied_boxes, predictied_labels, ax, plot_prob=False)
        ax.set_title('SAM-OWOD')

        ax.set_aspect('equal')
        ax.set_axis_off()

        # for i in range(2):
        #     ax[i].set_aspect('equal')
        #     ax[i].set_axis_off()

        plt.savefig(os.path.join(output_dir, f'img_{int(targets[0]["image_id"][0])}.jpg'))
