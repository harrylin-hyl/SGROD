# ------------------------------------------------------------------------
# OW-DETR: Open-world Detection Transformer
# Akshita Gupta^, Sanath Narayan^, K J Joseph, Salman Khan, Fahad Shahbaz Khan, Mubarak Shah
# https://arxiv.org/pdf/2112.01513.pdf
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------

def build_model(args, mode='owdetr'):
    if mode == 'prob':
        from .prob_deformable_detr import build
    elif mode == 'prob_sam':
        from prob_sam_deformable_detr import build
    elif mode =='sgrod':
        from .sgrod_deformable_detr import build
    else:
        from .deformable_owdetr import build
    return build(args)