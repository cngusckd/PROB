# ------------------------------------------------------------------------
# OW-DETR: Open-world Detection Transformer
# Akshita Gupta^, Sanath Narayan^, K J Joseph, Salman Khan, Fahad Shahbaz Khan, Mubarak Shah
# https://arxiv.org/pdf/2112.01513.pdf
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------

def build_model(args, mode='owdetr'):
    if 'prob' == mode:
        from .prob_deformable_detr import build
    elif 'lite-prob' == mode:
        from .prob_dino_lite import build_dino as build
    else:
        from .deformable_detr import build
    return build(args)