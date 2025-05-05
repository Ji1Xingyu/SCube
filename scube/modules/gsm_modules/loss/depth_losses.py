# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import torch

def affine_invariant_loss(preds, gts, masks):
    # preds, gts: [N, H, W]
    # gt_masks: [N, H, W]

    losses = []

    for pred, gt, mask in zip(preds, gts, masks):
        pred = pred.flatten()
        gt = gt.flatten()
        mask = mask.flatten() > 0

        pred = pred[mask]
        gt = gt[mask]
        
        pred_median = torch.median(pred)
        gt_median = torch.median(gt)

        gt_scale = (gt - gt_median).abs().mean()
        pred_scale = (pred - pred_median).abs().mean()

        gt_rescale = (gt - gt_median) / gt_scale
        pred_rescale = (pred - pred_median) / pred_scale

        loss = (gt_rescale - pred_rescale).abs().mean()
        losses.append(loss)

    return torch.stack(losses).mean()
