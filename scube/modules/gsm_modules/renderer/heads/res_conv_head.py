# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import torch
import torch.nn as nn

from scube.modules.basic_modules import ResBlock

class ResConvHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_residual_block = config.n_residual_block
        self.n_fitler_list = config.n_filter_list
        self.n_upsample_list = config.n_upsample_list
        self.n_downsample_list = config.n_downsample_list

        self.model = nn.Sequential(
            *[
                ResBlock(
                    channels=self.n_fitler_list[i],
                    out_channels=self.n_fitler_list[i+1],
                    up=self.n_upsample_list[i],
                    down=self.n_downsample_list[i],
                    use_gn=False, 
                ) if self.n_upsample_list[i] else nn.Conv2d(
                    self.n_fitler_list[i],
                    self.n_fitler_list[i+1],
                    kernel_size=3,
                    stride=2 if self.n_downsample_list[i] else 1,
                    padding=1,
                )
                for i in range(len(self.n_fitler_list) - 1)
            ]
        )

    def forward(self, target_2d_feature):
        """
        Args:
            target_2d_feature: torch.Tensor, [B, H, W, C]
        
        Returns:
            torch.Tensor, [B, H, W, C']
        """
        return self.model(target_2d_feature.permute(0,3,1,2)).permute(0,2,3,1)