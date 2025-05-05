# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import torch
import torch.nn as nn

class SkyboxNull(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

    def forward(self, skyenc_output, network_output):
        return network_output

    @staticmethod
    def sample_batch(pose_matrices, intrinsics, network_output, batchidx=0, scale_idx=None):
        pass

    def encode_sky_feature(self, batch, imgenc_output):
        return {}

