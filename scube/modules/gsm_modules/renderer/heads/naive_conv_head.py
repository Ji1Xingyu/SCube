# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from abc import abstractmethod

class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x

class AdaptiveGroupNorm(TimestepBlock):
    """
    https://github.com/NVlabs/denoising-diffusion-gan/blob/6818ded6443e10ab69c7864745457ce391d4d883/score_sde/models/layerspp.py
    """
    def __init__(self, in_channels=128, num_groups=32, style_dim=32):
        super().__init__()

        self.norm = nn.GroupNorm(num_groups, in_channels, affine=False, eps=1e-6)
        self.style = nn.Linear(style_dim, in_channels * 2)

        self.style.bias.data[:in_channels] = 1
        self.style.bias.data[in_channels:] = 0

    def forward(self, input, style):
        style = self.style(style).unsqueeze(2).unsqueeze(3)
        gamma, beta = style.chunk(2, 1)
        out = self.norm(input)
        out = gamma * out + beta

        return out

class GroupNorm(nn.Module):
    def __init__(self, num_channels, num_groups=32):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, num_channels)
    
    def forward(self, x):
        return self.norm(x)

class NaiveConvHead(nn.Module):

    def set(self, property, config, default_value):
        if property in config:
            setattr(self, property, config.get(property))
        else:
            setattr(self, property, default_value)

    def __init__(self, config):
        super().__init__()

        self.config = config

        self.channel = int(512 * self.config.dec_channel_multiplier)
        input_chan_size = self.config.feature_size

        self.set('num_classes', self.config, 0)
        self.set('padding_mode', self.config, 'zeros')

        self.cam_dep_norm = False
        self.ind_rgb_multiplier, self.ind_rgb_bias = None, None
        
        self.cam_dep_norm = True
        self.ind_emb_dim = 32
        self.ind_emb = nn.Embedding(self.num_classes, self.ind_emb_dim)
        # norm_fn = AdaptiveGroupNorm
        norm_fn = GroupNorm
        while self.channel % 32 != 0:
            self.channel += 1

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_chan_size, self.channel, 3, 1, 1,  padding_mode=self.padding_mode),
            nn.LeakyReLU(0.2)
        )

        self.convs = nn.ModuleList()
        # assert(self.config.decoder_upsample_evenly)

        if self.config.num_gen_layers == 1:
            self.upsample_every = 1
            self.skip_first = False
        elif self.config.num_to_upsample == 1:  # True
            self.upsample_every = self.config.num_gen_layers  // 2
            self.skip_first = True
        elif self.config.num_to_upsample == 0:
            self.upsample_every = 111111
            self.skip_first = True
        else:
            self.upsample_every = self.config.num_gen_layers //  self.config.num_to_upsample
            self.skip_first = False
        for layer_idx in range(self.config.num_gen_layers):
            layers = [
                nn.Conv2d(self.channel, self.channel, 3, 1, 1, padding_mode=self.padding_mode),
                norm_fn(self.channel),
                nn.LeakyReLU(0.2)
            ]
            self.convs.append(TimestepEmbedSequential(*layers))
        
        self.last_conv = nn.Conv2d(self.channel, 3, 3, 1, 1)

    def forward(self, target_2d_features):
        emb = None
        # ''' get camera embeddings '''
        # B, N = scene_data.target_cam_classes.shape
        # emb = self.ind_emb(scene_data.target_cam_classes).reshape(B*N, -1)

        ''' get target_2d_features'''
        # BS, N, H, W, C = target_2d_features.shape
        # target_2d_features = target_2d_features.reshape(BS*N, H, W, C).transpose(1,3).transpose(2,3) # B, C, H, W

        target_2d_features = target_2d_features.transpose(1,3).transpose(2,3) # suppose input is B, H, W, C, output is B, C, H, W
        target_2d_features = target_2d_features.contiguous()

        ''' convolution on the 2d feature image '''
        num_upsampled = 0
        out = self.conv1(target_2d_features)
        for layer_idx in range(self.config.num_gen_layers): # 4
            out = self.convs[layer_idx](out, emb)
            if self.skip_first and layer_idx == 0:
                continue
            if layer_idx % self.upsample_every == 0 and num_upsampled < self.config.num_to_upsample:
                out = F.interpolate(out, (out.shape[2] * 2, out.shape[3] * 2), mode="nearest")
                num_upsampled += 1

        image = self.last_conv(out)
        return image.permute(0, 2, 3, 1)
    
    @property
    def last_layer(self):
        return self.last_conv.weight