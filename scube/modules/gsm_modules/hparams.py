# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

def hparams_handler(hparams):  
    if not hasattr(hparams, 'keep_surface_voxel'):
        hparams.keep_surface_voxel = False
    if not hasattr(hparams, 'use_high_res_grid_for_alpha_mask'):
        hparams.use_high_res_grid_for_alpha_mask = False

    if not hasattr(hparams, "use_skybox"):
        hparams.use_skybox = True
    if not hasattr(hparams, "skybox_type"):
        hparams.skybox_type = "panorama_full"
    if not hasattr(hparams, "skybox_resolution"):
        hparams.skybox_resolution = 1024 # pixel

    hparams.with_render_branch = False
    if not hasattr(hparams.supervision, 'render_weight'):
        hparams.supervision.render_weight = 0.0
    if hparams.supervision.render_weight > 0:
        hparams.with_render_branch = True
        if not hasattr(hparams, 'perceptual_weight'):
            hparams.perceptual_weight = 0.0

    if not hasattr(hparams, 'pixel_loss'):
        hparams.pixel_loss = 'l1'

    if not hasattr(hparams.supervision, 'depth_weight'):
        hparams.supervision.depth_weight = 0.0

    if hparams.supervision.depth_weight == 0:
        hparams.use_sup_depth = False
    else:
        hparams.use_sup_depth = True
        assert hasattr(hparams, 'sup_depth_type'), \
            'must specify sup_depth_type, can be "lidar_depth" or "rectified_metric3d_depth" or "voxel_depth"' 
        
    if not hasattr(hparams, 'use_ssim_loss'):
        hparams.use_ssim_loss = False
    if not hasattr(hparams, 'gs_free_space'):
        hparams.gs_free_space = "tanh-3"
    if not hasattr(hparams, 'use_alex_metric'): # ! can set to default afterward
        hparams.use_alex_metric = False

    if not hasattr(hparams, 'render_alpha'):
        hparams.render_alpha = False
    if not hasattr(hparams, 'gt_alpha_from'):
        hparams.gt_alpha_from = 'grid'
    if not hasattr(hparams, 'only_sup_foreground'):
        hparams.only_sup_foreground = False
    if not hasattr(hparams, 'render_target_is_object'):
        hparams.render_target_is_object = True

    if not hasattr(hparams, 'gsplat_params'):
        hparams.gsplat_params = {'radius_clip': 0, 'rasterize_mode': 'classic'}
    if hasattr(hparams, 'feature_map_downsample'):
        print("feature_map_downsample is deprecated, use rasterizing_downsample instead")
        hparams.rasterizing_downsample = hparams.feature_map_downsample
    if not hasattr(hparams, 'rasterizing_downsample'):
        hparams.rasterizing_downsample = 1

    return hparams