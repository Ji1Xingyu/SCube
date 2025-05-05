# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
import numpy as np
import fvdb
import fvdb.nn
import imageio.v3 as imageio
import pytorch_lightning as pl
import sys
from pathlib import Path
from dataclasses import dataclass

sys.path.append(Path(__file__).parent.parent.as_posix())

from fvdb import JaggedTensor, GridBatch
from fvdb.nn import VDBTensor
from pathlib import Path
from pycg import exp
from tqdm import tqdm
from scube.utils.common_util import batch2device, get_default_parser, create_model_from_args
from scube.utils.gaussian_util import save_splat_file_RGB
from scube.data.base import DatasetSpec as DS
from scube.data.base import list_collate
from scube.utils.voxel_util import offsreen_mesh_renderer_for_vae

fvdb.nn.SparseConv3d.backend = 'igemm_mode1'

def save_pred_gt(res_feature_set, out_vdb_tensor, batch, batch_idx, saving_dir, save_render=False):
    grid = out_vdb_tensor.grid
    semantic = res_feature_set.semantic_features[-1].jdata.argmax(dim=-1)

    save_dict = {'points': grid.to('cpu'), 'semantics': semantic.to('cpu')}
    torch.save(save_dict, saving_dir / f"{batch_idx}.pt")

    gt_save_dict = {'points': batch[DS.INPUT_PC].to('cpu'), 
                    'semantics': batch[DS.GT_SEMANTIC][0].to('cpu')}
    torch.save(gt_save_dict, saving_dir / f"{batch_idx}_gt.pt")

    print(f"Save to {saving_dir / f'{batch_idx}.pt'}")
    print(f"Save to {saving_dir / f'{batch_idx}_gt.pt'}")

    if save_render:
        render_file_path = saving_dir / f"{batch_idx}_render.jpg"
        grid_semantic_pairs = [
            (grid, semantic),
            (batch[DS.INPUT_PC], batch[DS.GT_SEMANTIC][0])
        ]
        rendered_image = offsreen_mesh_renderer_for_vae(
            grid_semantic_pairs,
            default_camera_kwargs={"pitch_angle": 80.0, "fill_percent": 0.9, "fov": 40.0, 'plane_angle': 90, 'w': 1024, 'h': 1024}
        )
        imageio.imwrite(render_file_path.as_posix(), rendered_image)


def save_pts3d(save_dir, batch_idx, point_map, point_color):
    import open3d_pycg as open3d
    save_dir.mkdir(parents=True, exist_ok=True)

    point_map = point_map.view(-1, 3)
    point_color = point_color.view(-1, 3)
    valid_mask = point_map[..., 2] < 1e7
    
    point_map = point_map[valid_mask]
    point_color = point_color[valid_mask]

    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(point_map.cpu().numpy())
    pcd.colors = open3d.utility.Vector3dVector(point_color.cpu().numpy())
    open3d.io.write_point_cloud(save_dir / f"{batch_idx}.ply", pcd)
    print(f"Save to {save_dir / f'{batch_idx}.ply'}")


def render_and_save(grid_and_semantic, output_path):
    rendered = offsreen_mesh_renderer_for_vae(
        grid_and_semantic, extend_direction='x',
        default_camera_kwargs={"pitch_angle": 80.0, "fill_percent": 0.9, "fov": 40.0, 'plane_angle': 90, 'w': 1024, 'h': 1024}
    )
    imageio.imwrite(output_path, rendered)

def get_parser():
    parser = exp.ArgumentParserX(base_config_path='configs/default/param.yaml', parents=[get_default_parser()])
    parser.add_argument('--ckpt_name', type=str, required=False, default='last', help='if specify other ckpt name.')
    parser.add_argument('--ckpt_dm_c', type=str, required=True, help='Path to coarse stage diffusion ckpt file.')
    parser.add_argument('--ckpt_dm_f', type=str, default=None, help='Path to fine stage diffusion ckpt file.')
    parser.add_argument('--ckpt_gsm', type=str, default=None, help='Path to GSM ckpt file.')
    parser.add_argument('--nosync', action='store_true', help='Do not synchronize nas even if forced.')
    parser.add_argument('--wandb_base', type=str, default="../wandb/", help="Path to wandb base directory.")
    parser.add_argument('--split', type=str, default="test", help='Dataset split to evaluate on. test or train')
    parser.add_argument('--output_root', type=str, default="../cascading_diffusion_output_waymo_wds/", help='Output directory.')
    parser.add_argument('--suffix', type=str, default="", help='Suffix for output directory.')
    parser.add_argument('--val_starting_frame', type=int, default=100, help='Starting frame.')
    parser.add_argument('--input_frame_offsets', default=None, type=str, help='Input frame offsets.')
    parser.add_argument('--sup_frame_offsets', default=None, type=str, help='Input frame offsets.')
    parser.add_argument('--offset_unit', default='frame', type=str, help='Unit of the offset. frame or meter.')
    parser.add_argument('--use_view_5', action='store_true', help='Use 5 views for reconstruction')
    parser.add_argument('--dilate_road_ratio', default=2, type=int, help='Dilate ratio for road voxels. This fixes holes in the road.')

    parser.add_argument('--use_ddim', action='store_true', help='Use DDIM for diffusion.')
    parser.add_argument('--ddim_step', type=int, default=100, help='Number of steps to increase ddim.')
    parser.add_argument('--use_ema', action='store_true', help='Whether to turn on ema option.')
    parser.add_argument('--use_dpm', action='store_true', help='use DPM++ solver or not')
    parser.add_argument('--use_karras', action='store_true', help='use Karras noise schedule or not ')
    parser.add_argument('--solver_order', type=int, default=3, help='order of the solver; 3 for unconditional diffusion, 2 for guided sampling')
    parser.add_argument('--h_stride', type=int, default=2, help='Use for anisotropic pooling settting')
    parser.add_argument('--guidance_scale', type=float, default=1.0, help='Strength of the guidance for classifier-free.')
    return parser


def dilate_road_voxel(res_feature_set, out_vdb_tensor, dilate_road_ratio):
    from scube.utils.vis_util import WAYMO_CATEGORY_NAMES

    coarse_grid = out_vdb_tensor.grid
    coarse_semantic = res_feature_set.semantic_features[-1].jdata.argmax(dim=-1)
    road_voxel_label = WAYMO_CATEGORY_NAMES.index('ROAD')
    road_voxel_mask = coarse_semantic == road_voxel_label
    road_voxel_ijk = coarse_grid.ijk.jdata[road_voxel_mask]

    non_road_voxel_ijk = coarse_grid.ijk.jdata[~road_voxel_mask]
    non_road_voxel_semantic = coarse_semantic[~road_voxel_mask]

    road_voxel_grid = fvdb.gridbatch_from_ijk(road_voxel_ijk, voxel_sizes=coarse_grid.voxel_sizes, origins=coarse_grid.origins)
    road_voxel_grid_coarsen = road_voxel_grid.coarsened_grid([dilate_road_ratio,dilate_road_ratio,1])
    road_voxel_grid_coarsen_subdiv = road_voxel_grid_coarsen.subdivided_grid([dilate_road_ratio,dilate_road_ratio,1])
    road_voxel_dilated_ijk = road_voxel_grid_coarsen_subdiv.ijk.jdata

    # merge non_road_voxel_ijk and road_voxel_dilated_ijk. if ijk appears twice, use non_road_voxel_ijk and non_road_voxel_semantic
    merged_ijk = torch.cat([non_road_voxel_ijk, road_voxel_dilated_ijk], dim=0)
    merged_semantic = torch.cat([non_road_voxel_semantic, torch.full((road_voxel_dilated_ijk.shape[0],), road_voxel_label).to(non_road_voxel_semantic)], dim=0)

    unique_ijk, idx, counts = torch.unique(merged_ijk, dim=0, sorted=True, return_inverse=True, return_counts=True)
    _, ind_sorted = torch.sort(idx, stable=True)
    cum_sum = counts.cumsum(0)
    cum_sum = torch.cat((torch.tensor([0]).to(cum_sum), cum_sum[:-1]))
    first_indicies = ind_sorted[cum_sum]
    unique_semantic = merged_semantic[first_indicies]

    # create pseduo output
    @dataclass
    class FeatureSet:
        semantic_features: dict
    
    coarse_grid = fvdb.gridbatch_from_ijk(unique_ijk, voxel_sizes=coarse_grid.voxel_sizes, origins=coarse_grid.origins)
    ijk_order = coarse_grid.ijk_to_inv_index(unique_ijk)
    unique_semantic = unique_semantic[ijk_order.jdata]

    res_feature_set = FeatureSet(semantic_features={-1: VDBTensor(grid=coarse_grid, data=JaggedTensor([F.one_hot(unique_semantic, num_classes=23).float()]))})
    out_vdb_tensor = fvdb.nn.VDBTensor(grid=coarse_grid, data=JaggedTensor([F.one_hot(unique_semantic, num_classes=23).float()]))

    return res_feature_set, out_vdb_tensor

@torch.inference_mode()
def cascading_inference(net_model_diffusion_coarse, net_model_diffusion_fine, evaluation_kwargs_coarse, dilate_road_ratio, also_dilate_stage_2=False):
    assert evaluation_kwargs_coarse.get('batch', None) is None, 'batch should be None for pure generation'
    res_feature_set_coarse, out_vdb_tensor_coarse = \
        net_model_diffusion_coarse.evaluation_api(**evaluation_kwargs_coarse)

    if net_model_diffusion_fine is None:
        return (res_feature_set_coarse, out_vdb_tensor_coarse), (None, None)

    # get coarse stage output and prepare fine stage input
    if dilate_road_ratio > 1:
        res_feature_set_coarse, out_vdb_tensor_coarse = dilate_road_voxel(res_feature_set_coarse, out_vdb_tensor_coarse, dilate_road_ratio)

    coarse_grid = out_vdb_tensor_coarse.grid

    evaluation_kwargs_fine = evaluation_kwargs_coarse
    evaluation_kwargs_fine['cond_dict'].pop('semantics', None) # remove semantics from the first stage, if exists
    evaluation_kwargs_fine['grids'] = coarse_grid
    evaluation_kwargs_fine['res_coarse'] = res_feature_set_coarse
    
    # run fine stage
    res_feature_set_fine, out_vdb_tensor_fine = \
        net_model_diffusion_fine.evaluation_api(**evaluation_kwargs_fine)
    
    if dilate_road_ratio > 1 and also_dilate_stage_2:
        res_feature_set_fine, out_vdb_tensor_fine = dilate_road_voxel(res_feature_set_fine, out_vdb_tensor_fine, dilate_road_ratio)


    return (res_feature_set_coarse, out_vdb_tensor_coarse), (res_feature_set_fine, out_vdb_tensor_fine)


def cascading_diffusion_and_save(net_model_diffusion_coarse, net_model_diffusion_fine, net_model_gsm, dataset_c, dataset_f, saving_dir, args):
    saving_dir = saving_dir.resolve()

    dataloader_c = torch.utils.data.DataLoader(
        dataset_c,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=list_collate
    )

    dataloader_f = torch.utils.data.DataLoader(
        dataset_f,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=list_collate
    )


    evaluation_kwargs = {
        'use_ddim': args.use_ddim,
        'ddim_step': args.ddim_step,
        'use_ema': args.use_ema,
        'use_dpm': args.use_dpm,
        'use_karras': args.use_karras,
        'solver_order': args.solver_order,
        'guidance_scale': args.guidance_scale
    }
    
    for batch_idx, (batch, batch_high_ref) in enumerate(tqdm(zip(dataloader_c, dataloader_f))):

        print(batch[DS.SHAPE_NAME])
        print(batch_high_ref[DS.SHAPE_NAME])
        # if video exists, continue
        if (saving_dir / f"{batch_idx}_pred_images.mp4").exists():
            print(f"Skip {batch_idx} because '{batch_idx}_pred_images.mp4' exists.")
            continue

        batch = batch2device(batch, net_model_diffusion_coarse.device)
        batch_high_ref = batch2device(batch_high_ref, net_model_diffusion_coarse.device)

        net_model_diffusion_coarse.generate_fvdb_grid_on_the_fly(batch)
        net_model_diffusion_coarse.generate_fvdb_grid_on_the_fly(batch_high_ref)

        dense_latents = net_model_diffusion_coarse.create_dense_latents(batch_size=dataloader_c.batch_size, h_stride=args.h_stride)
        cond_dict = net_model_diffusion_coarse.create_cond_dict_from_batch(batch, dense_latents)
        view_num = len(dataset_c.input_slect_ids)

        evaluation_kwargs['grids'] = dense_latents
        evaluation_kwargs['cond_dict'] = cond_dict

        # high resolution outputs
        (res_feature_set_coarse, out_vdb_tensor_coarse), (res_feature_set_fine, out_vdb_tensor_fine) = \
            cascading_inference(net_model_diffusion_coarse, net_model_diffusion_fine, evaluation_kwargs, args.dilate_road_ratio)

        # save pred (coarse)
        save_pred_gt(res_feature_set_coarse, out_vdb_tensor_coarse, batch, batch_idx, saving_dir / 'coarse', save_render=True)
        output_grid = out_vdb_tensor_coarse.grid

        # save pred (fine)
        save_pred_gt(res_feature_set_fine, out_vdb_tensor_fine, batch_high_ref, batch_idx, saving_dir / 'fine')
        output_grid = out_vdb_tensor_fine.grid

        # save image
        if DS.IMAGES_INPUT in batch:
            if len(dataset_c.input_slect_ids) == 3:
                img_reorder = [1,0,2]
            elif len(dataset_c.input_slect_ids) == 5:
                img_reorder = [3,1,0,2,4]

            view_num = len(img_reorder)
            frame_num = batch[DS.IMAGES_INPUT][0].shape[0] // view_num

            torchvision.utils.save_image(
                torch.stack([batch[DS.IMAGES_INPUT][0][f*view_num+i] for f in range(frame_num) for i in img_reorder]).permute(0,3,1,2),
                saving_dir / f"{batch_idx}_image.jpg",
                nrow=len(img_reorder)
            )
            print(f"Save to {saving_dir / f'{batch_idx}_image.jpg'}")

        if net_model_gsm is not None:
            # create pseduo batch for inference
            batch_gsm = batch
            batch_gsm[DS.INPUT_PC] = output_grid
            with torch.inference_mode():
                renderer_output, network_output = net_model_gsm.forward(batch_gsm)
                gt_package = net_model_gsm.loss.prepare_resized_gt(batch_gsm)
                vis_images_dict = net_model_gsm.loss.assemble_visualization(gt_package, renderer_output)

                decoded_gaussians = network_output['decoded_gaussians'] # list. one element
                save_splat_file_RGB(decoded_gaussians[0], (saving_dir / f"{batch_idx}_splat.pkl").as_posix())

                print(f"Save to {saving_dir / f'{batch_idx}_splat.pkl'}")
                
                # save skybox representation
                net_model_gsm.skybox.save_skybox(network_output, (saving_dir / f"{batch_idx}_splat.pkl").as_posix())

                # save renderer decoder
                net_model_gsm.renderer.save_decoder((saving_dir / f"{batch_idx}_splat.pkl").as_posix())

                pd_images_fg = vis_images_dict['pd_images_fg'][0] # [N, H, W, 3]
                pd_images = vis_images_dict['pd_images'][0] # [N, H, W, 3]
                gt_images = vis_images_dict['gt_images'][0] # [N, H, W, 3]

                sup_frame_num = pd_images.shape[0] // view_num
                
                pd_image_fg_reorder = torch.cat([x[img_reorder] for x in torch.chunk(pd_images_fg, sup_frame_num, dim=0)], dim=0)
                pd_image_fg_reorder = pd_image_fg_reorder.permute(0, 3, 1, 2).clamp(0,1)

                pd_images_reorder = torch.cat([x[img_reorder] for x in torch.chunk(pd_images, sup_frame_num, dim=0)], dim=0)
                pd_images_reorder = pd_images_reorder.permute(0, 3, 1, 2).clamp(0,1)

                gt_images_reorder = torch.cat([x[img_reorder] for x in torch.chunk(gt_images, sup_frame_num, dim=0)], dim=0)
                gt_images_reorder = gt_images_reorder.permute(0, 3, 1, 2).clamp(0,1)

                torchvision.utils.save_image(pd_images_reorder, 
                                            saving_dir / f"{batch_idx}_pred_images.jpg", 
                                            nrow=len(img_reorder))

                torchvision.utils.save_image(pd_image_fg_reorder,
                                            saving_dir / f"{batch_idx}_pred_images_fg.jpg",
                                            nrow=len(img_reorder))
                
                torchvision.utils.save_image(gt_images_reorder,
                                            saving_dir / f"{batch_idx}_gt_images.jpg",
                                            nrow=len(img_reorder))


def main():
    known_args = get_parser().parse_known_args()[0]
    if known_args.suffix != "":
        known_args.suffix = "_" + known_args.suffix
    saving_dir = Path(known_args.output_root) / known_args.ckpt_dm_c.replace('/', '_') / \
                 (known_args.split + f"_starting_at_{known_args.val_starting_frame}" + \
                 f"_input_frame_offsets_{known_args.input_frame_offsets}_with_unit_{known_args.offset_unit}_view{5 if known_args.use_view_5 else 3}" + \
                  known_args.suffix)
    
    (saving_dir / 'fine' ).mkdir(parents=True, exist_ok=True)
    (saving_dir / 'coarse').mkdir(parents=True, exist_ok=True)

    hparam_update={'batch_size': 1, 
                   'batch_size_val': 1, 
                   'train_val_num_workers': 0, 
                   'skybox_forward_sky_only': True,
                   'skybox_resolution': 768}

    if known_args.input_frame_offsets is not None:
        known_args.input_frame_offsets = eval(known_args.input_frame_offsets)
        hparam_update['input_frame_offsets'] = known_args.input_frame_offsets # update using model dataset

    if known_args.sup_frame_offsets is None:
        known_args.sup_frame_offsets = known_args.input_frame_offsets
    else:
        known_args.sup_frame_offsets = eval(known_args.sup_frame_offsets)
        hparam_update['sup_frame_offsets'] = known_args.sup_frame_offsets

    # load model coarse
    net_model_diffusion_coarse, args, global_step_gsm = \
        create_model_from_args(known_args.ckpt_dm_c+":last", known_args, get_parser(), 
                               hparam_update=hparam_update)
    net_model_diffusion_coarse.to('cuda')
    
    # load model fine
    net_model_diffusion_fine, args, global_step_gsm = \
        create_model_from_args(known_args.ckpt_dm_f+":last", known_args, get_parser(),
                                hparam_update=hparam_update)
    net_model_diffusion_fine.to('cuda')

    # load model GSM
    if known_args.ckpt_gsm is not None:
        net_model_gsm, args, global_step_gsm = \
            create_model_from_args(known_args.ckpt_gsm+":last", known_args, get_parser(),
                                    hparam_update=hparam_update)
        net_model_gsm.to('cuda')
    else:
        net_model_gsm = None

    print("Loading to GPU done.")

    from scube.data.waymo_wds import WaymoWdsDataset

    if known_args.use_view_5:
        grid_crop_bbox_min = [-10.24, -51.2, -12.8]
        grid_crop_bbox_max = [92.16, 51.2, 38.4]
        slect_ids = [0,1,2,3,4]
        attr_subfolders = ['pose', 'intrinsic', 'pc_voxelsize_01',
                           'image_front', 'image_front_left', 'image_front_right', 'image_side_left', 'image_side_right',
                            'skymask_front', 'skymask_front_left', 'skymask_front_right', 'skymask_side_left', 'skymask_side_right',
                            'rectified_metric3d_depth_affine_100_front', 
                            'rectified_metric3d_depth_affine_100_front_left', 
                            'rectified_metric3d_depth_affine_100_front_right', 
                            'rectified_metric3d_depth_affine_100_side_left', 
                            'rectified_metric3d_depth_affine_100_side_right',
                            'all_object_info'
                           ]

    else:
        grid_crop_bbox_min = [0, -51.2, -12.8]
        grid_crop_bbox_max = [102.4, 51.2, 38.4]
        slect_ids = [0,1,2]
        attr_subfolders = ['pose', 'intrinsic', 'pc_voxelsize_01',
                            'image_front', 'image_front_left', 'image_front_right',
                            'skymask_front', 'skymask_front_left', 'skymask_front_right', 
                            'rectified_metric3d_depth_affine_100_front', 
                            'rectified_metric3d_depth_affine_100_front_left', 
                            'rectified_metric3d_depth_affine_100_front_right',
                            'all_object_info'
                           ]
    
    dataset_c = WaymoWdsDataset(
        wds_root_url='../waymo_webdataset',
        wds_scene_list_file='../waymo_split/official_val_static_scene.json' \
                            if (known_args.split == 'test' or known_args.split == 'val') else \
                            '../waymo_split/official_train_static_scene.json',
        attr_subfolders=attr_subfolders,
        spec=[DS.IMAGES_INPUT, DS.IMAGES_INPUT_POSE, DS.IMAGES_INPUT_INTRINSIC,
                DS.IMAGES, DS.IMAGES_POSE, DS.IMAGES_INTRINSIC,
                DS.GT_SEMANTIC, DS.IMAGES_INPUT_DEPTH],
        split='val',
        fvdb_grid_type='vs04',
        input_slect_ids=slect_ids,
        sup_slect_ids=slect_ids,
        input_frame_offsets=known_args.input_frame_offsets,
        sup_frame_offsets=known_args.sup_frame_offsets,
        offset_unit=known_args.offset_unit,
        grid_crop_bbox_min=grid_crop_bbox_min,
        grid_crop_bbox_max=grid_crop_bbox_max,
        val_starting_frame=known_args.val_starting_frame,
        input_depth_type='rectified_metric3d_depth_affine',
        replace_all_car_with_cad = True,
    )

    dataset_f = WaymoWdsDataset(
        wds_root_url='../waymo_webdataset',
        wds_scene_list_file='../waymo_split/official_val_static_scene.json' \
                            if (known_args.split == 'test' or known_args.split == 'val') else \
                            '../waymo_split/official_train_static_scene.json',
        attr_subfolders=attr_subfolders,
        spec=[DS.IMAGES_INPUT, DS.IMAGES_INPUT_POSE, DS.IMAGES_INPUT_INTRINSIC,
                DS.IMAGES, DS.IMAGES_POSE, DS.IMAGES_INTRINSIC,
                DS.GT_SEMANTIC],
        split='val',
        fvdb_grid_type='vs01',
        input_slect_ids=slect_ids,
        sup_slect_ids=slect_ids,
        input_frame_offsets=known_args.input_frame_offsets,
        sup_frame_offsets=known_args.sup_frame_offsets,
        offset_unit=known_args.offset_unit,
        grid_crop_bbox_min=grid_crop_bbox_min,
        grid_crop_bbox_max=grid_crop_bbox_max,
        val_starting_frame=known_args.val_starting_frame,
        replace_all_car_with_cad = True,
    )

    cascading_diffusion_and_save(net_model_diffusion_coarse, net_model_diffusion_fine, net_model_gsm, dataset_c, dataset_f, saving_dir, known_args)


if __name__ == "__main__":
    main()