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
import omegaconf
import sys
import importlib
import torch
import torchvision
import numpy as np
import fvdb
import fvdb.nn
import imageio.v3 as imageio
import pytorch_lightning as pl
from pathlib import Path

sys.path.append(Path(__file__).parent.parent.as_posix())

from omegaconf import OmegaConf
from fvdb import JaggedTensor, GridBatch
from fvdb.nn import VDBTensor
from pathlib import Path
from pycg import exp
from scube.utils import wandb_util
from loguru import logger 
from tqdm import tqdm
from scube.utils.common_util import batch2device, get_default_parser, create_model_from_args
from scube.utils.gaussian_util import save_splat_file_RGB
from scube.data.base import DatasetSpec as DS

fvdb.nn.SparseConv3d.backend = 'igemm_mode1'

def get_parser():
    parser = exp.ArgumentParserX(base_config_path='configs/default/param.yaml', parents=[get_default_parser()])
    parser.add_argument('--ckpt_name', type=str, required=False, default='last', help='if specify other ckpt name.')
    parser.add_argument('--ckpt_dm', type=str, required=True, help='Path to ckpt file.')
    parser.add_argument('--nosync', action='store_true', help='Do not synchronize nas even if forced.')
    parser.add_argument('--wandb_base', type=str, default="../wandb/", help="Path to wandb base directory.")
    parser.add_argument('--split', type=str, default="test", help='Dataset split to evaluate on. test or train')
    parser.add_argument('--output_root', type=str, default="../diffusion_output_waymo_wds/", help='Output directory.')
    parser.add_argument('--suffix', type=str, default="", help='Suffix for output directory.')
    parser.add_argument('--val_starting_frame', type=int, default=100, help='Starting frame.')

    parser.add_argument('--use_ddim', action='store_true', help='Use DDIM for diffusion.')
    parser.add_argument('--ddim_step', type=int, default=100, help='Number of steps to increase ddim.')
    parser.add_argument('--use_ema', action='store_true', help='Whether to turn on ema option.')
    parser.add_argument('--use_dpm', action='store_true', help='use DPM++ solver or not')
    parser.add_argument('--use_karras', action='store_true', help='use Karras noise schedule or not ')
    parser.add_argument('--solver_order', type=int, default=3, help='order of the solver; 3 for unconditional diffusion, 2 for guided sampling')
    parser.add_argument('--h_stride', type=int, default=1, help='Use for anisotropic pooling settting')
    parser.add_argument('--guidance_scale', type=float, default=1.0, help='Strength of the guidance for classifier-free.')
    return parser

@torch.inference_mode()
def diffusion_and_save(net_model_diffusion, dataloader, saving_dir, known_args):
    for batch_idx, batch in enumerate(tqdm(dataloader)):
        batch = batch2device(batch, net_model_diffusion.device)
        res_feature_set, out_vdb_tensor = \
            net_model_diffusion.evaluation_api(batch, grids=None, 
                                               use_ddim=known_args.use_ddim,
                                               ddim_step=known_args.ddim_step,
                                               use_ema=known_args.use_ema,
                                               use_dpm=known_args.use_dpm,
                                               use_karras=known_args.use_karras,
                                               solver_order=known_args.solver_order,
                                               h_stride=known_args.h_stride,
                                               guidance_scale=known_args.guidance_scale)

        # save / visualize out_vdb_tensor
        grid = out_vdb_tensor.grid
        semantic_prob = res_feature_set.semantic_features[-1].jdata # [n_voxel, 23]
        semantic = semantic_prob.argmax(dim=-1) # [n_voxel, ]

        # save pred
        save_dict = {'points': grid.to('cpu'), 'semantics': semantic.to('cpu')}
        torch.save(save_dict, saving_dir / f"{batch_idx}.pt")
        print(f"Save to {saving_dir / f'{batch_idx}.pt'}")

        # save GT
        gt_save_dict = {'points': batch[DS.INPUT_PC].to('cpu'), 
                        'semantics': batch[DS.GT_SEMANTIC][0].to('cpu')}
        torch.save(gt_save_dict, saving_dir / f"{batch_idx}_gt.pt")
        print(f"Save to {saving_dir / f'{batch_idx}_gt.pt'}")

        # save image
        if DS.IMAGES_INPUT in batch:
            if len(net_model_diffusion.hparams._input_slect_ids) == 3:
                img_reorder = [1,0,2]
            elif len(net_model_diffusion.hparams._input_slect_ids) == 5:
                img_reorder = [3,1,0,2,4]

            view_num = len(img_reorder)
            frame_num = batch[DS.IMAGES_INPUT][0].shape[0] // view_num

            torchvision.utils.save_image(
                torch.stack([batch[DS.IMAGES_INPUT][0][f*view_num+i] for f in range(frame_num) for i in img_reorder]).permute(0,3,1,2),
                saving_dir / f"{batch_idx}_image.jpg",
                nrow=len(img_reorder)
            )
            print(f"Save to {saving_dir / f'{batch_idx}_image.jpg'}")

        if DS.MAPS_3D in batch:
            # render map 
            from pycg import vis, render
            from matplotlib.colors import LinearSegmentedColormap
            from scube.utils.color_util import semantic_from_points

            colors = ["orange", "cyan", "red"]
            cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)

            pc_list = []
            maps_3d = batch[DS.MAPS_3D]  # dict, the key is map types, the value is [n_points, 3]

            rendered_images = []

            # this is saying to ego, not grid coordinate
            grid_crop_bbox_min = torch.tensor(net_model_diffusion.hparams.grid_crop_bbox_min, device=net_model_diffusion.device)
            grid_crop_bbox_max = torch.tensor(net_model_diffusion.hparams.grid_crop_bbox_max, device=net_model_diffusion.device)

            # convert to grid coordinate
            grid_crop_bbox_max = (grid_crop_bbox_max - grid_crop_bbox_min) / 2
            grid_crop_bbox_min = - grid_crop_bbox_max
            
            # crop the points 
            for map_type, map_points in maps_3d.items():
                map_points = map_points[0] # extract the first sample
                map_points = map_points.to(torch.int32)
                # discard points outside the grid
                map_points = map_points[(map_points >= grid_crop_bbox_min).all(dim=1) & (map_points < grid_crop_bbox_max).all(dim=1)]
                # only keep unique points
                map_points = torch.unique(map_points, dim=0)

                maps_3d[map_type] = map_points.float()

            # prepare pc_list
            for idx, map_type in enumerate(maps_3d):
                if maps_3d[map_type].shape[0] == 0:
                    continue
                map_color = np.array(cmap(idx / len(maps_3d)))[:3].reshape(1, 3).repeat(maps_3d[map_type].shape[0], axis=0)
                map_pc = vis.pointcloud(pc=maps_3d[map_type].to('cpu').numpy(), color=map_color)
                pc_list.append(map_pc)

            # render the map
            for plane_angle in [90, 180, 270, 0]:
                scene: render.Scene = vis.show_3d(pc_list, show=False, up_axis='+Z', default_camera_kwargs={"pitch_angle": 45.0, "fill_percent": 0.7, "fov": 40.0, 'plane_angle': plane_angle})
                img = scene.render_filament()
                rendered_images.append(img)

            rendered_images = np.concatenate(rendered_images, axis=1)
            imageio.imwrite(saving_dir / f"{batch_idx}_map.jpg", rendered_images)

            # save as fvdb grid as well (for easier visualization)
            origins = grid.origins
            voxel_sizes = grid.voxel_sizes
            ijk_collection = []
            point_collection = []
            semantic_collection = []
            for idx, (map_type, map_points) in enumerate(maps_3d.items()):
                if map_points.shape[0] == 0:
                    continue

                point_collection.append(map_points)
                semantic_collection.append(torch.tensor([idx] * map_points.shape[0], dtype=torch.int32, device=map_points.device))

                grid = fvdb.gridbatch_from_points(map_points, voxel_sizes=voxel_sizes, origins=origins)
                ijk_collection.append(grid.ijk.jdata)

            ijk_collection = torch.cat(ijk_collection, dim=0)
            merged_grid = fvdb.gridbatch_from_ijk(ijk_collection, voxel_sizes=voxel_sizes, origins=origins)

            merged_grid_semantic = semantic_from_points(
                merged_grid.grid_to_world(merged_grid.ijk.float()).jdata,
                torch.cat(point_collection, dim=0),
                torch.cat(semantic_collection, dim=0)
            )

            # save pt
            save_dict = {'points': merged_grid.to('cpu'), 'semantics': merged_grid_semantic.to('cpu')}

            # we save it in another folder, i.e, saving_dir with extra '_map' suffix. not a subfolder!
            saving_dir_map = Path(str(saving_dir) + "_map")
            saving_dir_map.mkdir(parents=True, exist_ok=True)

            torch.save(save_dict, saving_dir_map / f"{batch_idx}.pt")

def main():
    known_args = get_parser().parse_known_args()[0]
    if known_args.suffix != "":
        known_args.suffix = "_" + known_args.suffix
    saving_dir = Path(known_args.output_root) / known_args.ckpt_dm.replace('/', '_') / \
                 (known_args.split + f"_starting_at_{known_args.val_starting_frame}" + known_args.suffix)
    saving_dir.mkdir(parents=True, exist_ok=True)

    net_model_diffusion, args, global_step_gsm = \
        create_model_from_args(known_args.ckpt_dm+":last", known_args, get_parser(), 
                               hparam_update={'batch_size': 1, 'batch_size_val': 1, 'train_val_num_workers': 0})
    net_model_diffusion.cuda()

    if known_args.split == 'test':
        dataset_kwargs = net_model_diffusion.hparams.test_kwargs
    else:
        dataset_kwargs = net_model_diffusion.hparams.train_kwargs

    # update dataset_kwargs if needed
    dataset_kwargs['split'] = 'not_train'

    if known_args.split == 'test':
        dataloader = net_model_diffusion.test_dataloader()
    else:
        dataloader = net_model_diffusion.train_dataloader() 

    diffusion_and_save(net_model_diffusion, dataloader, saving_dir, known_args)


if __name__ == "__main__":
    main()