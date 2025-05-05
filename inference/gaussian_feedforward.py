# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import sys, os
from pathlib import Path
sys.path.append(Path(__file__).parent.parent.as_posix())

import torch
import torchvision
import numpy as np
import fvdb
import fvdb.nn
import imageio.v3 as imageio
import pytorch_lightning as pl

import torch.nn.functional as F
from omegaconf import OmegaConf
from pycg import exp
from loguru import logger 
from tqdm import tqdm
from scube.utils.common_util import batch2device, get_default_parser, create_model_from_args
from scube.utils.gaussian_util import save_splat_file_RGB
from scube.data.base import DatasetSpec as DS

fvdb.nn.SparseConv3d.backend = 'igemm_mode1'

def get_parser():
    parser = exp.ArgumentParserX(base_config_path='configs/default/param.yaml', parents=[get_default_parser()])
    parser.add_argument('--ckpt_name', type=str, required=False, default='last', help='if specify other ckpt name.')
    parser.add_argument('--ckpt_gsm', type=str, required=True, help='Path to ckpt file.')
    parser.add_argument('--nosync', action='store_true', help='Do not synchronize nas even if forced.')
    parser.add_argument('--wandb_base', type=str, default="../wandb/", help="Path to wandb base directory.")
    parser.add_argument('--split', type=str, default="test", help='Dataset split to evaluate on. test or train')
    parser.add_argument('--output_root', type=str, default="../splat_output_waymo_wds/", help='Output directory.')
    parser.add_argument('--suffix', type=str, default="", help='Suffix for output directory.')
    parser.add_argument('--save_img_separately', action='store_true', help='save pred image separately in one folder')
    parser.add_argument('--save_gs', action='store_true', help='save gaussians to .pkl file')
    parser.add_argument('--input_frame_offsets', type=int, nargs='+', default=None, help='Input frame offsets.')
    parser.add_argument('--val_starting_frame', type=int, default=100, help='Starting frame.')
    parser.add_argument('--skybox_resolution', type=int, default=768, help='Skybox resolution.')
    return parser

@torch.inference_mode()
def render_and_save_gsm(net_model_gsm, dataloader, saving_dir, img_reorder, 
                        save_img_together, 
                        save_img_separately,
                        save_gaussians):
    for batch_idx, batch in enumerate(tqdm(dataloader)):
        batch = batch2device(batch, net_model_gsm.device)
        renderer_output, network_output = net_model_gsm.forward(batch)
        gt_package = net_model_gsm.loss.prepare_resized_gt(batch)
        vis_images_dict = net_model_gsm.loss.assemble_visualization(gt_package, renderer_output)

        if save_img_together or save_img_separately:
            gt_images = vis_images_dict['gt_images'][0] # [N, H, W, 3]
            pd_images = vis_images_dict['pd_images'][0] # [N, H, W, 3]
            pd_images_fg = vis_images_dict['pd_images_fg'][0] # [N, H, W, 3]

            # reorder for better visualization. [N, H, W, 3]
            n_frames = gt_images.shape[0] // len(img_reorder)
            pd_images_reorder = torch.cat([x[img_reorder] for x in torch.chunk(pd_images, n_frames, dim=0)], dim=0)
            pd_images_fg_reorder = torch.cat([x[img_reorder] for x in torch.chunk(pd_images_fg, n_frames, dim=0)], dim=0)
            gt_images_reorder = torch.cat([x[img_reorder] for x in torch.chunk(gt_images, n_frames, dim=0)], dim=0)

        if save_img_together:
            DOWNSAMPLE = False
            if DOWNSAMPLE:
                pd_images_reorder_resize = F.interpolate(pd_images_reorder.permute(0, 3, 1, 2), scale_factor=1/4, mode='bilinear', antialias=True).clamp(0,1)
                pd_images_fg_reorder = F.interpolate(pd_images_fg_reorder.permute(0, 3, 1, 2), scale_factor=1/4, mode='bilinear', antialias=True).clamp(0,1)
                gt_images_reorder_resize = F.interpolate(gt_images_reorder.permute(0, 3, 1, 2), scale_factor=1/4, mode='bilinear', antialias=True).clamp(0,1)

            else:
                pd_images_reorder_resize = pd_images_reorder.permute(0, 3, 1, 2)
                pd_images_fg_reorder = pd_images_fg_reorder.permute(0, 3, 1, 2)
                gt_images_reorder_resize = gt_images_reorder.permute(0, 3, 1, 2)

            torchvision.utils.save_image(pd_images_reorder_resize, 
                                         saving_dir / f"{batch_idx}_pred_images.jpg", 
                                         nrow=len(img_reorder))
            torchvision.utils.save_image(pd_images_fg_reorder,
                                         saving_dir / f"{batch_idx}_pred_images_fg.jpg",
                                         nrow=len(img_reorder))
            torchvision.utils.save_image(gt_images_reorder_resize, 
                                         saving_dir / f"{batch_idx}_gt_images.jpg", 
                                         nrow=len(img_reorder))

        if save_img_separately:
            out_folder = saving_dir / f"{batch_idx}"
            out_folder.mkdir(parents=True, exist_ok=True)
            pd_images_numpy = (pd_images.clamp(0,1).cpu().numpy() * 255).astype(np.uint8)
            pd_images_fg_numpy = (pd_images_fg.clamp(0,1).cpu().numpy() * 255).astype(np.uint8)
            gt_images_numpy = (gt_images.clamp(0,1).cpu().numpy() * 255).astype(np.uint8)
            sup_frame_idxs = batch[DS.SHAPE_NAME][0].split('_with_sup_frames_')[-1]
            sup_frame_idxs = sup_frame_idxs.split('_')
            
            for idx, frame_name in enumerate(sup_frame_idxs):
                for view_idx in range(len(img_reorder)):
                    imageio.imwrite(out_folder / f"{frame_name}_{view_idx}.jpg",
                                    pd_images_numpy[idx * len(img_reorder) + view_idx])
        
            for idx, frame_name in enumerate(sup_frame_idxs):
                for view_idx in range(len(img_reorder)):
                    imageio.imwrite(out_folder / f"{frame_name}_{view_idx}_fg.jpg",
                                    pd_images_fg_numpy[idx * len(img_reorder) + view_idx])
            
            for idx, frame_name in enumerate(sup_frame_idxs):
                for view_idx in range(len(img_reorder)):
                    imageio.imwrite(out_folder / f"{frame_name}_{view_idx}_gt.jpg",
                                    gt_images_numpy[idx * len(img_reorder) + view_idx])

        if save_gaussians:
            decoded_gaussians = network_output['decoded_gaussians'][0] 
            assert decoded_gaussians.shape[1] == 14
            output_path = saving_dir / f"{batch_idx}_rgb_gaussians.pkl"
            save_splat_file_RGB(decoded_gaussians, output_path.as_posix())

            # save skybox representation
            net_model_gsm.skybox.save_skybox(network_output, output_path)

            # save renderer decoder
            net_model_gsm.renderer.save_decoder(output_path)


def main():
    known_args = get_parser().parse_known_args()[0]
    if known_args.suffix != "":
        known_args.suffix = "_" + known_args.suffix
    saving_dir = Path(known_args.output_root) / known_args.ckpt_gsm.replace('/', '_') / \
                 (known_args.split + f"_starting_at_{known_args.val_starting_frame}" + known_args.suffix)
    saving_dir.mkdir(parents=True, exist_ok=True)

    hparam_update = {
        'skybox_resolution': known_args.skybox_resolution,
        'skybox_forward_sky_only': True,
        'train_val_num_workers': 0
    }

    net_model_gsm, args, global_step_gsm = create_model_from_args(known_args.ckpt_gsm+":last", known_args, get_parser(), hparam_update=hparam_update)
    net_model_gsm.cuda()

    if known_args.split == 'test':
        dataset_kwargs = net_model_gsm.hparams.test_kwargs
    else:
        dataset_kwargs = net_model_gsm.hparams.train_kwargs
    
    # change data set configs if needed
    if known_args.input_frame_offsets is not None:
        input_frame_offsets = known_args.input_frame_offsets
    else:
        input_frame_offsets = dataset_kwargs['input_frame_offsets']

    # update dataset_kwargs
    dataset_kwargs['split'] = 'test' # we can use the train_dataset, but set split to test for no random selection
    dataset_kwargs['val_starting_frame'] = known_args.val_starting_frame
    dataset_kwargs['input_frame_offsets'] = input_frame_offsets
    dataset_kwargs['sup_slect_ids'] = dataset_kwargs['sup_slect_ids']
    dataset_kwargs['sup_frame_offsets'] = dataset_kwargs['sup_frame_offsets']
    dataset_kwargs['n_image_per_iter_sup'] = None
    
    # reorder for better visualization
    if len(dataset_kwargs['sup_slect_ids']) == 3:
        img_reorder = [1,0,2]
    elif len(dataset_kwargs['sup_slect_ids']) == 5:
        img_reorder = [3,1,0,2,4]
    elif len(dataset_kwargs['sup_slect_ids']) == 1:
        img_reorder = [0]
    else:
        raise NotImplementedError

    if known_args.split == 'test':
        dataloader = net_model_gsm.test_dataloader()
    else:
        dataloader = net_model_gsm.train_dataloader() 

    render_and_save_gsm(net_model_gsm, dataloader, saving_dir, img_reorder, 
                        save_img_together=True,
                        save_img_separately=known_args.save_img_separately,
                        save_gaussians=known_args.save_gs)

if __name__ == "__main__":
    main()