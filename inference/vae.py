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
import argparse
import importlib
import torch
import torchvision
import numpy as np
import fvdb
import fvdb.nn
import imageio.v3 as imageio
import pytorch_lightning as pl
import sys

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
    parser.add_argument('--ckpt_vae', type=str, required=True, help='Path to ckpt file.')
    parser.add_argument('--nosync', action='store_true', help='Do not synchronize nas even if forced.')
    parser.add_argument('--wandb_base', type=str, default="../wandb/", help="Path to wandb base directory.")
    parser.add_argument('--split', type=str, default="test", help='Dataset split to evaluate on. test or train')
    parser.add_argument('--output_root', type=str, default="../vae_output_waymo_wds/", help='Output directory.')
    parser.add_argument('--suffix', type=str, default="", help='Suffix for output directory.')
    parser.add_argument('--val_starting_frame', type=int, default=100, help='Starting frame.')

    return parser

@torch.inference_mode()
def diffusion_and_save(net_model_vae, dataloader, saving_dir, known_args):
    for batch_idx, batch in enumerate(tqdm(dataloader)):
        batch = batch2device(batch, net_model_vae.device)
        output_dict = net_model_vae(batch, {})

        # save / visualize out_vdb_tensor
        grid = output_dict['tree'][0]
        semantic_prob = output_dict['semantic_features'][-1].data.jdata
        semantic = torch.argmax(semantic_prob, dim=-1)

        # save pred
        save_dict = {'points': grid.to('cpu'), 'semantics': semantic.to('cpu')}
        torch.save(save_dict, saving_dir / f"{batch_idx}.pt")
        print(f"Save to {saving_dir / f'{batch_idx}.pt'}")

        # save GT
        gt_save_dict = {'points': batch[DS.INPUT_PC].to('cpu'), 
                        'semantics': batch[DS.GT_SEMANTIC][0].to('cpu')}
        torch.save(gt_save_dict, saving_dir / f"{batch_idx}_gt.pt")
        print(f"Save to {saving_dir / f'{batch_idx}_gt.pt'}")


def main():
    known_args = get_parser().parse_known_args()[0]
    if known_args.suffix != "":
        known_args.suffix = "_" + known_args.suffix
    saving_dir = Path(known_args.output_root) / known_args.ckpt_vae.replace('/', '_') / \
                 (known_args.split + f"_starting_at_{known_args.val_starting_frame}" + known_args.suffix)
    saving_dir.mkdir(parents=True, exist_ok=True)

    net_model_vae, args, global_step_gsm = create_model_from_args(known_args.ckpt_vae+":last", known_args, get_parser())
    net_model_vae.cuda()

    if known_args.split == 'test':
        dataset_kwargs = net_model_vae.hparams.test_kwargs
    else:
        dataset_kwargs = net_model_vae.hparams.train_kwargs

    # update dataset_kwargs if needed

    if known_args.split == 'test':
        dataloader = net_model_vae.test_dataloader()
    else:
        dataloader = net_model_vae.train_dataloader() 

    diffusion_and_save(net_model_vae, dataloader, saving_dir, known_args)


if __name__ == "__main__":
    main()