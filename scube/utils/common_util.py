import torch
import numpy as np
import argparse
import pytorch_lightning as pl
import omegaconf
import importlib
import os

from loguru import logger 
from fvdb import GridBatch
from scube.utils import wandb_util
from omegaconf import OmegaConf

def batch2device(batch, device):
    """Send a batch to GPU"""
    if batch is None:
        return None
    for k, v in batch.items():
        if isinstance(v, list) and isinstance(v[0], torch.Tensor):
            batch[k] = [v[i].to(device) for i in range(len(v))]
        elif isinstance(v, GridBatch):
            batch[k] = v.to(device)
        elif isinstance(v, dict):
            batch[k] = batch2device(v, device)
    return batch

def get_default_parser():
    default_parser = argparse.ArgumentParser(add_help=False)
    default_parser = pl.Trainer.add_argparse_args(default_parser)
    return default_parser

def create_model_from_args(args_ckpt, known_args, parser, ckpt_name=None, hparam_update=None):
    wdb_run, args_ckpt = wandb_util.get_wandb_run(args_ckpt, wdb_base=known_args.wandb_base, default_ckpt="test_auto")
    logger.info(f"Use wandb run: {wdb_run.name}")
    assert args_ckpt is not None, "Please specify checkpoint version!"
    assert args_ckpt.exists(), "Selected checkpoint does not exist!"

    model_args = omegaconf.OmegaConf.create(wandb_util.recover_from_wandb_config(wdb_run.config))
    args = parser.parse_args(additional_args=model_args)
    if hasattr(args, 'nosync'):
        # Force not to sync to shorten bootstrap time.
        os.environ['NO_SYNC'] = '1'

    net_module = importlib.import_module("scube.models." + args.model).Model
    ckpt_path = args_ckpt

    if ckpt_name is not None:
        ckpt_path = str(ckpt_path).replace("last", ckpt_name)
        logger.info(f"Use ckpt: {ckpt_path}")

    print(f"Load model from {ckpt_path}")
    # change model config here
    if hparam_update is not None:
        for k, v in hparam_update.items():
            OmegaConf.update(args, k, v)
    net_model = net_module.load_from_checkpoint(ckpt_path, hparams=args, strict=False)

    # get net_state_dict
    net_state_dict = torch.load(ckpt_path, map_location="cpu")
    # get global_step
    global_step = net_state_dict["global_step"]

    return net_model.eval(), args, global_step

def merge_images_in_folder(image_folder):
    """
    image folder: str,
        several images in the folder, it's name follows
            {frame_id}_{view}.jpg

    We should open all the images as tensors, [N, C, H, W]
    downsample the resolution to 1/4, 
    save in one image use torchvision.utils.save_image, with nrow=num_view
    """
    from PIL import Image
    import torch.nn.functional as F
    import torchvision

    image_files = os.listdir(image_folder)
    image_files = [x for x in image_files if x.endswith(".jpg")] + [x for x in image_files if x.endswith(".png")]
    image_files.sort()

    num_view = len(set([x.split("_")[1] for x in image_files]))
    num_frame = len(image_files) // num_view

    if num_view == 3:
        reorder = [1,0,2]
    elif num_view == 5:
        reorder = [3,1,0,2,4]

    images = []
    for image_file in image_files:
        image = Image.open(os.path.join(image_folder, image_file))
        image = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255 # [C, H, W]
        images.append(image)

    images = torch.stack(images, dim=0)
    images = F.interpolate(images, scale_factor=1/4, mode='bilinear', antialias=True).clamp(0,1) # [N_frame*N_view, C, H, W]

    # reorder the images
    images = torch.cat([x[reorder] for x in torch.chunk(images, num_frame, dim=0)], dim=0)
    
    torchvision.utils.save_image(images, os.path.join(image_folder, "merged.jpg"), nrow=num_view)

def mask_image_patches(images: torch.Tensor, P: int, p_mask: float) -> torch.Tensor:
    """
    Masks patches of images with a specified probability.

    Parameters:
        images (torch.Tensor): Input tensor of shape [B, N, H, W, 1].
        P (int): Size of each patch.
        p_mask (float): Probability of masking each patch.

    Returns:
        torch.Tensor: Masked images of the same shape as input.
    """
    B, N, H, W, _ = images.shape
    
    # Calculate number of patches in height and width
    num_patches_h = H // P
    num_patches_w = W // P

    # Create a random mask for patches
    # Shape [B, N, num_patches_h * num_patches_w]
    random_mask = (torch.rand(B, N, num_patches_h, num_patches_w) < p_mask)
    random_mask = torch.repeat_interleave(random_mask, P, dim=2)
    random_mask = torch.repeat_interleave(random_mask, P, dim=3)
    random_mask = random_mask.unsqueeze(-1)
    
    return images * random_mask.to(images.device)