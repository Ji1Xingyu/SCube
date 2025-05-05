import argparse
import os
import pickle
from collections import OrderedDict
from io import BytesIO

import numpy as np
import torch
import torchvision
from plyfile import PlyData
from tqdm import tqdm, trange


def process_gaussian_params_to_splat(xyz, scaling, rotation, opacity, color):
    """
    xyz, scaling, rotation, opacity, color are all explicit parameters of the gaussian.

    xyz: (N, 3) float32
    scaling: (N, 3) float32
    rotation: (N, 4) float32
    opacity: (N,) or (N, 1) float32, range [0, 1]
    color: (N, 3) float32, range [0, 1]
    """
    if isinstance(xyz, torch.Tensor):
        xyz = xyz.detach().cpu().numpy()
        scaling = scaling.detach().cpu().numpy()
        rotation = rotation.detach().cpu().numpy()
        opacity = opacity.detach().cpu().numpy()
        color = color.detach().cpu().numpy()

    # sorted_indices = np.argsort(-scaling[:,0] * scaling[:,1] * scaling[:,2] * opacity)
    sorted_indices = np.arange(xyz.shape[0]) # no need to sort

    xyz = xyz.astype(np.float32)
    rotation = rotation.astype(np.float32)
    scaling = scaling.astype(np.float32)
    opacity = opacity.astype(np.float32)
    color = color.astype(np.float32)

    if len(opacity.shape) == 2:
        opacity = opacity[:, 0]

    buffer = BytesIO()
    for idx in tqdm(sorted_indices):
        position = xyz[idx]
        scales = scaling[idx]
        rot = rotation[idx]
        rgba = np.array(
            [
                color[idx][0],
                color[idx][1],
                color[idx][2],
                opacity[idx],
            ]
        )
        buffer.write(position.tobytes())
        buffer.write(scales.tobytes())
        buffer.write((rgba * 255).clip(0, 255).astype(np.uint8).tobytes())
        buffer.write(
            ((rot / np.linalg.norm(rot)) * 128 + 128)
            .clip(0, 255)
            .astype(np.uint8)
            .tobytes()
        )

    return buffer.getvalue()

def process_gaussian_params_to_dict(xyz, scaling, rotation, opacity, color):
    if isinstance(xyz, torch.Tensor):
        xyz = xyz.detach().cpu().numpy()
        scaling = scaling.detach().cpu().numpy()
        rotation = rotation.detach().cpu().numpy()
        opacity = opacity.detach().cpu().numpy()
        color = color.detach().cpu().numpy()
    
    gaussians = OrderedDict()
    gaussians['xyz'] = xyz
    gaussians['opacity'] = opacity
    gaussians['scaling'] = scaling
    gaussians['rotation'] = rotation
    gaussians['rgbs'] = color
    
    return gaussians

def save_splat_file(xyz, scaling, rotation, opacity, color, output_path):
    if output_path.endswith(".splat"):
        splat_data = process_gaussian_params_to_splat(xyz, scaling, rotation, opacity, color)
        with open(output_path, "wb") as f:
            f.write(splat_data)

    elif output_path.endswith(".pkl"):
        gaussians = process_gaussian_params_to_dict(xyz, scaling, rotation, opacity, color)
        with open(output_path, 'wb') as f:
            pickle.dump(gaussians, f)


def save_splat_file_RGB(gaussian_params, output_path):
    save_splat_file(*gaussian_params.split([3,3,4,1,3], dim=-1), output_path)

save_splat_file_concat = save_splat_file_RGB