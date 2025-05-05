# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import collections
import multiprocessing
import pathlib
from enum import Enum

import fvdb
import torch
from numpy.random import RandomState
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset

from scube.utils import exp

class DatasetSpec(Enum):
    SAVE_PATH = 3000
    SAVE_DICT = 3001    

    SHAPE_NAME = 100
    INPUT_PC = 200
    INPUT_PC_HIGHRES = 201
    INPUT_PC_RAW = 210
    INPUT_PC_RAW_HIGHRES = 211
    TARGET_NORMAL = 300
    INPUT_COLOR = 350
    INPUT_INTENSITY = 360
    GT_DENSE_PC = 400
    GT_DENSE_NORMAL = 500
    GT_DENSE_COLOR = 550
    GT_MESH = 600
    GT_MESH_SOUP = 650
    GT_ONET_SAMPLE = 700
    GT_GEOMETRY = 800
    DATASET_CFG = 1000
    GT_DYN_FLAG = 1100
    GT_SEMANTIC = 1200
    LATENT_SEMANTIC = 1300
    SINGLE_SCAN_CROP = 1400
    SINGLE_SCAN_INTENSITY_CROP = 1410
    SINGLE_SCAN = 1450
    SINGLE_SCAN_INTENSITY = 1460
    CLASS = 1500
    TEXT_EMBEDDING = 1600
    TEXT_EMBEDDING_MASK = 1610
    TEXT = 1620
    MICRO = 1630
    # images
    GRID_TO_FIRST_CAMERA_FLU = 1800
    GRID_CROP_RANGE = 1801
    GRID_TO_WORLD = 1802
    IMAGES_INPUT = 1900
    IMAGES_INPUT_MASK = 1901
    IMAGES_INPUT_POSE = 1910
    IMAGES_INPUT_FY = 1915
    IMAGES_INPUT_FOVY = 1916
    IMAGES_INPUT_INTRINSIC = 1917
    IMAGES_INPUT_DEPTH = 1918
    IMAGES = 2000
    IMAGES_MASK = 2001
    IMAGES_POSE = 2010
    IMAGES_FY = 2015
    IMAGES_CAMERA_DISTANCE = 2016
    IMAGES_INTRINSIC = 2017
    IMAGES_DINO_EMBEDDING = 2020
    IMAGES_DINO_POSE = 2030
    IMAGES_DINO_PATH = 2040
    IMAGES_DEPTH_MONO_EST = 2050
    IMAGES_DEPTH_MONO_EST_RECTIFIED = 2051
    IMAGES_DEPTH_LIDAR_PROJECT = 2052
    IMAGES_DEPTH_ANYTHING_V2_DEPTH_INV = 2053
    IMAGES_DEPTH_VOXEL = 2054
    IMAGES_NORMAL = 2060 
    # 3d map
    MAPS_3D = 2100
    MAPS_3D_DENSE_ROAD_SURFACE = 2101 # we provide dense road surface
    # 3d bbox
    BOXES_3D = 2200

class RandomSafeDataset(Dataset):
    """
    A dataset class that provides a deterministic random seed.
    However, in order to have consistent validation set, we need to set is_val=True for validation/test sets.
    Usage: First, inherent this class.
           Then, at the beginning of your get_item call, get an rng;
           Last, use this rng as the random state for your program.
    """

    def __init__(self, seed: int, _is_val: bool = False, skip_on_error: bool = False):
        self._seed = seed
        self._is_val = _is_val
        self.skip_on_error = skip_on_error
        if not self._is_val:
            self._manager = multiprocessing.Manager()
            self._read_count = self._manager.dict()
            self._rc_lock = multiprocessing.Lock()

    def get_rng(self, idx):
        if self._is_val:
            return RandomState(self._seed)
        with self._rc_lock:
            if idx not in self._read_count:
                self._read_count[idx] = 0
            rng = RandomState(exp.deterministic_hash((idx, self._read_count[idx], self._seed)))
            self._read_count[idx] += 1
        return rng

    def sanitize_specs(self, old_spec, available_spec):
        old_spec = set(old_spec)
        available_spec = set(available_spec)
        for os in old_spec:
            assert isinstance(os, DatasetSpec)
        new_spec = old_spec.intersection(available_spec)
        # lack_spec = old_spec.difference(new_spec)
        # if len(lack_spec) > 0:
        #     exp.logger.warning(f"Lack spec {lack_spec}.")
        return new_spec

    def _get_item(self, data_id, rng):
        raise NotImplementedError

    def __getitem__(self, data_id):
        rng = self.get_rng(data_id)
        if self.skip_on_error:
            try:
                return self._get_item(data_id, rng)
            except ConnectionAbortedError:
                return self.__getitem__(rng.randint(0, len(self) - 1))
            except Exception:
                # Just return a random other item.
                print(f"Warning: Get item {data_id} error, but handled.")
                return self.__getitem__(rng.randint(0, len(self) - 1))
        else:
            try:
                return self._get_item(data_id, rng)
            except ConnectionAbortedError:
                return self.__getitem__(rng.randint(0, len(self) - 1))


def list_collate(batch):
    """
    This just do not stack batch dimension.
    """
    from fvdb import GridBatch, JaggedTensor

    elem = None
    for e in batch:
        if e is not None:
            elem = e
            break
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        return batch
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            return list_collate([torch.as_tensor(b) if b is not None else None for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, str):
        return batch
    elif isinstance(elem, DictConfig) or isinstance(elem, ListConfig):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        if DatasetSpec.MAPS_3D in elem:
            collated_map = {key: [] for key in elem[DatasetSpec.MAPS_3D].keys()}
            for d in batch:
                for key in collated_map.keys():
                    collated_map[key].append(d[DatasetSpec.MAPS_3D][key])

            common_batch = {key: list_collate([d[key] for d in batch]) for key in elem if key != DatasetSpec.MAPS_3D}
            return {DatasetSpec.MAPS_3D: collated_map, **common_batch}
    
        else:
            return {key: list_collate([d[key] for d in batch]) for key in elem}
        
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [list_collate(samples) for samples in transposed]
    elif isinstance(elem, GridBatch):
        if fvdb.__version__ == '0.0.0':
            return fvdb.cat(batch)
        else:
            return fvdb.jcat(batch)
    elif isinstance(elem, JaggedTensor):
        return fvdb.jcat(batch)
    
    # elif isinstance(elem, pathlib.Path):
    #     return batch
    # elif elem is None:
    #     return batch

    # raise NotImplementedError
    return batch