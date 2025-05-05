# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import numpy as np
import webdataset as wds
import json
import time
import imageio.v2 as imageio
import torch
import torch.nn.functional as F
import random
import os
import braceexpand
import fvdb
import traceback

try:
    import open3d as o3d
except ImportError:
    import open3d_pycg as o3d

from loguru import logger
from fvdb import GridBatch
from omegaconf import OmegaConf
from torch.distributed import get_rank, get_world_size
from scube.data.base import DatasetSpec as DS
from scube.utils.wds_util import s3_url_opener, imagehandler
from webdataset.tariterators import url_opener, tarfile_samples
from webdataset import autodecode
from omegaconf import ListConfig
from functools import partial
from termcolor import colored
from pytorch3d.ops.iou_box3d import _check_coplanar, _check_nonzero

class BaseWdsDataset(torch.utils.data.IterableDataset):
    def __init__(self, wds_root_url, wds_scene_list_file, attr_subfolders=['image_front'],
                spec=None, split='train', 
                frame_start_num=0, frame_end_num=170,
                grid_crop_bbox_min=[-10.24, -51.2, -12.8], grid_crop_bbox_max=[92.16, 51.2, 38.4],
                input_slect_ids=[0,1,2], input_frame_offsets=[0], offset_unit='frame', # frame or meter
                sup_slect_ids=[0,1,2], sup_frame_offsets=[0], n_image_per_iter_sup=None,
                fvdb_grid_type='vs01', finest_voxel_size_goal='vs01',
                val_starting_frame=100, map_types=None,
                skip_on_error=False, random_seed=0, shuffle_buffer=128, input_depth_type='metric3d_depth',
                custom_name='scene_base', hparams=None, s3_cfg=None, add_high_res_grid=False, 
                tolerent_shorter_offsets=False,
                grid_crop_augment=False, grid_crop_augment_range=[12.8, 12.8, 3.2],
                replace_all_car_with_cad=False,
                add_road_line_to_GT=False, voxel_shape = (256, 256, 32), save_path = './waymo_semcity/', 
                **kwargs):
        """
        Args:
            offset_unit: 'frame' or 'meter'
                if 'frame', input_frame_offsets and sup_frame_offsets refer to the frame index diff.
                if 'meter', input_frame_offsets and sup_frame_offsets refer to the distance in meters.
        """

        self.split = split
        self.wds_root_url = wds_root_url
        self.attr_subfolders = attr_subfolders
        self.custom_name = custom_name
        self.shuffle_buffer = shuffle_buffer
        self.hparams = hparams
        self.s3_cfg = s3_cfg
        self.input_slect_ids = input_slect_ids
        self.input_frame_offsets = input_frame_offsets
        self.offset_unit = offset_unit
        self.sup_slect_ids = sup_slect_ids
        self.sup_frame_offsets = sup_frame_offsets
        self.input_depth_type = input_depth_type
        self.map_types = map_types
        self.tolerent_shorter_offsets = tolerent_shorter_offsets
        # save bbx
        self.voxel_shape = voxel_shape
        self.save_bbx = True
        self.save_path = save_path

        self.grid_crop_bbox_min = grid_crop_bbox_min if isinstance(grid_crop_bbox_min, list) else \
            OmegaConf.to_container(grid_crop_bbox_min)
        self.grid_crop_bbox_max = grid_crop_bbox_max if isinstance(grid_crop_bbox_max, list) else \
            OmegaConf.to_container(grid_crop_bbox_max)

        self.grid_length_in_meter = [self.grid_crop_bbox_max[i] - self.grid_crop_bbox_min[i] for i in range(3)]
        self.grid_half_diagonal = (self.grid_length_in_meter[0]**2 + self.grid_length_in_meter[1]**2)**0.5 / 2
        

        self.val_starting_frame = val_starting_frame

        self.frame_start_num = frame_start_num

        if offset_unit == 'frame':
            self.last_starting_frame = frame_end_num - max(max(input_frame_offsets), max(sup_frame_offsets)) - 1  # in the case of offset_unit='frame'
        else:
            logger.warning(f"offset_unit is meter, we can not determine the last starting frame, set it to a big number {frame_end_num}." + 
                            "We will skip the sample if the sequence is too short.")
            self.last_starting_frame = frame_end_num

        self.sample_time_from_shard = self.last_starting_frame if split == 'train' else 1

        self.spec = spec
        self.add_high_res_grid = add_high_res_grid
        self.fvdb_grid_type = fvdb_grid_type
        self.finest_voxel_size_goal = finest_voxel_size_goal

        if n_image_per_iter_sup is None:
            self.n_image_per_iter_sup = len(sup_slect_ids) * len(sup_frame_offsets)
        else:
            self.n_image_per_iter_sup = n_image_per_iter_sup
        
        # should be .json file. otherwise only for debug use.
        if wds_scene_list_file.endswith('.json'):
            wds_scene_list = json.load(open(wds_scene_list_file, 'r'))
        else: 
            wds_scene_list = [wds_scene_list_file]

        self.grid_crop_augment = grid_crop_augment
        self.grid_crop_augment_range = grid_crop_augment_range
        self.replace_all_car_with_cad = replace_all_car_with_cad
        self.add_road_line_to_GT = add_road_line_to_GT

        self.wds_scene_list = wds_scene_list
        self.prepare_pre_pipeline()


    def prepare_pre_pipeline(self):
        if 's3://' not in self.wds_root_url:
            self.url_opener_custom = url_opener
        else:
            self.url_opener_custom = partial(s3_url_opener, s3_cfg=self.s3_cfg)

        self.wds_url_collection = [os.path.join(self.wds_root_url, self.attr_subfolders[0], f"{scene}.tar") 
                                   for scene in self.wds_scene_list]
        
        decode_handlers = [imagehandler('npraw')] 
        self.decode_func = autodecode.Decoder(decode_handlers)

    def __len__(self):
        _, world_size, _ = self.get_rank_worker()
        
        if self.split == 'train':
            return len(self.wds_scene_list) * (self.last_starting_frame - self.frame_start_num) // world_size
        else:
            return len(self.wds_scene_list) # each worker reads all the data
        
    def get_rank_worker(self, string=None):
        from torch.utils.data import get_worker_info
        try:
            rank = get_rank()
            world_size = get_world_size()
        except ValueError:  # single gpu
            rank = 0
            world_size = 1

        worker_info = get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
        else: # single process
            worker_id = 0
            
        if string is not None:
            print(f"[{colored('rank='+str(rank), 'cyan')} {colored('worker='+str(worker_id), 'red')}] {string}")
        
        return rank, world_size, worker_id

    
    def _assemble_data(self, attr_subfolder, use_frame_idxs, reassemble_sample, sample):
        # Case 1: per frame data with frame indexing
        if 'image' in attr_subfolder or 'skymask' in attr_subfolder or \
            'pose' in attr_subfolder or 'depth' in attr_subfolder or \
            'gbuffer' in attr_subfolder or 'all_object_info' in attr_subfolder:
            for use_frame_idx in use_frame_idxs:
                key_names = [x for x in sample.keys() if f"{use_frame_idx:06d}" in x]
                for key_name in key_names:
                    reassemble_sample[key_name] = sample[key_name]

        elif attr_subfolder == 'dynamic_object_transformation':
            for use_frame_idx in use_frame_idxs:
                key_names = [x for x in sample.keys() if f"{use_frame_idx:06d}" in x]
                for key_name in key_names:
                    reassemble_sample[key_name] = sample[key_name]

        # Case 2: per scene data without frame indexing
        elif 'grid' in attr_subfolder:
            key_name = [x for x in sample.keys() if 'grid' in x][0]
            reassemble_sample[key_name] = sample[key_name]

        elif 'pc' in attr_subfolder:
            key_name = [x for x in sample.keys() if 'pcd' in x][0]
            reassemble_sample[key_name] = sample[key_name]

        elif 'intrinsic' in attr_subfolder:
            key_names = [x for x in sample.keys() if 'intrinsic' in x]
            for key_name in key_names:
                reassemble_sample[key_name] = sample[key_name]
        
        elif 'static_object' in attr_subfolder:
            # `static_object.npy` and `point_num_in_static_object.npy`
            key_names = [x for x in sample.keys() if '.npy' in x]
            for key_name in key_names:
                reassemble_sample[key_name] = sample[key_name]
        
        elif attr_subfolder.startswith('3d_') and attr_subfolder.endswith('merged'):
            key_name = [x for x in sample.keys() if '.npy' in x][0]
            reassemble_sample[key_name] = sample[key_name]

        elif attr_subfolder == 'artificial_dense_road_surface' or attr_subfolder == 'artificial_dense_road_surface_v2':
            key_name = [x for x in sample.keys() if '.npy' in x][0]
            reassemble_sample[key_name] = sample[key_name]

        elif attr_subfolder == 'dynamic_object_points_canonical':
            key_name = [x for x in sample.keys() if '.npz' in x][0]
            reassemble_sample[key_name] = sample[key_name]
        
        else:
            raise NotImplementedError(f"attr_subfolder {attr_subfolder} is not implemented yet.")

    def get_depth_images(self, sample, frame_idxs, select_view_ids, sup_image_indices=None, depth_type='none'):
        """
        depth_type can be: 
        - 'none'
        - 'lidar_depth'
        - 'rectified_metric3d_depth' 
        - 'metric3d_depth' 
        - 'rectified_metric3d_depth_affine' 
        - 'rectified_depth_anything_v2_depth'
        - 'depth_anything_v2_depth_inv'

        Returns:
            torch.Tensor, shape (N, H, W, 1)
        """
        all_depth_images_tensor = []

        if sup_image_indices is None:
            sup_image_indices = np.arange(len(frame_idxs) * len(select_view_ids))

        use_frame_idxs = np.array(frame_idxs)[np.unique(sup_image_indices // len(select_view_ids))]
        
        for ii, frame_idx in enumerate(use_frame_idxs):
            for jj, select_id in enumerate(select_view_ids):
                if ii * len(select_view_ids) + jj not in sup_image_indices:
                    continue

                # - camera depth image
                depth_image = sample[f"{frame_idx:06d}.{depth_type}.{self.CAMERA_ID_TO_NAME[select_id]}.png"] / 100.0
                depth_image = torch.tensor(depth_image).float().unsqueeze(-1)
                
                if select_id > 2: # side view, pad height
                    depth_image_ = torch.zeros(1280, 1920, 1)
                    depth_image_[:depth_image.shape[0], :, :] = depth_image
                    depth_image = depth_image_

                all_depth_images_tensor.append(depth_image)

        return torch.stack(all_depth_images_tensor)

    def set_input_sup_frame_idx(self, reassemble_sample, starting_frame):
        """
            set the input and sup frame idx (json), and return all needed frame idxs for assembling data
        """
        if self.offset_unit == 'frame':
            use_frame_idxs = [starting_frame + offset for offset in self.input_frame_offsets] + \
                    [starting_frame + offset for offset in self.sup_frame_offsets]
            use_frame_idxs = sorted(list(set(use_frame_idxs)))

            # prepare for auto decoding
            reassemble_sample['input_frame_idx.json'] = json.dumps([starting_frame + offset for offset in self.input_frame_offsets]).encode('utf-8')
            reassemble_sample['sup_frame_idx.json'] = json.dumps([starting_frame + offset for offset in self.sup_frame_offsets]).encode('utf-8')

            return use_frame_idxs

        elif self.offset_unit == 'meter':
            # need to load the 'pose' data to calculate the distance
            pose_tarfile_path = reassemble_sample['__url__'].replace(self.attr_subfolders[0], 'pose')
            pose_sample = next(iter(wds.DataPipeline(
                wds.SimpleShardList(pose_tarfile_path), tarfile_samples
            )))
            pose_sample = self.decode_func(pose_sample)

            first_camera_lower_case = self.CAMERA_ID_TO_NAME[0].lower()
            max_frame_idx = max([int(x.split('.')[0]) for x in pose_sample.keys() if first_camera_lower_case in x])

            # build [K, 4, 4] matrix
            first_camera_poses = np.stack(
                [pose_sample[f"{frame_idx:06d}.pose.{first_camera_lower_case}.npy"] for frame_idx in range(max_frame_idx + 1)], axis=0
            )

            # calculate the distance to the starting frame
            distance_each_frame = np.linalg.norm(first_camera_poses[1:, :3, 3] - first_camera_poses[:-1, :3, 3], axis=-1) # [K-1]
            # add 0 to the first frame
            distance_each_frame = np.concatenate([np.array([0]), distance_each_frame], axis=0) # [K]
            distance_accumulate = np.cumsum(distance_each_frame) # [K]

            # select the frame idx that is closest to the distance, also larger than the starting frame
            distance_accumulate_from_starting_frame = distance_accumulate - distance_accumulate[starting_frame]
            # make negative distance to inf
            distance_accumulate_from_starting_frame[distance_accumulate_from_starting_frame < 0] = np.inf

            input_frame_idxs = [
                np.argmin(np.abs(distance_accumulate_from_starting_frame - offset)).tolist() for offset in self.input_frame_offsets
            ]

            sup_frame_idxs = [
                np.argmin(np.abs(distance_accumulate_from_starting_frame - offset)).tolist() for offset in self.sup_frame_offsets
            ]

            # if there is repeated frame idx, we need to skip this sample
            if len(set(input_frame_idxs)) != len(input_frame_idxs) or len(set(sup_frame_idxs)) != len(sup_frame_idxs):
                if not self.tolerent_shorter_offsets:
                    raise ValueError(f"Current sequence is too short to find the frame idxs with offsets: {self.input_frame_offsets} and {self.sup_frame_offsets}.")
            
            # prepare for auto decoding
            reassemble_sample['input_frame_idx.json'] = json.dumps(input_frame_idxs).encode('utf-8')
            reassemble_sample['sup_frame_idx.json'] = json.dumps(sup_frame_idxs).encode('utf-8')

            # print(f"input_frame_idxs: {input_frame_idxs}, sup_frame_idxs: {sup_frame_idxs}")

            use_frame_idxs = list(set(input_frame_idxs + sup_frame_idxs))
            use_frame_idxs.sort()

            return use_frame_idxs

        else:
            raise NotImplementedError(f"Unknown offset unit {self.offset_unit} !.")
        
    def __iter__(self):
        rank, world_size, worker_id = self.get_rank_worker()
        # 对于顺序遍历，不需要打乱，也不需要重复采样
        shards_this_rank = [tar for i, tar in enumerate(self.wds_url_collection) if i % world_size == rank]
        dataset_this_rank = wds.DataPipeline([
            wds.SimpleShardList(shards_this_rank),
            # 移除 shuffle 步骤
            wds.split_by_worker,
            tarfile_samples
        ])

        for sample in dataset_this_rank:
            # 遍历所有可能的起始帧
            for starting_frame in range(self.frame_start_num, self.last_starting_frame + 1):
                reassemble_sample = {}
                reassemble_sample['__key__'] = sample['__key__']
                reassemble_sample['__url__'] = sample['__url__']

                try:
                    use_frame_idxs = self.set_input_sup_frame_idx(reassemble_sample, starting_frame)
                except ValueError:
                    print(f"Skip sample {sample['__key__']} at starting frame {starting_frame}")
                    continue

                # 1) 收集主要图像数据
                primary_attr_subfolder = self.attr_subfolders[0]
                self._assemble_data(primary_attr_subfolder, use_frame_idxs, reassemble_sample, sample)

                # 收集其他子文件夹的数据
                for attr_subfolder in self.attr_subfolders[1:]:
                    tarfile_path = reassemble_sample['__url__'].replace(primary_attr_subfolder, attr_subfolder)
                    sample_other = next(iter(wds.DataPipeline(
                        wds.SimpleShardList(tarfile_path), tarfile_samples
                    )))
                    self._assemble_data(attr_subfolder, use_frame_idxs, reassemble_sample, sample_other)

                # 2) 对 reassemble_sample 进行解码
                reassemble_sample = self.decode_func(reassemble_sample)

                # 3) 数据变换
                try:
                    reassemble_sample = self.data_transform(reassemble_sample)
                except Exception as e:
                    print("Error in data transform:")
                    traceback.print_exc()
                    print(f"Skip sample: {sample['__key__']}")
                    continue

                yield reassemble_sample

    def data_transform(self, sample):
        data = {}
        data[DS.SHAPE_NAME] = sample['__url__'] + \
            "_with_input_frames_" + '_'.join([str(x) for x in sample['input_frame_idx.json']]) + \
            "_with_sup_frames_" + '_'.join([str(x) for x in sample['sup_frame_idx.json']])
        
        grid_to_world = self.crop_pc_gen_grid(sample)
        
        # data[DS.GRID_TO_FIRST_CAMERA_FLU] = grid_to_first_camera_flu
        data[DS.GRID_TO_WORLD] = grid_to_world
        data[DS.GRID_CROP_RANGE] = torch.tensor([self.grid_crop_bbox_min, self.grid_crop_bbox_max])

        data[DS.INPUT_PC] = "Generate on the fly from DS.INPUT_PC_RAW"
        data[DS.INPUT_PC_RAW] = sample[f"grid_raw.{self.fvdb_grid_type}.pth"] # real pc data to be voxelized

        if self.add_high_res_grid:
            raise NotImplementedError("High res grid is not implemented yet, generate on the fly")

        if DS.GT_SEMANTIC in self.spec:
            data[DS.GT_SEMANTIC] = "Generate on the fly from DS.INPUT_PC_RAW"

        if DS.IMAGES_INPUT in self.spec:
            input_image, input_mask, input_pose, input_intrinsic = \
                self.get_images(sample, sample['input_frame_idx.json'],
                                self.input_slect_ids)
            
            data[DS.IMAGES_INPUT] = input_image
            data[DS.IMAGES_INPUT_MASK] = input_mask
            data[DS.IMAGES_INPUT_POSE] = input_pose
            data[DS.IMAGES_INPUT_INTRINSIC] = input_intrinsic

        all_sup_img_num = len(sample['sup_frame_idx.json']) * len(self.sup_slect_ids)
        if self.n_image_per_iter_sup < all_sup_img_num:
            sup_image_indices = np.random.choice(all_sup_img_num, self.n_image_per_iter_sup, replace=False)
        else:
            sup_image_indices = np.arange(all_sup_img_num)

        if DS.IMAGES in self.spec:
            sup_image, sup_mask, sup_pose, sup_intrinsic = \
                self.get_images(sample, sample['sup_frame_idx.json'],
                                self.sup_slect_ids,
                                sup_image_indices)
            
            data[DS.IMAGES] = sup_image
            data[DS.IMAGES_MASK] = sup_mask
            data[DS.IMAGES_POSE] = sup_pose
            data[DS.IMAGES_INTRINSIC] = sup_intrinsic

        if DS.IMAGES_INPUT_DEPTH in self.spec:
            depth = self.get_depth_images(sample, sample['input_frame_idx.json'],
                                          self.input_slect_ids,
                                          None, depth_type=self.input_depth_type)
            data[DS.IMAGES_INPUT_DEPTH] = depth
        
        if DS.IMAGES_DEPTH_MONO_EST_RECTIFIED in self.spec:
            depth = self.get_depth_images(sample, sample['sup_frame_idx.json'],
                                          self.sup_slect_ids,
                                          sup_image_indices, depth_type='rectified_metric3d_depth')
            data[DS.IMAGES_DEPTH_MONO_EST_RECTIFIED] = depth
    
        if DS.IMAGES_DEPTH_LIDAR_PROJECT in self.spec:
            depth = self.get_depth_images(sample, sample['sup_frame_idx.json'],
                                          self.sup_slect_ids,
                                          sup_image_indices, depth_type='lidar_depth')
            data[DS.IMAGES_DEPTH_LIDAR_PROJECT] = depth

        if DS.IMAGES_DEPTH_ANYTHING_V2_DEPTH_INV in self.spec:
            depth = self.get_depth_images(sample, sample['sup_frame_idx.json'],
                                          self.sup_slect_ids,
                                          sup_image_indices, depth_type='depth_anything_v2_depth_inv')
            data[DS.IMAGES_DEPTH_ANYTHING_V2_DEPTH_INV] = depth

        if DS.IMAGES_NORMAL in self.spec:
            raise NotImplementedError("Normal is not implemented yet")

        if DS.MAPS_3D in self.spec:
            data[DS.MAPS_3D] = {}
            grid_to_world = sample['grid_to_world']
            world_to_grid = torch.inverse(grid_to_world)
            for map_type in self.map_types:
                if map_type != 'dense_road_surface':
                    # transform to grid coordinate
                    map_point_padded = torch.tensor(sample[f"{map_type}.npy"]).float()
                    map_point_padded = torch.cat([map_point_padded, torch.ones_like(map_point_padded[:,0:1])], dim=1)
                    data[DS.MAPS_3D][map_type] = torch.einsum('ij,nj->ni', world_to_grid, map_point_padded)[:,:3]

                else:
                    from scube.utils.vis_util import WAYMO_CATEGORY_NAMES
                    points = sample[f"grid_raw.{self.fvdb_grid_type}.pth"]['points_finest'] # [N_points, 3]
                    semantics = sample[f"grid_raw.{self.fvdb_grid_type}.pth"]['semantics_finest'] # [N_points]
                    road_semantic_label = WAYMO_CATEGORY_NAMES.index('ROAD')
                    road_points = points[semantics == road_semantic_label].numpy()

                    # it is too dense, when it is used in condition, the voxel size is usually 3.2m^3 or 1.6m^3
                    if self.fvdb_grid_type == 'vs01':
                        down_vs = 0.1
                    elif self.fvdb_grid_type == 'vs02':
                        down_vs = 0.2
                    elif self.fvdb_grid_type == 'vs04':
                        down_vs = 0.4
                    
                    # downsample the road points
                    road_points_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(road_points))
                    road_points_pcd_down = road_points_pcd.voxel_down_sample(voxel_size=down_vs)
                    road_points_tensor = torch.tensor(np.array(road_points_pcd_down.points)).float()

                    data[DS.MAPS_3D][map_type] = road_points_tensor

        if DS.BOXES_3D in self.spec:
            from scube.utils.vis_util import WAYMO_CATEGORY_NAMES
            from scube.utils.box_util import build_scene_bounding_boxes_from_all_object_info
            """
            we will consider static and dynamic objects together! 
            And we can just use all_object_info to build the BOXES_3D condition.
            """

            data[DS.BOXES_3D] = {}
            grid_to_world = sample['grid_to_world']
            world_to_grid = torch.inverse(grid_to_world)

            # create 3d bounding boxes from all_object_info
            all_car_dict = sample[f"{sample['input_frame_idx.json'][0]:06d}.all_object_info.json"]

            crop_half_range_canonical = (torch.tensor(self.grid_crop_bbox_max) - torch.tensor(self.grid_crop_bbox_min)) / 2
            bounding_box_in_grid = build_scene_bounding_boxes_from_all_object_info(
                all_car_dict, world_to_grid, crop_half_range_canonical
            )

            data[DS.BOXES_3D] = torch.tensor(bounding_box_in_grid)

            # print(f"shape of BOXES_3D: {data[DS.BOXES_3D].shape}")

            if data[DS.BOXES_3D].shape[0] > 0:
                _check_coplanar(data[DS.BOXES_3D].float())
                _check_nonzero(data[DS.BOXES_3D].float())


        return data
    

    def crop_pc_gen_grid(self, sample):
        raise NotImplementedError('implement it in the child class')
    
    def get_images(self, sample, frame_idxs, select_view_ids, sup_image_indices=None):
        raise NotImplementedError('implement it in the child class')
    
    def get_voxel(self):
        raise NotImplementedError('implement it in the child class')
