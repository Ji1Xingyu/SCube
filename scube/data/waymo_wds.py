# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import json
import os
import random
import sys
import time

import braceexpand
import fvdb
import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
import webdataset as wds
from pathlib import Path

sys.path.append(Path(__file__).parent.parent.as_posix())

from fvdb import GridBatch
from omegaconf import OmegaConf
from scube.data.base import DatasetSpec as DS
from scube.data.base_wds import BaseWdsDataset
from scube.utils.vis_util import WAYMO_CATEGORY_NAMES

CAMERA_ID_TO_NAME = {
    0: 'front',
    1: 'front_left',
    2: 'front_right',
    3: 'side_left',
    4: 'side_right'
}


class WaymoWdsDataset(BaseWdsDataset):
    def __init__(self, wds_root_url, wds_scene_list_file, attr_subfolders=['image_front'],
                spec=None, split='train', 
                frame_start_num=0, frame_end_num=190,
                grid_crop_bbox_min=[-10.24, -51.2, -12.8], grid_crop_bbox_max=[92.16, 51.2, 38.4],
                input_slect_ids=[0,1,2], input_frame_offsets=[0], offset_unit='frame',
                sup_slect_ids=[0,1,2], sup_frame_offsets=[0], n_image_per_iter_sup=None,
                fvdb_grid_type='vs01', finest_voxel_size_goal='vs01',
                val_starting_frame=100, map_types=None,
                skip_on_error=False, random_seed=0, shuffle_buffer=128, input_depth_type='metric3d_depth',
                custom_name='scene_base', hparams=None, s3_cfg=None, add_high_res_grid=False, 
                tolerent_shorter_offsets=False,
                grid_crop_augment=False, grid_crop_augment_range=[0, 0, 0],
                replace_all_car_with_cad=False,
                add_road_line_to_GT=False, save_path = './waymo_semcity/',
                **kwargs):
        
        super().__init__(wds_root_url, wds_scene_list_file, attr_subfolders,
                        spec, split, frame_start_num, frame_end_num,
                        grid_crop_bbox_min, grid_crop_bbox_max, input_slect_ids, input_frame_offsets,
                        offset_unit, sup_slect_ids, sup_frame_offsets, n_image_per_iter_sup,
                        fvdb_grid_type, finest_voxel_size_goal, val_starting_frame, map_types,
                        skip_on_error, random_seed, shuffle_buffer, input_depth_type,
                        custom_name, hparams, s3_cfg, add_high_res_grid, tolerent_shorter_offsets, 
                        grid_crop_augment, grid_crop_augment_range,
                        replace_all_car_with_cad, add_road_line_to_GT,
                        **kwargs)
        
        self.CAMERA_ID_TO_NAME = CAMERA_ID_TO_NAME

    def crop_pc_gen_grid(self, sample):
        """
        Rather than reading scene-level fvdb grid, here we read from pc and generate the grid on the fly.

        Args:s
            cam2world: the front camera c2w (note that opencv convention, RDF) matrix
                in the first frame from input frames

            grid's coordinate is grid_to_world (FLU convention)

        """
        pc_names = [x for x in sample.keys() if 'pcd' in x]
        if len(pc_names) == 0:
            cam2world = sample[f"{sample['input_frame_idx.json'][0]:06d}.pose.front.npy"].astype(np.float32)
            cam2world = torch.from_numpy(cam2world)
            cam2world_FLU = torch.cat([cam2world[:,2:3], -cam2world[:,0:1], -cam2world[:,1:2], cam2world[:,3:4]], axis=1) # opencv -> FLU
            camera_pos = cam2world_FLU[:3, 3]
            camera_front = cam2world_FLU[:3, 0] # unit 
            camera_left = cam2world_FLU[:3, 1] # unit 
            camera_up = cam2world_FLU[:3, 2] # unit 

            new_grid_pos = camera_pos + \
                            camera_front * (self.grid_crop_bbox_min[0] + self.grid_crop_bbox_max[0]) / 2 + \
                            camera_left * (self.grid_crop_bbox_min[1] + self.grid_crop_bbox_max[1]) / 2 + \
                            camera_up * (self.grid_crop_bbox_min[2] + self.grid_crop_bbox_max[2]) / 2

            rnd = []

            if self.grid_crop_augment and self.split == 'train':
                rnd = [random.uniform(-self.grid_crop_augment_range[0], self.grid_crop_augment_range[0]),
                                            random.uniform(-self.grid_crop_augment_range[1], self.grid_crop_augment_range[1]),
                                            random.uniform(-self.grid_crop_augment_range[2], self.grid_crop_augment_range[2])]
            new_grid_pos += torch.tensor(rnd)
        
            grid2world = torch.clone(cam2world_FLU)
            grid2world[:3, 3] = new_grid_pos

            return grid2world
        
        
        assert len(pc_names) == 1, 'only need to put the finest pc in the scene'
        # assert self.finest_voxel_size_goal in pc_names[0], 'only need to input the finest pc in the scene,' + \
        #     f'but {self.finest_voxel_size_goal} is not in {pc_names[0]}'
        assert 'vs01' in pc_names[0], 'I think we only have this type of pc...'
        pc_name = pc_names[0]


        cam2world = sample[f"{sample['input_frame_idx.json'][0]:06d}.pose.front.npy"].astype(np.float32)
        cam2world = torch.from_numpy(cam2world)
        cam2world_FLU = torch.cat([cam2world[:,2:3], -cam2world[:,0:1], -cam2world[:,1:2], cam2world[:,3:4]], axis=1) # opencv -> FLU
        camera_pos = cam2world_FLU[:3, 3]
        camera_front = cam2world_FLU[:3, 0] # unit 
        camera_left = cam2world_FLU[:3, 1] # unit 
        camera_up = cam2world_FLU[:3, 2] # unit 

        new_grid_pos = camera_pos + \
                        camera_front * (self.grid_crop_bbox_min[0] + self.grid_crop_bbox_max[0]) / 2 + \
                        camera_left * (self.grid_crop_bbox_min[1] + self.grid_crop_bbox_max[1]) / 2 + \
                        camera_up * (self.grid_crop_bbox_min[2] + self.grid_crop_bbox_max[2]) / 2

        rnd = []

        if self.grid_crop_augment and self.split == 'train':
            rnd = [random.uniform(-self.grid_crop_augment_range[0], self.grid_crop_augment_range[0]),
                                          random.uniform(-self.grid_crop_augment_range[1], self.grid_crop_augment_range[1]),
                                          random.uniform(-self.grid_crop_augment_range[2], self.grid_crop_augment_range[2])]
            new_grid_pos += torch.tensor(rnd)
        
        grid2world = torch.clone(cam2world_FLU)
        grid2world[:3, 3] = new_grid_pos
        world2grid = torch.inverse(grid2world)
        
        crop_half_range_canonical = (torch.tensor(self.grid_crop_bbox_max) - torch.tensor(self.grid_crop_bbox_min)) / 2
        
        crop_center = (torch.tensor(self.grid_crop_bbox_max) + torch.tensor(self.grid_crop_bbox_min)) / 2


        # retrieve the point cloud data
        points_static = sample[pc_name]['points']
        semantics_static = sample[pc_name]['semantics']

        pc_to_world = sample[pc_name]['pc_to_world']
        print(pc_to_world)
        pc2grid = world2grid @ pc_to_world
        # under grid coordinate
        points = torch.einsum('ij,nj->ni', pc2grid, torch.cat([points_static, torch.ones_like(points_static[:,0:1])], dim=1))[:,:3]
        semantics = semantics_static

        # placeholder
        all_car_meshes_vertices = np.zeros((0,3))
        all_car_meshes_faces = np.zeros((0,3))

        if self.replace_all_car_with_cad:
            from scube.utils.mesh_util import \
                build_scene_mesh_from_all_object_info
            all_object_dict = sample[f"{sample['input_frame_idx.json'][0]:06d}.all_object_info.json"]

            all_car_meshes_vertices, all_car_meshes_faces = build_scene_mesh_from_all_object_info(
                all_object_dict, world2grid, crop_half_range_canonical, plyfile='assets/car.ply'
            )

            # we need remove all the car points
            non_car_mask = (semantics != WAYMO_CATEGORY_NAMES.index('CAR')) & \
                           (semantics != WAYMO_CATEGORY_NAMES.index('TRUCK')) & \
                            (semantics != WAYMO_CATEGORY_NAMES.index('BUS')) & \
                            (semantics != WAYMO_CATEGORY_NAMES.index('OTHER_VEHICLE'))

            points = points[non_car_mask]
            semantics = semantics[non_car_mask]

        else:
            """
            else, we use accumulated LiDAR points for each dynamic object. (static object is already accumulated in the points_static)
            """
            points_dynamic = torch.zeros((0,3))
            semantics_dynamic = torch.zeros((0,))

            dynamic_objects_transformations = sample[f"{sample['input_frame_idx.json'][0]:06d}.dynamic_object_transformation.npz"]
            for gid, tfm_to_world in dynamic_objects_transformations.items():
                tfm_to_world = torch.from_numpy(tfm_to_world).float()

                points_in_canonical = sample['dynamic_object_points_canonical.npz'][f'{gid}_xyz']
                point_semantic = sample['dynamic_object_points_canonical.npz'][f'{gid}_semantic_waymo'] # numpy array
                point_semantic = point_semantic.item() # convert to python scaler
                points_in_canonical = torch.from_numpy(points_in_canonical).float()
                points_in_world = torch.einsum('ij,nj->ni', tfm_to_world[:3,:3], points_in_canonical) + tfm_to_world[:3,3]

                points_dynamic = torch.cat([points_dynamic, points_in_world], dim=0)
                semantics_dynamic = torch.cat([semantics_dynamic, torch.tensor([point_semantic] * points_in_world.shape[0])], dim=0)

            # ! note that these points are in original waymo world coordinate, while static points are in first vehicle frame
            points_dynamic = torch.einsum('ij,nj->ni', world2grid, torch.cat([points_dynamic, torch.ones_like(points_dynamic[:,0:1])], dim=1))[:,:3]

            # merge static and dynamic points
            points = torch.cat([points, points_dynamic], dim=0)
            semantics = torch.cat([semantics, semantics_dynamic], dim=0).to(torch.int32)

        if self.add_road_line_to_GT:
            raise NotImplementedError
        else:
            pass

        # crop the point cloud
        crop_mask = (points[:,0] > -crop_half_range_canonical[0]) & \
                    (points[:,0] < crop_half_range_canonical[0]) & \
                    (points[:,1] > -crop_half_range_canonical[1]) & \
                    (points[:,1] < crop_half_range_canonical[1]) & \
                    (points[:,2] > -crop_half_range_canonical[2]) & \
                    (points[:,2] < crop_half_range_canonical[2])

        cropped_points = points[crop_mask]
        cropped_semantics = semantics[crop_mask]

        centered_points = cropped_points - crop_center

        # create the grid on the fly. vs01 = voxel size 0.1m
        if self.fvdb_grid_type == 'vs01':
            voxel_sizes_target = torch.tensor([0.1,0.1,0.1])
        elif self.fvdb_grid_type == 'vs02':
            voxel_sizes_target = torch.tensor([0.2,0.2,0.2])
        elif self.fvdb_grid_type == 'vs04':
            voxel_sizes_target = torch.tensor([0.4,0.4,0.4])
        else:
            raise ValueError(f"Unknown fvdb grid type: {self.fvdb_grid_type}")
        origins_target = voxel_sizes_target / 2
        grid_batch_kwargs_target = {'voxel_sizes': voxel_sizes_target, 'origins': origins_target}

        # we also need our finest goal voxel size
        if self.finest_voxel_size_goal == 'vs01':
            voxel_sizes_finest = torch.tensor([0.1,0.1,0.1])
        elif self.finest_voxel_size_goal == 'vs02':
            voxel_sizes_finest = torch.tensor([0.2,0.2,0.2])
        elif self.finest_voxel_size_goal == 'vs04':
            voxel_sizes_finest = torch.tensor([0.4,0.4,0.4])
        else:
            raise ValueError(f"Unknown finest voxel size goal: {self.finest_voxel_size_goal}")
        origins_finest = voxel_sizes_finest / 2 
        grid_batch_kwargs_finest = {'voxel_sizes': voxel_sizes_finest, 'origins': origins_finest}

        sample['grid_to_world'] = grid2world

        assert cropped_points.shape[0] == cropped_semantics.shape[0], 'points and msemantics should have the same length'
        assert cropped_points.shape[0] > 0, 'no points in the cropped point cloud'

        grid_name_raw = f'grid_raw.{self.fvdb_grid_type}.pth'

        # test
        dir_path = self.save_path + sample['__key__']

        save_path = os.path.join(dir_path, f"{sample['input_frame_idx.json'][0]:06d}.pcd")

        sample[grid_name_raw] = {'points_finest': cropped_points, 'semantics_finest': cropped_semantics, 
                                 'grid_batch_kwargs_target': grid_batch_kwargs_target,
                                 'grid_batch_kwargs_finest': grid_batch_kwargs_finest,
                                 'idx' : sample['input_frame_idx.json'],
                                 'save_name': save_path,
                                 'grid_to_world': grid2world}

        # already the grid coordinate
        sample[grid_name_raw]['extra_meshes'] = \
            {'vertices': torch.tensor(all_car_meshes_vertices).float(), 'faces': torch.tensor(all_car_meshes_faces).int()}
        

        return grid2world


    def get_images(self, sample, frame_idxs, select_view_ids, sup_image_indices=None):
        all_images_tensor = []
        all_masks_tensor = []
        all_poses_tensor = []
        all_intrinsics_tensor = []

        if sup_image_indices is None:
            sup_image_indices = np.arange(len(frame_idxs) * len(select_view_ids))
        
        for ii, frame_idx in enumerate(frame_idxs):
            for jj, select_id in enumerate(select_view_ids):
                if ii * len(select_view_ids) + jj not in sup_image_indices:
                    continue

                # - camera pose (camera to grid)
                camera_to_world = sample[f"{frame_idx:06d}.pose.{CAMERA_ID_TO_NAME[select_id]}.npy"].astype(np.float32)
                camera_to_world = torch.from_numpy(camera_to_world)
                grid_to_world = sample['grid_to_world']
                cam2grid = torch.inverse(grid_to_world) @ camera_to_world
                all_poses_tensor.append(cam2grid)

                # - camera intrinsics
                intrinsic = sample[f"intrinsic.{CAMERA_ID_TO_NAME[select_id]}.npy"].astype(np.float32)
                intrinsic = torch.from_numpy(intrinsic)
                if select_id > 2: # side view, we change the intrinsic h to 1280
                    intrinsic[5] = 1280

                all_intrinsics_tensor.append(intrinsic)
                
                # - camera image
                image = sample[f"{frame_idx:06d}.image.{CAMERA_ID_TO_NAME[select_id]}.jpg"]
                image = torch.tensor(image) / 255.0
                if select_id > 2: # side view, we pad the image [H, W, 3] -> [1280, W, 3]
                    image_ = torch.zeros((1280, image.shape[1], 3), dtype=torch.float32)
                    image_[:image.shape[0], :, :] = image
                    image = image_

                all_images_tensor.append(image)

                # - camera mask, all kinds of masks.
                # (1) foreground mask from segmentation:    0 for background, 1 for foreground,
                # (2) non dynamic mask:                     leave it all 1 since we not use dynamic scene in appearance reconstruction. 
                # (3) non padding mask:                     0 for padding, 1 for non-padding
                # (4) foreground mask from grid:            0 for background, 1 for foreground. generate on the fly.
                mask = torch.ones(1280, 1920, 4, dtype=torch.bool)

                # (1) foreground mask from segmentation
                skymask = torch.tensor(sample[f"{frame_idx:06d}.skymask.{CAMERA_ID_TO_NAME[select_id]}.png"])
                foreground_mask_from_seg = skymask == 0
                mask[:skymask.shape[0],:,0] = foreground_mask_from_seg # the first channel stores foreground mask

                # (3) padding mask
                if select_id > 2: # side view, padding mask.
                    mask[skymask.shape[0]:, :, 2] = 0
                    
                all_masks_tensor.append(mask)


        return torch.stack(all_images_tensor), torch.stack(all_masks_tensor), torch.stack(all_poses_tensor), torch.stack(all_intrinsics_tensor)