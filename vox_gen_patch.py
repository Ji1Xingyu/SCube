import fvdb.types
from torch.utils.data import DataLoader
import torch
import os
from scube.data.waymo_wds import WaymoWdsDataset
from scube.data.base import list_collate

from scube.utils.common_util import batch2device, get_default_parser, create_model_from_args
from scube.utils.voxel_util import create_fvdb_grid_w_semantic_from_points
from scube.data.base import DatasetSpec as DS
from tqdm import tqdm
import numpy as np
import gc


import fvdb
from fvdb import GridBatch
import torch.nn.utils.rnn as rnn_utils
import json
import setproctitle
setproctitle.setproctitle("xingyu_waymo2kitti")  # Change "save_triplane" to your preferred name

attr_subfolders = ['pose', 'intrinsic', 'pc_voxelsize_01', 'all_object_info']

# grid_crop_bbox_min = [-10.24, -51.2, -6.4]
# grid_crop_bbox_max = [92.16, 51.2, 6.4]

# with open('./waymo_split/official_train_w_dynamic_w_ego_motion_gt_30m_good_voxel.json', 'r') as f:
#     filelist = json.load(f)
# with open('./waymo_split/official_val_w_dynamic_w_ego_motion_gt_30m_good_voxel.json', 'r') as f:
#     filelist_val = json.load(f)
# yaml_data = {
#     'split': {
#         'train': filelist,
#         'val': filelist_val
#     }
# }
# import yaml
# with open('waymo_semcity.yaml', 'w') as f:
#     yaml.dump(yaml_data, f)

# aasdasd = 1


# rid_crop_bbox_min=[-10.24, -51.2, -12.8], grid_crop_bbox_max=[92.16, 51.2, 38.4]


def save_semantic_kitti(paths, pose, valid:np.ndarray, invalid:np.ndarray, labels:np.ndarray):
    dir_path, file_name = os.path.split(paths)

    vox_dir = os.path.join(dir_path, 'voxels')
    pos_dir = os.path.join(dir_path, 'pose')
    os.makedirs(vox_dir, exist_ok=True)
    os.makedirs(pos_dir, exist_ok=True)
    pose = pose.reshape(1, -1).cpu().numpy()


    bins = (labels>0).astype(int)
    occuluded = (~valid).astype(int)

    bin_path = os.path.join(vox_dir, file_name.replace('.pcd', '.bin'))
    lab_path = os.path.join(vox_dir, file_name.replace('.pcd', '.label'))
    occ_path = os.path.join(vox_dir, file_name.replace('.pcd', '.occuluded'))
    inv_path = os.path.join(vox_dir, file_name.replace('.pcd', '.invalid'))
    pos_path = os.path.join(pos_dir, file_name.replace('.pcd', '.npy'))

    labels.astype(np.uint16).tofile(lab_path)

    # compress
    def compress(voxel_bool_array):
        voxel_bool_array = voxel_bool_array.reshape(-1, 8)
        compressed =(voxel_bool_array[:, 0] << 7) | (voxel_bool_array[:, 1] << 6) | \
                    (voxel_bool_array[:, 2] << 5) | (voxel_bool_array[:, 3] << 4) | \
                    (voxel_bool_array[:, 4] << 3) | (voxel_bool_array[:, 5] << 2) | \
                    (voxel_bool_array[:, 6] << 1) | (voxel_bool_array[:, 7])
        return compressed
    occuluded = compress(occuluded)
    invalid = compress(invalid)
    bins = compress(bins)

    pose.astype(np.float32).tofile(pos_path)
    occuluded.astype(np.uint8).tofile(occ_path)
    invalid.astype(np.uint8).tofile(inv_path)
    bins.astype(np.uint8).tofile(bin_path)


def waymo2semkitti(ep_ts, batch, Nx, Ny, Nz, Minx, Miny, Minz):

    batch = batch2device(batch, device)
    generated_grids, generated_semantics = create_fvdb_grid_w_semantic_from_points(
            batch[DS.INPUT_PC_RAW]['points_finest'], 
            batch[DS.INPUT_PC_RAW]['semantics_finest'], 
            {'voxel_sizes': batch[DS.INPUT_PC_RAW]['grid_batch_kwargs_target']['voxel_sizes'][0],
                'origins': batch[DS.INPUT_PC_RAW]['grid_batch_kwargs_target']['origins'][0]},
            {'voxel_sizes': batch[DS.INPUT_PC_RAW]['grid_batch_kwargs_finest']['voxel_sizes'][0],
                'origins': batch[DS.INPUT_PC_RAW]['grid_batch_kwargs_finest']['origins'][0]},
            batch[DS.INPUT_PC_RAW]['extra_meshes'], 
        )
    
    for pcds, semantics, paths, pose in zip(generated_grids, generated_semantics, batch[DS.INPUT_PC_RAW]['save_name'], batch[DS.INPUT_PC_RAW]['grid_to_world']):

        # generate a dense voxel for ray tracing
        dense_grid = fvdb.gridbatch_from_dense(num_grids=1,
                                               dense_dims=[Nx, Ny, Nz],
                                               ijk_min=[Minx, Miny, Minz],
                                               voxel_sizes=batch[DS.INPUT_PC_RAW]['grid_batch_kwargs_target']['voxel_sizes'][0],
                                               origins=batch[DS.INPUT_PC_RAW]['grid_batch_kwargs_target']['origins'][0],
                                               device="cuda")
        
        num_dense_grids = dense_grid.num_voxels.item()
        x_range = torch.arange(Minx, Nx + Minx, device='cuda')  
        y_range = torch.arange(Miny, Ny + Miny, device='cuda')
        z_range = torch.arange(Minz, Nz + Minz, device='cuda')
        grid_x, grid_y, grid_z = torch.meshgrid(x_range, y_range, z_range, indexing='ij')
        vox_coords = torch.stack([grid_x.flatten(), grid_y.flatten(), grid_z.flatten()], dim=1)  # shape: [num_voxels, 3]
        vox_coords = vox_coords.to('cuda')
        vox_jt = fvdb.JaggedTensor(vox_coords)  

        world_coords_jt = dense_grid.grid_to_world(vox_jt.float())
        if ep_ts.dim() == 1:
            ep_ts = ep_ts.unsqueeze(0)  
        ep_ts_exp = ep_ts.expand(world_coords_jt.jdata.shape[0], -1)  
        dirs = ep_ts_exp - world_coords_jt.jdata
        dir_norm = dirs / torch.norm(dirs, dim=1, keepdim=True)
        rayOrigins_jt = fvdb.JaggedTensor(ep_ts_exp)
        rayDirections_jt = fvdb.JaggedTensor(dir_norm)
        # ray start from end point, end at all voxels in the grid
        vox_coord_jt, _ = dense_grid.voxels_along_rays(rayOrigins_jt, rayDirections_jt, max_voxels=-1, eps=0.0)
        # del rayOrigins_jt, rayDirections_jt
        ray_offset = vox_coord_jt.joffsets
        idx_ray_occupied_jt = pcds.coords_in_active_voxel(vox_coord_jt.jdata)
        idx_grid_jt = dense_grid.ijk_to_index(vox_coord_jt.jdata)
        # del vox_coord_jt
        torch.cuda.empty_cache()
        # test: new algo
        global_idx = torch.arange(idx_grid_jt.jdata.shape[0], device=idx_grid_jt.device)
        num_rays = num_dense_grids
        ray_lengths = ray_offset[1:] - ray_offset[:-1] 
        groups = torch.bucketize(global_idx, ray_offset, right=False) - 1

        groups[0] = 0

        global_idx_true = global_idx[idx_ray_occupied_jt.jdata]          
        groups_true   = groups[idx_ray_occupied_jt.jdata]                  
        relative_idx_true = global_idx_true - ray_offset[groups_true]
        relative_idx = global_idx - ray_offset[groups]

        first_true_relative = torch.full((num_rays,), torch.iinfo(torch.int64).max, device=idx_grid_jt.device)
        first_true_relative = first_true_relative.scatter_reduce(0, groups_true, relative_idx_true, reduce="amin", include_self=False)

        mask_no_true = first_true_relative == torch.iinfo(torch.int64).max
        first_true_relative[mask_no_true] = ray_lengths[mask_no_true]

        min_to_prev_offset = first_true_relative
        # ***************************CHANGE HERE FOR RAY TRAVERSING VOXEL NUMS************************************
        mask_up_to_first_true = relative_idx <= min_to_prev_offset[groups]
        # need to inject the idx_ray back to idx_grid
        # like for idx in range(vox_coord_jt.jdata.shape[0]) : idx_grid_occupied[dense_grid.ijk_to_index(vox_coord_jt.jdata[idx]).jdata[0]] = mask_up_to_first_true[idx]
        
        labels = torch.full((num_dense_grids,), -1, dtype=semantics.dtype, device='cuda')
        labels[dense_grid.ijk_to_index(pcds.ijk).jdata] = semantics

        dense_valid = torch.full((num_dense_grids,), False, dtype=torch.bool, device='cuda')
        dense_valid[idx_grid_jt.jdata] = mask_up_to_first_true

        # invalid shouble be those, not in labels & not valid
        dense_invalid = ~dense_valid
        dense_invalid[labels>=0] = False

        # save_pcd_manual(paths, frameIdx, torch_dict)

        # test: visualize either valid or labeled
        # pt = dense_grid.grid_to_world(dense_grid.ijk.jdata[labels>=0].float())
        # valid_ijk = dense_grid.ijk.jdata[dense_valid>0]
        # valid_points = dense_grid.grid_to_world(valid_ijk.float())
        # valid_sem = torch.full((valid_points.jdata.shape[0],), 0, dtype=torch.int64)
        # valid_sem = valid_sem.cuda()
        # allval = torch.cat((pt.jdata, valid_points.jdata), dim=0)
        # allsem = torch.cat((semantics, valid_sem), dim=0)
        # ts_dict = {'points': allval, 'semantics':allsem}
        # torch.save(ts_dict, './occ.pcd.vs01.pth')

        # test: visualize those valid that in labeled
        # labels[dense_invalid == True] = -1
        # pt = dense_grid.grid_to_world(dense_grid.ijk.jdata[dense_invalid == False].float())
        # sem = torch.full((pt.jdata.shape[0],), 1, dtype=semantics.dtype, device='cuda')
        # ts_dict = {'points': pt.jdata, 'semantics':sem}
        # torch.save(ts_dict, './occ.pcd.vs01.pth')

        

        ijk = dense_grid.ijk.jdata
        ijk += torch.tensor([Nx/2, Ny/2, Nz/2], device=ijk.device, dtype=torch.int32)

        reshape_labels = torch.zeros((Nx, Ny, Nz), dtype=labels.dtype, device='cpu')
        reshape_invalid = torch.zeros((Nx, Ny, Nz), dtype=torch.bool, device='cpu')
        reshape_valid = torch.zeros((Nx, Ny, Nz), dtype=torch.bool, device='cpu')

        reshape_labels[ijk[:, 0], ijk[:, 1], ijk[:, 2]]     = labels.cpu()
        reshape_invalid[ijk[:, 0], ijk[:, 1], ijk[:, 2]]    = dense_invalid.cpu()
        reshape_valid[ijk[:, 0], ijk[:, 1], ijk[:, 2]]      = dense_valid.cpu()



        save_semantic_kitti(paths, pose,
                            valid=reshape_valid.numpy(), 
                            invalid=reshape_invalid.numpy(), 
                            labels=reshape_labels.numpy())
        

if __name__ == "__main__":
    # Example usage:
    # Replace with your directory containing files and desired output JSON path
    
    parser = get_default_parser()
    parser.add_argument('--in_path', type=str, default='./waymo_semcity/')
    parser.add_argument('--wds_scene_list_file', type=str, default='./waymo_semcity/')
    parser.add_argument('--out_path', type=str, default='./waymo_semcity/')
    args = parser.parse_args()
    
    grid_crop_bbox_min = [-5.12, -25.6, -3.2]
    grid_crop_bbox_max = [46.08, 25.6, 3.2]
    end_point = [-(grid_crop_bbox_min[0] + grid_crop_bbox_max[0]) / 2.0,
                -(grid_crop_bbox_min[1] + grid_crop_bbox_max[1]) / 2.0,
                -(grid_crop_bbox_min[2] + grid_crop_bbox_max[2]) / 2.0]

    Nx = 256
    Ny = 256
    Nz = 32
    Minx = -128
    Miny = -128
    Minz = -16
    
    dataset_c = WaymoWdsDataset(
            wds_root_url= './waymo_webdataset',
            # wds_scene_list_file= './folders.json',
            wds_scene_list_file= './waymo_split/train.json',
            attr_subfolders=attr_subfolders,
            spec=[DS.GT_SEMANTIC],
            split='train',
            fvdb_grid_type='vs02',
            grid_crop_augment=True,
            grid_crop_bbox_min=grid_crop_bbox_min,
            grid_crop_bbox_max=grid_crop_bbox_max,
            input_depth_type='rectified_metric3d_depth_affine',
            replace_all_car_with_cad = True,
            add_road_line_to_GT=False,
            save_path = args.save_path,
    )
    
    dataloader_c = DataLoader(
        dataset_c,
        batch_size=1,
        shuffle=True if not isinstance(dataset_c, torch.utils.data.IterableDataset) else False,
        num_workers=1,
        collate_fn=list_collate
    )
    
    device = torch.device("cuda")
    frameIdx = 0

    # 
    bbx = -np.array(grid_crop_bbox_min) + np.array(grid_crop_bbox_max)
    ep_ts = torch.tensor(end_point, device='cuda')
    ep_ts = ep_ts.unsqueeze(0)

    for batch_idx, batch in enumerate(tqdm(dataloader_c)):
        # with torch.no_grad():
        #     waymo2semkitti(ep_ts=ep_ts, batch=batch)
        # torch.cuda.empty_cache()
        # gc.collect()
        
        batch = batch2device(batch, device)
         
        generated_grids, generated_semantics = create_fvdb_grid_w_semantic_from_points(
                batch[DS.INPUT_PC_RAW]['points_finest'], 
                batch[DS.INPUT_PC_RAW]['semantics_finest'], 
                {'voxel_sizes': batch[DS.INPUT_PC_RAW]['grid_batch_kwargs_target']['voxel_sizes'][0],
                    'origins': batch[DS.INPUT_PC_RAW]['grid_batch_kwargs_target']['origins'][0]},
                {'voxel_sizes': batch[DS.INPUT_PC_RAW]['grid_batch_kwargs_finest']['voxel_sizes'][0],
                    'origins': batch[DS.INPUT_PC_RAW]['grid_batch_kwargs_finest']['origins'][0]},
                batch[DS.INPUT_PC_RAW]['extra_meshes'], 
            )
        
        for pcds, semantics, paths, pose in zip(generated_grids, generated_semantics, batch[DS.INPUT_PC_RAW]['save_name'], batch[DS.INPUT_PC_RAW]['grid_to_world']):

            # generate a dense voxel for ray tracing
            dense_grid = fvdb.gridbatch_from_dense(num_grids=1,
                                                dense_dims=[Nx, Ny, Nz],
                                                ijk_min=[Minx, Miny, Minz],
                                                voxel_sizes=batch[DS.INPUT_PC_RAW]['grid_batch_kwargs_target']['voxel_sizes'][0],
                                                origins=batch[DS.INPUT_PC_RAW]['grid_batch_kwargs_target']['origins'][0],
                                                device="cuda")
            
            num_dense_grids = dense_grid.num_voxels.item()
            x_range = torch.arange(Minx, Nx + Minx, device='cuda')  
            y_range = torch.arange(Miny, Ny + Miny, device='cuda')
            z_range = torch.arange(Minz, Nz + Minz, device='cuda')
            grid_x, grid_y, grid_z = torch.meshgrid(x_range, y_range, z_range, indexing='ij')
            vox_coords = torch.stack([grid_x.flatten(), grid_y.flatten(), grid_z.flatten()], dim=1)  # shape: [num_voxels, 3]
            vox_coords = vox_coords.to('cuda')
            vox_jt = fvdb.JaggedTensor(vox_coords)  

            world_coords_jt = dense_grid.grid_to_world(vox_jt.float())
            if ep_ts.dim() == 1:
                ep_ts = ep_ts.unsqueeze(0)  
            ep_ts_exp = ep_ts.expand(world_coords_jt.jdata.shape[0], -1)  
            dirs = ep_ts_exp - world_coords_jt.jdata
            dir_norm = dirs / torch.norm(dirs, dim=1, keepdim=True)
            rayOrigins_jt = fvdb.JaggedTensor(ep_ts_exp)
            rayDirections_jt = fvdb.JaggedTensor(dir_norm)
            # ray start from end point, end at all voxels in the grid
            vox_coord_jt, _ = dense_grid.voxels_along_rays(rayOrigins_jt, rayDirections_jt, max_voxels=-1, eps=0.0)
            # del rayOrigins_jt, rayDirections_jt
            ray_offset = vox_coord_jt.joffsets
            idx_ray_occupied_jt = pcds.coords_in_active_voxel(vox_coord_jt.jdata)
            idx_grid_jt = dense_grid.ijk_to_index(vox_coord_jt.jdata)
            # del vox_coord_jt
            torch.cuda.empty_cache()
            # test: new algo
            global_idx = torch.arange(idx_grid_jt.jdata.shape[0], device=idx_grid_jt.device)
            num_rays = num_dense_grids
            ray_lengths = ray_offset[1:] - ray_offset[:-1] 
            groups = torch.bucketize(global_idx, ray_offset, right=False) - 1

            groups[0] = 0

            global_idx_true = global_idx[idx_ray_occupied_jt.jdata]          
            groups_true   = groups[idx_ray_occupied_jt.jdata]                  
            relative_idx_true = global_idx_true - ray_offset[groups_true]
            relative_idx = global_idx - ray_offset[groups]

            first_true_relative = torch.full((num_rays,), torch.iinfo(torch.int64).max, device=idx_grid_jt.device)
            first_true_relative = first_true_relative.scatter_reduce(0, groups_true, relative_idx_true, reduce="amin", include_self=False)

            mask_no_true = first_true_relative == torch.iinfo(torch.int64).max
            first_true_relative[mask_no_true] = ray_lengths[mask_no_true]

            min_to_prev_offset = first_true_relative
            # ***************************CHANGE HERE FOR RAY TRAVERSING VOXEL NUMS************************************
            mask_up_to_first_true = relative_idx <= min_to_prev_offset[groups]
            # need to inject the idx_ray back to idx_grid
            # like for idx in range(vox_coord_jt.jdata.shape[0]) : idx_grid_occupied[dense_grid.ijk_to_index(vox_coord_jt.jdata[idx]).jdata[0]] = mask_up_to_first_true[idx]
            
            labels = torch.full((num_dense_grids,), -1, dtype=semantics.dtype, device='cuda')
            labels[dense_grid.ijk_to_index(pcds.ijk).jdata] = semantics

            dense_valid = torch.full((num_dense_grids,), False, dtype=torch.bool, device='cuda')
            dense_valid[idx_grid_jt.jdata] = mask_up_to_first_true

            # invalid shouble be those, not in labels & not valid
            dense_invalid = ~dense_valid
            dense_invalid[labels>=0] = False

            # save_pcd_manual(paths, frameIdx, torch_dict)

            # test: visualize either valid or labeled
            # pt = dense_grid.grid_to_world(dense_grid.ijk.jdata[labels>=0].float())
            # valid_ijk = dense_grid.ijk.jdata[dense_valid>0]
            # valid_points = dense_grid.grid_to_world(valid_ijk.float())
            # valid_sem = torch.full((valid_points.jdata.shape[0],), 0, dtype=torch.int64)
            # valid_sem = valid_sem.cuda()
            # allval = torch.cat((pt.jdata, valid_points.jdata), dim=0)
            # allsem = torch.cat((semantics, valid_sem), dim=0)
            # ts_dict = {'points': allval, 'semantics':allsem}
            # torch.save(ts_dict, './occ.pcd.vs01.pth')

            # test: visualize those valid that in labeled
            # labels[dense_invalid == True] = -1
            # pt = dense_grid.grid_to_world(dense_grid.ijk.jdata[dense_invalid == False].float())
            # sem = torch.full((pt.jdata.shape[0],), 1, dtype=semantics.dtype, device='cuda')
            # ts_dict = {'points': pt.jdata, 'semantics':sem}
            # torch.save(ts_dict, './occ.pcd.vs01.pth')

            

            ijk = dense_grid.ijk.jdata
            ijk += torch.tensor([Nx/2, Ny/2, Nz/2], device=ijk.device, dtype=torch.int32)

            reshape_labels = torch.zeros((Nx, Ny, Nz), dtype=labels.dtype, device='cpu')
            reshape_invalid = torch.zeros((Nx, Ny, Nz), dtype=torch.bool, device='cpu')
            reshape_valid = torch.zeros((Nx, Ny, Nz), dtype=torch.bool, device='cpu')

            reshape_labels[ijk[:, 0], ijk[:, 1], ijk[:, 2]]     = labels.cpu()
            reshape_invalid[ijk[:, 0], ijk[:, 1], ijk[:, 2]]    = dense_invalid.cpu()
            reshape_valid[ijk[:, 0], ijk[:, 1], ijk[:, 2]]      = dense_valid.cpu()



            save_semantic_kitti(paths, pose,
                                valid=reshape_valid.numpy(), 
                                invalid=reshape_invalid.numpy(), 
                                labels=reshape_labels.numpy())

        frameIdx += 1

