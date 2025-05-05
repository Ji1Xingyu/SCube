import fvdb.types
from torch.utils.data import DataLoader
import torch
import os
from scube.data.waymo_wds import WaymoWdsDataset
from scube.data.base import list_collate
import io

# from scube.utils.common_util import batch2device, get_default_parser, create_model_from_args
from scube.utils.voxel_util import create_fvdb_grid_w_semantic_from_points
from scube.data.base import DatasetSpec as DS
from tqdm import tqdm
import numpy as np
import gc
import tarfile

import fvdb
from fvdb import GridBatch
import torch.nn.utils.rnn as rnn_utils
import json
import setproctitle
setproctitle.setproctitle("xingyu_waymo2kitti")  # Change "save_triplane" to your preferred name

from tqdm import tqdm
import shutil
for path in tqdm(os.listdir('/data2/xingyu/waymo_semcity')):
    vox_path = os.path.join('/data2/xingyu/waymo_semcity', path, 'voxels')
    for file in os.listdir(vox_path):
        if file.endswith('occ.label'):
            occ_path = os.path.join(vox_path, file.replace('_occ.label', '.occ.label'))
            os.rename(os.path.join(vox_path, file), occ_path)
        if file.endswith('_seed.label'):
            seed_path = os.path.join(vox_path, file.replace('_seed.label', '.seed.label'))
            os.rename(os.path.join(vox_path, file), seed_path)
        if file.endswith('all.invalid'):
            invalid_path = os.path.join(vox_path, file.replace('all.invalid', '.occ.invalid'))
            os.rename(os.path.join(vox_path, file), invalid_path)
        if file.endswith('seed.invalid'):
            seed_invalid_path = os.path.join(vox_path, file.replace('seed.invalid', '.seed.invalid'))
            os.rename(os.path.join(vox_path, file), seed_invalid_path)
    asdasdasd = 1

attr_subfolders = ['pose', 'intrinsic', 'pc_voxelsize_01',
                        #    'image_front', 'image_front_left', 'image_front_right', 'image_side_left', 'image_side_right',
                            # 'skymask_front', 'skymask_front_left', 'skymask_front_right', 'skymask_side_left', 'skymask_side_right',
                            # 'rectified_metric3d_depth_affine_100_front', 
                            # 'rectified_metric3d_depth_affine_100_front_left', 
                            # 'rectified_metric3d_depth_affine_100_front_right', 
                            # 'rectified_metric3d_depth_affine_100_side_left', 
                            # 'rectified_metric3d_depth_affine_100_side_right',
                            'all_object_info'
]

# grid_crop_bbox_min = [-10.24, -51.2, -6.4]
# grid_crop_bbox_max = [92.16, 51.2, 6.4]

# grid_crop_bbox_min = [-5.12, -25.6, -3.2]
# grid_crop_bbox_max = [46.08, 25.6, 3.2]

# rid_crop_bbox_min=[-10.24, -51.2, -12.8], grid_crop_bbox_max=[92.16, 51.2, 38.4]

grid_crop_bbox_min = [-25.6, -25.6, -3.2]
grid_crop_bbox_max = [25.6, 25.6, 3.2]

end_point = [-(grid_crop_bbox_min[0] + grid_crop_bbox_max[0]) / 2.0,
             -(grid_crop_bbox_min[1] + grid_crop_bbox_max[1]) / 2.0,
             -(grid_crop_bbox_min[2] + grid_crop_bbox_max[2]) / 2.0]
ep_ts = torch.tensor(end_point, device='cuda')
ep_ts = ep_ts.unsqueeze(0)

root_path = "/home/xingyu/gs_ws/SCube-release/waymo_webdataset/pose"
save_path = "/data/xingyu/waymo_semcity"

# to make more dense
vox_size = 1.171875

all_tars = os.listdir(root_path)

# 生成每个维度上的采样点
# x_range = np.arange(-25, 230, 0.25)
# y_range = np.arange(-128, 128, 0.25)
# z_range = np.arange(-64, 64, 0.25)

# # 生成 3D 网格点
# xx, yy, zz = np.meshgrid(x_range, y_range, z_range, indexing='ij')

# reshape 成 (N, 3)
# dense_grid = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)

def voxelize_semantics(points: torch.Tensor, semantics: torch.Tensor, voxel_size: float, 
                       special_labels=(17, 19), threshold=0.2):
    """
    Efficient voxelization using tensor-based unique indexing with semantic label aggregation.
    
    Args:
        points: [N, 3] float32
        semantics: [N] int64
        voxel_size: float
        special_labels: labels like 17/19 that override majority if over threshold
        threshold: float, e.g. 0.3
    
    Returns:
        voxel_centers: [M, 3] float32, voxel center positions
        voxel_semantics: [M] int64, majority/override label per voxel
    """
    assert points.shape[0] == semantics.shape[0]
    semantics = semantics.to(torch.long)

    # 1. Get voxel indices for each point
    voxel_indices = torch.floor(points / voxel_size).to(torch.int32)  # [N, 3]

    # 2. Unique voxel index & point-to-voxel mapping
    unique_voxels, inv = torch.unique(voxel_indices, return_inverse=True, dim=0)  # [M, 3], [N]
    M = unique_voxels.shape[0]
    max_label = int(semantics.max().item()) + 1

    # 3. Compute histogram: [M, num_labels]
    one_hot = torch.nn.functional.one_hot(semantics, num_classes=max_label).to(torch.int32)
    label_hist = torch.zeros((M, max_label), dtype=torch.int32, device=points.device)
    label_hist.index_add_(0, inv, one_hot)

    # 4. Decide voxel-level semantics with special label override
    total_count = label_hist.sum(dim=1).clamp(min=1)
    voxel_labels = torch.zeros(M, dtype=torch.int64, device=points.device)
    special_mask = torch.zeros(M, dtype=torch.bool, device=points.device)

    for s in special_labels:
        if s >= max_label: continue
        frac = label_hist[:, s].float() / total_count
        mask = frac > threshold
        voxel_labels[mask] = s
        special_mask |= mask

    not_special = ~special_mask
    voxel_labels[not_special] = label_hist[not_special].argmax(dim=1)

    # 5. Compute voxel centers
    voxel_centers = (unique_voxels.float() + 0.5) * voxel_size

    return unique_voxels, voxel_centers, voxel_labels

Nx = 128
Ny = 128
Nz = 16
Minx = -64
Miny = -64
Minz = -8

dense_grid = fvdb.gridbatch_from_dense(num_grids=1,
                                    dense_dims=[Nx, Ny, Nz],
                                    ijk_min=[Minx, Miny, Minz],
                                    voxel_sizes=torch.tensor([0.4, 0.4, 0.4], device='cuda'),
                                    origins=torch.tensor([0.2, 0.2, 0.2], device='cuda'),
                                    device="cuda")

def ray_tracing(pcds, semantics, ep_ts):
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
    mask_up_to_first_true = relative_idx <= min_to_prev_offset[groups] + 2
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
    ijk += torch.tensor([Nx//2, Ny//2, Nz//2], device=ijk.device, dtype=torch.int32)
    reshape_labels = torch.zeros((Nx, Ny, Nz), dtype=labels.dtype, device='cpu')
    reshape_invalid = torch.zeros((Nx, Ny, Nz), dtype=torch.bool, device='cpu')
    reshape_valid = torch.zeros((Nx, Ny, Nz), dtype=torch.bool, device='cpu')
    
    reshape_labels[ijk[:, 0], ijk[:, 1], ijk[:, 2]]     = labels.cpu()
    reshape_invalid[ijk[:, 0], ijk[:, 1], ijk[:, 2]]    = dense_invalid.cpu()
    reshape_valid[ijk[:, 0], ijk[:, 1], ijk[:, 2]]      = dense_valid.cpu()

    return reshape_labels, reshape_invalid, reshape_valid

if __name__ == "__main__":
    list_clips = []
    for tar in tqdm(all_tars, desc="Processing tar files"):
        # tar = '11623618970700582562_2840_367_2860_367.tar'
        tar_path = os.path.join(root_path, tar)
        clip = tar.split('.')[0]

        with tarfile.open(tar_path, "r") as tar_ext:
            members = [m for m in tar_ext.getmembers() if m.name.endswith("pose.front.npy")]
            
            list_poses = []
            for member in tqdm(members, desc=f"Extracting {tar}", leave=False):
                file = tar_ext.extractfile(member)
                if file:
                    cam2world = np.load(io.BytesIO(file.read())).astype(np.float32)
                    cam2world = torch.from_numpy(cam2world)
                    w_T_cam = torch.cat([cam2world[:,2:3], -cam2world[:,0:1], -cam2world[:,1:2], cam2world[:,3:4]], axis=1)  # OpenCV -> FLU
                    
                    # camera_pos = cam2world_FLU[:3, 3]
                    # camera_front = cam2world_FLU[:3, 0]
                    # camera_left = cam2world_FLU[:3, 1]
                    # camera_up = cam2world_FLU[:3, 2]                
                    list_poses.append(w_T_cam)
                    # new_grid_pos = camera_pos + \
                    #                 camera_front * (grid_crop_bbox_min[0] + grid_crop_bbox_max[0]) / 2 + \
                    #                 camera_left * (grid_crop_bbox_min[1] + grid_crop_bbox_max[1]) / 2 + \
                    #                 camera_up * (grid_crop_bbox_min[2] + grid_crop_bbox_max[2]) / 2

                    # grid2world = torch.clone(cam2world_FLU)
                    # grid2world[:3, 3] = new_grid_pos
                    # grid2world = grid2world @ b_T_l
                    # save_dir = os.path.join(save_path, clip, 'pose')
                    # os.makedirs(save_dir, exist_ok=True)

                    # save_filename = member.name.split('.')[-4] + '.' + member.name.split('.')[-1]
                    # save_filepath = os.path.join(save_dir, save_filename)

                    # # 保存 .npy
                    # np.save(save_filepath, grid2world)
        list_poses = list_poses[20:-20]
        w_T_cam0 = list_poses[0]
        w_T_camn = list_poses[-1]
        delta_pose =  torch.linalg.inv(w_T_cam0) @ w_T_camn
        delta_distance = torch.norm(delta_pose[:3, 3])
        if delta_distance < 50:
            continue
        list_clips.append(clip)
        grid_path = os.path.join('/data2/xingyu/pc_voxelsize_01', tar)
        with tarfile.open(grid_path, "r") as tar_ext:
            members = [m for m in tar_ext.getmembers() if m.name.endswith(".pth")]
            pth_member = members[0]

                # 提取二进制文件对象
            file_obj = tar_ext.extractfile(pth_member)

            # 用torch.load从二进制流读取
            data_dict = torch.load(file_obj, map_location="cpu")
            pc_pts = data_dict['points'].to('cuda')
            semantics = data_dict['semantics'].to('cuda')
            w_T_pc = data_dict['pc_to_world']
            
            # ts_dict = {}
            # ts_dict['points'] = pc_pts.to('cpu') * 0.5
            # ts_dict['semantics'] = semantics.to('cpu')
            # torch.save(ts_dict, '/home/xingyu/gs_ws/SemCity/exp/points/all.pcd.vs01.pth')

            
            mask_out_vehicles = ~torch.isin(semantics, torch.tensor([1, 2, 3, 4], device=semantics.device))
            pc_pts = pc_pts[mask_out_vehicles]
            semantics = semantics[mask_out_vehicles]
            
            cached_invalids = cached_valid = camilast_T_pc = None
            cami_T_pc_stack = {}
            saved_cnt = 0
            for idx in range(len(list_poses)):
                w_T_cami = list_poses[idx]
                pc_T_cami = torch.linalg.inv(w_T_pc) @ w_T_cami
                cami_T_pc = torch.linalg.inv(pc_T_cami).to('cuda')
                cami_T_pc_stack[idx] = cami_T_pc
                if idx == 0:
                    camilast_T_pc = cami_T_pc

                
                # vs01 cloud
                cami_pts = torch.einsum('ij,nj->ni', cami_T_pc, torch.cat([pc_pts, torch.ones_like(pc_pts[:,0:1]).to('cuda')], dim=1))[:,:3]
                
                crop_mask = (cami_pts[:,0] >    grid_crop_bbox_min[0]) & \
                            (cami_pts[:,0] <    grid_crop_bbox_max[0]) & \
                            (cami_pts[:,1] >    grid_crop_bbox_min[1]) & \
                            (cami_pts[:,1] <    grid_crop_bbox_max[1]) & \
                            (cami_pts[:,2] >    grid_crop_bbox_min[2]) & \
                            (cami_pts[:,2] <    grid_crop_bbox_max[2])

                det_pos = cami_T_pc @ torch.linalg.inv(camilast_T_pc)
                det_pos = det_pos[:3, 3].norm()
                
                if idx == 0:
                    camilast_T_pc = cami_T_pc
                
                overlap_threshold = int(np.random.rand() * 6) + 40
                if det_pos > overlap_threshold or idx == len(list_poses) - 1:
                    # cropout two scenes: one half last scene, one full current;
                    
                    # current full scene
                    cami_crop_pts = cami_pts[crop_mask]
                    cami_crop_sem = semantics[crop_mask]
                    # move to 
                    # last half scene
                    camlast_pts = torch.einsum('ij,nj->ni', camilast_T_pc, torch.cat([pc_pts, torch.ones_like(pc_pts[:,0:1]).to('cuda')], dim=1))[:,:3]
                    camlast_crop_mask = (camlast_pts[:,0] >    grid_crop_bbox_min[0]) & \
                                        (camlast_pts[:,0] <    grid_crop_bbox_max[0]) & \
                                        (camlast_pts[:,1] >    grid_crop_bbox_min[1]) & \
                                        (camlast_pts[:,1] <    grid_crop_bbox_max[1]) & \
                                        (camlast_pts[:,2] >    grid_crop_bbox_min[2]) & \
                                        (camlast_pts[:,2] <    grid_crop_bbox_max[2])
                    
                    # last half scene
                    camlast_crop_pts = camlast_pts[camlast_crop_mask]
                    camlast_crop_sem = semantics[camlast_crop_mask]
                    # transform to cami coordinate
                    cami_T_camlast = cami_T_pc @torch.linalg.inv(camilast_T_pc)
                    cami_last_pts = torch.einsum('ij,nj->ni', cami_T_camlast, torch.cat([camlast_crop_pts, torch.ones_like(camlast_crop_pts[:,0:1]).to('cuda')], dim=1))[:,:3]
                    cami_last_crop_mask =   (cami_last_pts[:,0] >    grid_crop_bbox_min[0]) & \
                                            (cami_last_pts[:,0] <    grid_crop_bbox_max[0]) & \
                                            (cami_last_pts[:,1] >    grid_crop_bbox_min[1]) & \
                                            (cami_last_pts[:,1] <    grid_crop_bbox_max[1]) & \
                                            (cami_last_pts[:,2] >    grid_crop_bbox_min[2]) & \
                                            (cami_last_pts[:,2] <    grid_crop_bbox_max[2])
                    # last half scene under cami coordinate
                    cami_last_crop_pts = cami_last_pts[cami_last_crop_mask]
                    cami_last_crop_sem = camlast_crop_sem[cami_last_crop_mask]                                          
                                 
                                            
                    # 0.2 size last half scene
                    generated_grids, generated_semantics = create_fvdb_grid_w_semantic_from_points(
                        [cami_last_crop_pts], 
                        [cami_last_crop_sem], 
                        {'voxel_sizes': torch.tensor([0.2, 0.2, 0.2], device=cami_pts.device),
                            'origins':  torch.tensor([0.1, 0.1, 0.1], device=cami_pts.device)},
                        {'voxel_sizes': torch.tensor([0.1000, 0.1000, 0.1000], device=cami_pts.device),
                            'origins': torch.tensor([0.0500, 0.0500, 0.0500], device=cami_pts.device)},
                    )
                    # add half of bbx size to the last half scene
                    # ready to save
                    cami_last_crop_pts = generated_grids.ijk.jdata + torch.tensor([128, 128, 16], device=generated_grids.ijk.device, dtype=torch.int32)
                    cami_last_crop_sem = generated_semantics[0]                    

                    # 0.2 size current full scene
                    generated_grids, generated_semantics = create_fvdb_grid_w_semantic_from_points(
                        [cami_crop_pts], 
                        [cami_crop_sem], 
                        {'voxel_sizes': torch.tensor([0.2, 0.2, 0.2], device=cami_pts.device),
                            'origins':  torch.tensor([0.1, 0.1, 0.1], device=cami_pts.device)},
                        {'voxel_sizes': torch.tensor([0.1000, 0.1000, 0.1000], device=cami_pts.device),
                            'origins': torch.tensor([0.0500, 0.0500, 0.0500], device=cami_pts.device)},
                    )
                    # add half of bbx size to the last half scene
                    # ready to save
                    cami_ori_crop_pts = generated_grids.ijk.jdata + torch.tensor([128, 128, 16], device=generated_grids.ijk.device, dtype=torch.int32)
                    cami_ori_crop_sem = generated_semantics[0]                    

                    # ts_dict = {}
                    # ts_dict['points'] = cami_last_crop_pts * 0.2
                    # ts_dict['semantics'] = generated_semantics
                    # torch.save(ts_dict, '/data/xingyu/SemCity/exp/points/all.pcd.vs01.pth')

                    # current full scene to be downsampled
                    mask_in_layout = torch.isin(cami_crop_sem, torch.tensor([14, 15, 17, 18, 19, 20, 21, 22], device=cami_crop_sem.device))
                    cami_crop_pts = cami_crop_pts[mask_in_layout]
                    cami_crop_sem = cami_crop_sem[mask_in_layout]
                    # gridi_crop_pts = cami_crop_pts + torch.tensor([25.6, 25.6, 3.2], device=cami_crop_pts.device, dtype=torch.int32)
                    # first downsample to 1.171875
                    cami_crop_coords, _, cami_crop_sem = voxelize_semantics(cami_crop_pts.to('cuda'), cami_crop_sem.to('cuda'), voxel_size=vox_size)
                    # ts_dict = {}
                    # ts_dict['points'] = cami_crop_coords * vox_size
                    # ts_dict['semantics'] = cami_crop_sem
                    # torch.save(ts_dict, '/data/xingyu/SemCity/exp/points/all.pcd.vs01.pth')
                    # now upsample to 0.2
                    coords = cami_crop_coords.long()  # [N,3]
                    coords += torch.tensor([22, 22, 3], device=coords.device, dtype=torch.int32)
                    valid_mask = (
                        (coords[:, 0] >= 0) & (coords[:, 0] <= 43) &
                        (coords[:, 1] >= 0) & (coords[:, 1] <= 43) &
                        (coords[:, 2] >= 0) & (coords[:, 2] <= 5)
                    )
                    coords = coords[valid_mask]
                    
                    sem    = cami_crop_sem.long()     # [N]

                    H = 44
                    W = 44
                    D = 6

                    
                    # shape = [1,1,D,H,W]
                    vol = torch.zeros((1,1,D,H,W), device=coords.device, dtype=torch.int64)
                    vol[0,0, coords[:,2], coords[:,0], coords[:,1]] = sem

                    new_D = 32
                    new_H = 256
                    new_W = 256

                    vol_up = torch.nn.functional.interpolate(vol.float(), size=(new_D,new_H,new_W), mode='nearest').long()
                    vol_up = vol_up.squeeze().squeeze().permute(1, 2, 0)

                    # points = torch.nonzero(vol_up, as_tuple=False)
                    # semantics = vol_up[points[:,0], points[:,1], points[:,2]].to(torch.int32)
                    # ts_dict = {
                    #     'points': cami_ori_crop_pts.to(torch.float32) * 0.2,
                    #     'semantics': cami_ori_crop_sem.to(torch.int32)
                    # }
                    # torch.save(ts_dict, '/data/xingyu/SemCity/exp/points/all.pcd.vs01.pth')
                    
                    # save camilast_crop_pts & vol_up
                    vox_label_prev = np.zeros((256, 256, 32), dtype=np.uint16)
                    cami_last_crop_sem = cami_last_crop_sem.cpu().numpy()
                    cami_last_crop_pts = cami_last_crop_pts.cpu().numpy()
                    vox_label_prev[cami_last_crop_pts[:,0], cami_last_crop_pts[:,1], cami_last_crop_pts[:,2]] = cami_last_crop_sem
                    
                    vox_label_curr = np.zeros((256, 256, 32), dtype=np.uint16)
                    cami_ori_crop_sem = cami_ori_crop_sem.cpu().numpy()
                    cami_ori_crop_pts = cami_ori_crop_pts.cpu().numpy()
                    vox_label_curr[cami_ori_crop_pts[:,0], cami_ori_crop_pts[:,1], cami_ori_crop_pts[:,2]] = cami_ori_crop_sem
                    
                    vox_label_curr_down = vol_up.cpu().numpy().astype(np.uint16)
                    
                    save_dir = os.path.join(save_path, clip, 'voxels_down')
                    os.makedirs(save_dir, exist_ok=True)
                    prev_path = os.path.join(save_dir, f'{saved_cnt:06d}.seed.label')
                    curr_path = os.path.join(save_dir, f'{saved_cnt:06d}.occ.label')
                    curr_down_path = os.path.join(save_dir, f'{saved_cnt:06d}.traj')
                    
                    # save voxels
                    
                    
                    vox_label_prev.astype(np.uint16).tofile(prev_path)
                    vox_label_curr.astype(np.uint16).tofile(curr_path)
                    vox_label_curr_down.astype(np.uint16).tofile(curr_down_path)
                    saved_cnt += 1
                    camilast_T_pc = cami_T_pc
    print(list_clips)
            
        