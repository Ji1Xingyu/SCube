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

grid_crop_bbox_min = [-25.6, -25.6, -6.4]
grid_crop_bbox_max = [25.6, 25.6, 6.4]

end_point = [-(grid_crop_bbox_min[0] + grid_crop_bbox_max[0]) / 2.0,
             -(grid_crop_bbox_min[1] + grid_crop_bbox_max[1]) / 2.0,
             -(grid_crop_bbox_min[2] + grid_crop_bbox_max[2]) / 2.0]
ep_ts = torch.tensor(end_point, device='cuda')
ep_ts = ep_ts.unsqueeze(0)

root_path = "/home/xingyu/gs_ws/SCube-release/waymo_webdataset/pose"
save_path = "/data2/xingyu/waymo_semcity"

# to make more dense
vox_size = 1.171875 / 4

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
Nz = 32
Minx = -64
Miny = -64
Minz = -16

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
        grid_path = os.path.join('/home/xingyu/gs_ws/SCube-release/waymo_webdataset/pc_voxelsize_01', tar)
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
            for idx in range(len(list_poses)):
                w_T_cami = list_poses[idx]
                pc_T_cami = torch.linalg.inv(w_T_pc) @ w_T_cami
                cami_T_pc = torch.linalg.inv(pc_T_cami).to('cuda')
                cami_T_pc_stack[idx] = cami_T_pc
                if idx == 0:
                    camilast_T_pc = cami_T_pc

                    
                cami_pts = torch.einsum('ij,nj->ni', cami_T_pc, torch.cat([pc_pts, torch.ones_like(pc_pts[:,0:1]).to('cuda')], dim=1))[:,:3]
                
                crop_mask = (cami_pts[:,0] >    grid_crop_bbox_min[0]) & \
                            (cami_pts[:,0] <    grid_crop_bbox_max[0]) & \
                            (cami_pts[:,1] >    grid_crop_bbox_min[1]) & \
                            (cami_pts[:,1] <    grid_crop_bbox_max[1]) & \
                            (cami_pts[:,2] >    grid_crop_bbox_min[2]) & \
                            (cami_pts[:,2] <    grid_crop_bbox_max[2])

                det_pos = cami_T_pc @ torch.linalg.inv(camilast_T_pc)
                det_pos = det_pos[:3, 3].norm()
                
                if det_pos > 10 or idx == len(list_poses) - 1 or idx == 0:
                    camilast_T_pc = cami_T_pc

                    ts_dict = {}
                    ts_dict['points'] = cami_pts[crop_mask]
                    ts_dict['semantics'] = semantics[crop_mask]
                    # torch.save(ts_dict, '/home/xingyu/gs_ws/SemCity/exp/points/all.pcd.vs01.pth')
                    
                    generated_grids, generated_semantics = create_fvdb_grid_w_semantic_from_points(
                        [ts_dict['points']], 
                        [ts_dict['semantics']], 
                        {'voxel_sizes': torch.tensor([0.4, 0.4, 0.4], device=cami_pts.device),
                            'origins':  torch.tensor([0.2, 0.2, 0.2], device=cami_pts.device)},
                        {'voxel_sizes': torch.tensor([0.1000, 0.1000, 0.1000], device=cami_pts.device),
                            'origins': torch.tensor([0.0500, 0.0500, 0.0500], device=cami_pts.device)},
                    )
                    generated_semantics = generated_semantics[0]
                    
                    # ts_dict['points'] = generated_grids.ijk.jdata.float() * 0.5
                    # ts_dict['semantics'] = generated_semantics[0]
                    # torch.save(ts_dict, '/home/xingyu/gs_ws/SemCity/exp/points/all.pcd.vs01.pth')
                    
                    reshape_labels, reshape_invalid, reshape_valid = ray_tracing(generated_grids, generated_semantics, ep_ts)
                    reshape_labels[reshape_labels == -1] = 0
                    # reshape_invalid[reshape_labels > 0] = False
                    
                    cami_invalid_pts = torch.nonzero(reshape_invalid) - torch.tensor([Nx//2, Ny//2, Nz//2], device=reshape_labels.device, dtype=torch.int32)

                    cami_invalid_pts = cami_invalid_pts.float() * 0.4
                    cami_valid_pts = torch.nonzero(reshape_valid) - torch.tensor([Nx//2, Ny//2, Nz//2], device=reshape_labels.device, dtype=torch.int32)
                    cami_valid_pts = cami_valid_pts.float() * 0.4
                    
                    pc_invalid_pts = torch.einsum('ij,nj->ni', torch.linalg.inv(cami_T_pc).to(reshape_labels.device), torch.cat([cami_invalid_pts, torch.ones_like(cami_invalid_pts[:,0:1]).to(reshape_labels.device)], dim=1))[:,:3]
                    pc_valid_pts = torch.einsum('ij,nj->ni', torch.linalg.inv(cami_T_pc).to(reshape_labels.device), torch.cat([cami_valid_pts, torch.ones_like(cami_valid_pts[:,0:1]).to(reshape_labels.device)], dim=1))[:,:3]
                    
                    if cached_invalids is None:
                        cached_invalids = pc_invalid_pts
                    else:
                        cached_invalids = torch.cat((cached_invalids, pc_invalid_pts), dim=0)
                    
                    if cached_valid is None:
                        cached_valid = pc_valid_pts
                    else:
                        cached_valid = torch.cat((cached_valid, pc_valid_pts), dim=0)
                    
                    # do ray tracing for all the points within the crop mask


                if idx == 0:
                    all_mask = crop_mask
                else:
                    all_mask = all_mask | crop_mask
                
                if idx == 0:
                    positions = pc_T_cami[:3, 3].reshape(1, 3)
                else:
                    positions = torch.cat((positions, pc_T_cami[:3, 3].reshape(1, 3)), dim=0)
            # ts_dict = {}
            # ts_dict['points'] = cached_valid
            # ts_dict['semantics'] = torch.full((cached_valid.shape[0],), 1, dtype=torch.int64, device=cached_valid.device)
            # torch.save(ts_dict, '/home/xingyu/gs_ws/SemCity/exp/points/all.pcd.vs01.pth')
            
            all_mask = all_mask.to('cpu')
            
            pc_pts = pc_pts[all_mask,:]
            semantics = semantics[all_mask]
            
            # two grids, one for occ, one for traj:
            traj_sem = torch.arange(10000, 10000 + positions.shape[0], device='cuda')
            occ_coords, occ_points, occ_semantics = voxelize_semantics(pc_pts.to('cuda'), semantics.to('cuda'), voxel_size=vox_size)
            traj_coords, traj_points, traj_semantics = voxelize_semantics(positions.to('cuda'), traj_sem, voxel_size=vox_size)
            
            invalid_coords, invalid_points, invalid_semantics = voxelize_semantics(cached_invalids.to('cuda'), torch.full((cached_invalids.shape[0],), 1).to('cuda'), voxel_size=vox_size)
            valid_coords, valid_points, valid_semantics = voxelize_semantics(cached_valid.to('cuda'), torch.full((cached_valid.shape[0],), 1).to('cuda'), voxel_size=vox_size)
            
            # filter invalid coords that are in valid coords
            # invalid_coords = filter_invalid_coords_by_valid(invalid_coords, valid_coords, radius=1)
            # invalid_semantics = torch.full((invalid_coords.shape[0],), 1, dtype=torch.int64, device=invalid_coords.device)

            mean_occ_coords = occ_coords.to(torch.float32).mean(dim=0)
            occ_coords = occ_coords - mean_occ_coords
            traj_coords = traj_coords - mean_occ_coords
            invalid_coords = invalid_coords - mean_occ_coords
            valid_coords = valid_coords - mean_occ_coords
            
            ts_dict = {}
            ts_dict['points'] = torch.cat((occ_coords*0.25, valid_coords*0.25), dim=0)
            ts_dict['semantics'] = torch.cat((occ_semantics, valid_semantics), dim=0)
            torch.save(ts_dict, '/home/xingyu/gs_ws/SemCity/exp/points/all.pcd.vs01.pth')
            
            # bbx center:
            points_all = torch.cat((occ_coords*0.25, traj_coords*0.25), dim=0)
            valid_coords = valid_coords * 0.25
            mean_occ_coords = mean_occ_coords * 0.25
            for k, v in cami_T_pc_stack.items():
                v[:3, 3] = v[:3, 3] / 1.171875
                
            voxelized_pos_idc = traj_semantics[traj_semantics >= 10000] - 10000
            cami_T_pc_stack_th = torch.stack([cami_T_pc_stack[voxelized_pos_idc[i].item()] for i in range(voxelized_pos_idc.shape[0])], dim=0)
            traj_semantics = torch.arange(10000, 10000 + voxelized_pos_idc.shape[0], device='cuda')
            
            semantics_all = torch.cat((occ_semantics, traj_semantics), dim=0)

            # ts_dict = {}
            # ts_dict['points'] = points_seed.to('cpu') * 0.5
            # ts_dict['semantics'] = semantics_seed.to('cpu')
            # torch.save(ts_dict, './exp/points/all.pcd.vs01.pth')

            ts_dict = {}
            ts_dict['points'] = points_all
            ts_dict['semantics'] = semantics_all
            ts_dict['cami_T_pc_stack'] = cami_T_pc_stack_th
            ts_dict['valid_coords'] = valid_coords
            ts_dict['centre'] = mean_occ_coords
            ts_path = os.path.join(save_path, clip, 'vox_1.171875.pth')
            os.makedirs(os.path.dirname(ts_path), exist_ok=True)
            torch.save(ts_dict, ts_path)
            asdasdasd = 1

    print(list_clips)
            
        