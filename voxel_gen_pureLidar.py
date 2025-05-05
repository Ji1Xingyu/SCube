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


import os
import open3d as o3d

import torch
import tensorflow.compat.v1 as tf

import numpy as np
from tqdm import tqdm
import cv2
import argparse
import faiss
import open3d.ml.torch as ml3d

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

grid_crop_bbox_min = [-25.6, -25.6, -3.2]
grid_crop_bbox_max = [25.6, 25.6, 3.2]

# grid_crop_bbox_min = [-5.12, -25.6, -3.2]
# grid_crop_bbox_max = [46.08, 25.6, 3.2]
end_point = [-(grid_crop_bbox_min[0] + grid_crop_bbox_max[0]) / 2.0,
             -(grid_crop_bbox_min[1] + grid_crop_bbox_max[1]) / 2.0,
             -(grid_crop_bbox_min[2] + grid_crop_bbox_max[2]) / 2.0]

Nx = 256
Ny = 256
Nz = 32
Minx = -128
Miny = -128
Minz = -16

vox_size = 0.2

target_grid_size = torch.tensor([0.2, 0.2, 0.2])
coarse_grid_size = torch.tensor([0.4, 0.4, 0.4])
finest_grid_size = torch.tensor([0.1, 0.1, 0.1])
crop_half_range_canonical = (torch.tensor(grid_crop_bbox_max) - torch.tensor(grid_crop_bbox_min)) / 2
crop_center = (torch.tensor(grid_crop_bbox_max) + torch.tensor(grid_crop_bbox_min)) / 2

center_finest = torch.tensor([0.0500, 0.0500, 0.0500], device='cuda')
center_target = torch.tensor([0.1000, 0.1000, 0.1000], device='cuda')
center_coarse = torch.tensor([0.2000, 0.2000, 0.2000], device='cuda')



# tf.enable_eager_execution()

from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.protos import segmentation_metrics_pb2
from waymo_open_dataset.protos import segmentation_submission_pb2



def save_kitti_format(
    base_path: str,
    frame_idx: int,
    points_xyz: np.ndarray,
    labels: np.ndarray
):
    """
    Save the given point cloud & labels in KITTI-like format:
      - velodyne/<frame_idx:06d>.bin  (x, y, z, r)
      - labels/<frame_idx:06d>.label  (uint32, one label per point)
      - poses.txt                     (if append_pose=True, we append or create lines)
      - calib.txt                     (if calib_str is provided, we create or skip if exists)
    Args:
        base_path   : root folder for your dataset
        frame_idx   : numeric index (e.g. 0, 1, 2, ...) for the filename
        points_xyz  : (N, 3) numpy array of point cloud coordinates
        labels      : (N,)  numpy array of semantic labels
        pose        : (4, 4) or (3, 4) transformation matrix for this frame's pose
        calib_str   : if provided, the text that goes into calib.txt (only created once)
        append_pose : if True, appends current pose to poses.txt
    """

    # 1. Ensure subfolders exist
    velodyne_dir = os.path.join(base_path, 'velodyne')
    label_dir    = os.path.join(base_path, 'labels')
    os.makedirs(velodyne_dir, exist_ok=True)
    os.makedirs(label_dir,    exist_ok=True)

    # 2. Save .bin file: typical KITTI format has (x, y, z, reflectance)
    #    If you don't have reflectance, you can store zeros.
    reflectance = np.zeros((points_xyz.shape[0], 1), dtype=np.float32)
    # shape: (N,4)
    points_to_save = np.hstack((points_xyz.astype(np.float32), reflectance))

    # Path to velodyne bin
    bin_path = os.path.join(velodyne_dir, f"{frame_idx:06d}.bin")
    points_to_save.tofile(bin_path)

    # 3. Save .label file: typical format is just an array of int32 or uint32 
    #    matching the # of points in the cloud
    label_path = os.path.join(label_dir, f"{frame_idx:06d}.label")
    labels.astype(np.uint32).tofile(label_path)

def save_pcd_manual(file_path, frameIdx, torch_dict):
    """
    以 ASCII 格式手动存储 PCD 文件，包含 x, y, z, label (pointxyzl)
    
    :param file_path: 目标 PCD 文件路径
    :param torch_dict: {'points': Tensor[N, 3], 'semantics': Tensor[N]}
    """

    grid_batch = torch_dict['points']


    points_jdata = grid_batch.grid_to_world(grid_batch.ijk.float())
    points = points_jdata.jdata.cpu().numpy()  # 转换为 NumPy 数组
    labels = torch_dict['semantics'].cpu().numpy().reshape(-1, 1)  # 确保是 (N, 1) 形状

    # save_kitti_format(file_path, frameIdx, points, labels)

    # return

    # 合并数据: (N, 4) -> [x, y, z, label]
    pointxyzl = np.hstack((points, labels))

    # 获取点的数量
    num_points = pointxyzl.shape[0]

    # 创建 PCD 文件头
    header = f"""# .PCD v0.7 - Point Cloud Data file format
VERSION 0.7
FIELDS x y z label
SIZE 4 4 4 4
TYPE F F F U
COUNT 1 1 1 1
WIDTH {num_points}
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS {num_points}
DATA ascii
"""

    # 将点云数据转换为字符串
    point_data = "\n".join(f"{p[0]} {p[1]} {p[2]} {int(p[3])}" for p in pointxyzl)
    
    # 写入 PCD 文件
    with open(file_path, 'w') as f:
        f.write(header)
        f.write(point_data)

    # print(f"PCD 文件已保存: {file_path}")

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

def xyz2count(ts : torch.tensor):
    x = ts[0][0].item()
    y = ts[0][1].item()
    z = ts[0][2].item()
    return (x - Minx)*Ny*Nz + (y - Miny)*Nz + (z - Minz)

def convert_range_image_to_point_cloud_labels(frame,
                                              range_images,
                                              segmentation_labels,
                                              ri_index=0):
    """Convert segmentation labels from range images to point clouds.

    Args:
    frame: open dataset frame
    range_images: A dict of {laser_name, [range_image_first_return,
        range_image_second_return]}.
    segmentation_labels: A dict of {laser_name, [range_image_first_return,
        range_image_second_return]}.
    ri_index: 0 for the first return, 1 for the second return.

    Returns:
    point_labels: {[N, 2]} list of 3d lidar points's segmentation labels. 0 for
        points that are not labeled.
    """
    calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
    point_labels = []
    for c in calibrations:
        range_image = range_images[c.name][ri_index]
        range_image_tensor = tf.reshape(
            tf.convert_to_tensor(range_image.data), range_image.shape.dims)
        range_image_mask = range_image_tensor[..., 0] > 0

        if c.name in segmentation_labels:
            sl = segmentation_labels[c.name][ri_index]
            sl_tensor = tf.reshape(tf.convert_to_tensor(sl.data), sl.shape.dims)
            sl_points_tensor = tf.gather_nd(sl_tensor, tf.where(range_image_mask))
        else:
            num_valid_point = tf.math.reduce_sum(tf.cast(range_image_mask, tf.int32))
            sl_points_tensor = tf.zeros([num_valid_point, 2], dtype=tf.int32)
            
        point_labels.append(sl_points_tensor.numpy())
    return point_labels

def get_pointNlabel(frame):
    (range_images, camera_projections, segmentation_labels,
     range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(frame)
    points, cp_points = frame_utils.convert_range_image_to_point_cloud(
                    frame, range_images, camera_projections, range_image_top_pose, keep_polar_features=True)
    point_labels = convert_range_image_to_point_cloud_labels(
                        frame, range_images, segmentation_labels)
    
    return points, point_labels

def extract_pc_img(seq_data, scen_dir):
    frame_num = 0

    # extract frame by frame     
    pose_list = []
    # l0_pts = []
    # sems_all = []
    l0_pts = None
    sems_all = None
    fnum_list = []
    pose_save = []
    frameCnt = 0
    frame_with_label = 0
    for idx, data in enumerate(seq_data):
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        if not frame.lasers[0].ri_return1.segmentation_label_compressed:
            continue
        # point cloud processing
        points, point_labels = get_pointNlabel(frame=frame)
        points_all = np.concatenate(points, axis=0)
        points_all = np.concatenate((points_all[:,3:6],points_all[:,1].reshape(-1, 1)), axis=1)
        points_all[:, 3] = np.tanh(points_all[:, 3])
        labels_all = np.concatenate(point_labels, axis=0)[:, 1]

        # point_labels_all = point_labels[0]      # preserve points from top LiDAR only
        # point_labels_all = point_labels_all[:, 1]

        # save bin
        points_all = points_all.astype(np.float32)
        # points_all.tofile(os.path.join(bin_dir,"frame{0:06d}.bin".format(frame_num)))
        
        pose = frame.pose.transform
        if frameCnt == 0:
            l0_T_g = np.linalg.inv(np.array(pose).reshape(4, 4))
        g_T_cur = np.array(pose).reshape(4, 4)  # Reshape into a 4x4 matrix
        l0_T_cur = l0_T_g @ g_T_cur

        # pose_list.append(pose_matrix)
        pose_save.append(l0_T_cur)

        tf_pts =    torch.tensor(points_all, device='cuda', dtype=torch.float32)
        tf_lb =     torch.tensor(labels_all.astype(np.int16), device='cuda', dtype=torch.int16)
        tf_pose =   torch.tensor(l0_T_cur, device='cuda', dtype=torch.float32)
        tf_pose = [tf_pose] * labels_all.shape[0]
        tf_pose = torch.stack(tf_pose, axis=0)
        if l0_pts is None:
            l0_pts = tf_pts
            sems_all = tf_lb
            pose_list = tf_pose
        else:
            l0_pts =      torch.concat([l0_pts, tf_pts], axis=0)
            sems_all =   torch.concat([sems_all, tf_lb], axis=0)
            pose_list =     torch.concat([pose_list, tf_pose], axis=0)
        # update frame_num
        frameCnt += 1
        fnum_list.append(labels_all.shape[0])
    
    pts2save = l0_pts
    l0_pts = l0_pts[:, :3]
    l0tog = torch.linalg.inv(pose_list[0])
    pose_list = l0tog @ pose_list
    ones_column = torch.ones((l0_pts.shape[0], 1)).cuda()
    l0_pts_homogeneous = torch.cat([l0_pts, ones_column], dim=1).reshape(-1, 4, 1)
    l0_pts = pose_list @ l0_pts_homogeneous
    l0_pts = l0_pts[:, :3, :]

    l0_pts = l0_pts.reshape(-1, 3)
    sems_all = sems_all.reshape(-1, 1)
    # Step 1: Separate labeled and unlabeled points
    unlabeled_mask = (sems_all == 0).squeeze()  # Get mask of unlabeled points
    labeled_mask = (sems_all != 0).squeeze()  # Get mask of labeled points

    unlabeled_points = l0_pts[unlabeled_mask]  # Extract unlabeled points
    labeled_points = l0_pts[labeled_mask]  # Extract labeled points
    labeled_labels = sems_all[labeled_mask]  # Extract their labels


    # Convert to NumPy (FAISS requires NumPy arrays)
    # unlabeled_points_np = unlabeled_points.cpu().numpy().astype('float32')
    # labeled_points_np = labeled_points.cpu().numpy().astype('float32')
    # labeled_labels_np = labeled_labels.cpu().numpy()

    k = 20
    nsearch = ml3d.layers.KNNSearch(return_distances=False)
    ans = nsearch(labeled_points.cpu(), unlabeled_points.cpu(), k)
    neighbors_index = ans.neighbors_index.reshape(-1, k)
    neighbors_label = labeled_labels[neighbors_index].reshape(-1, k)
    
    query_labels = torch.mode(neighbors_label, dim=1)[0]
    # incase loss lanemarks
    mask_mode_18 = (query_labels == 18)
    has_label_20 = (neighbors_label == 20).any(dim=1)
    update_mask = mask_mode_18 & has_label_20
    query_labels[update_mask] = 20
    sems_all[unlabeled_mask] = query_labels.reshape(-1, 1)
    
    # do lane expansion
    lane_mask = (sems_all == 20).squeeze() 
    unlane_mask = (sems_all != 20).squeeze()  # Get mask of unlabeled points

    unlane_points = l0_pts[unlane_mask]  # Extract unlabeled points
    unlane_labels = sems_all[unlane_mask]
    lane_points = l0_pts[lane_mask]  # Extract labeled points
    ans_lanes = nsearch(unlane_points.cpu(), lane_points.cpu(), 20)
    neighbors_index = ans_lanes.neighbors_index.reshape(-1, 1)
    sems_all[neighbors_index] = 20


    # filter out CAR, TRUCK, BUS AND OTHER VEHICLE
    non_car_mask =  (sems_all != 1) & \
                    (sems_all != 2) & \
                    (sems_all != 3) & \
                    (sems_all != 4)
    non_car_mask = non_car_mask.reshape(-1)
    l0_pts = l0_pts[non_car_mask]
    sems_all = sems_all[non_car_mask]

    l0_pts = l0_pts.to('cpu')
    sems_all = sems_all.to('cpu')
    # save for vis
    # all_pcd = o3d.t.geometry.PointCloud()
    # all_pcd.point["positions"] = o3d.core.Tensor(l0_pts.cpu().numpy())
    # all_pcd.point["label"] = o3d.core.Tensor(sems_all.cpu().numpy().astype(np.uint16))
    # o3d.t.io.write_point_cloud((os.getcwd() + "all.pcd"), all_pcd)

    list_lcur_cropped_pts = []
    list_lcur_cropped_sem = []
    for pos_idx, l0_T_cur in enumerate(pose_save):
        l0_T_cur_pos = l0_T_cur[:3, 3]
        l0_T_cur_front = l0_T_cur[:3, 0] # unit 
        l0_T_cur_left = l0_T_cur[:3, 1] # unit 
        l0_T_cur_up = l0_T_cur[:3, 2] # unit 

        new_grid_pos =  l0_T_cur_pos + \
                        l0_T_cur_front * (grid_crop_bbox_min[0] + grid_crop_bbox_max[0]) / 2 + \
                        l0_T_cur_left * (grid_crop_bbox_min[1] + grid_crop_bbox_max[1]) / 2 + \
                        l0_T_cur_up * (grid_crop_bbox_min[2] + grid_crop_bbox_max[2]) / 2

        l0_T_cur[:3, 3] = new_grid_pos
        l0_T_cur = torch.from_numpy(l0_T_cur).to(l0_pts.device).to(l0_pts.dtype)

        lcur_p = torch.einsum('ij,nj->ni', torch.linalg.inv(l0_T_cur), torch.cat([l0_pts, torch.ones_like(l0_pts[:,0:1])], dim=1))[:,:3]
        lcur_p = lcur_p.to('cpu')
        l0_T_cur = l0_T_cur.to('cpu')
        l0_centre = l0_T_cur[:3, 3] + crop_center

        crop_mask = (lcur_p[:,0] > -crop_half_range_canonical[0]) & \
                    (lcur_p[:,0] <  crop_half_range_canonical[0]) & \
                    (lcur_p[:,1] > -crop_half_range_canonical[1]) & \
                    (lcur_p[:,1] <  crop_half_range_canonical[1]) & \
                    (lcur_p[:,2] > -crop_half_range_canonical[2]) & \
                    (lcur_p[:,2] <  crop_half_range_canonical[2])

        list_lcur_cropped_pts.append(lcur_p[crop_mask])
        list_lcur_cropped_sem.append(sems_all[crop_mask])

        # all_pcd = o3d.t.geometry.PointCloud()
        # all_pcd.point["positions"] = o3d.core.Tensor(lcur_cropped_pts.cpu().numpy())
        # all_pcd.point["label"] = o3d.core.Tensor(lcur_cropped_sem.cpu().numpy().astype(np.uint16))
        # o3d.t.io.write_point_cloud((os.getcwd() + "/cropped.pcd"), all_pcd)
        # l0_pts under l0 frame
        
    return list_lcur_cropped_pts, list_lcur_cropped_sem, pose_save, l0_pts, sems_all


def ray_tracing(list_lcur_cropped_pts, list_lcur_cropped_sem, pose_save, path):
    
    device = torch.device("cuda")
    frameIdx = 0
    bbx = -np.array(grid_crop_bbox_min) + np.array(grid_crop_bbox_max)
    bbx_vox_size = bbx / 0.2
    ep_ts = torch.tensor(end_point, device='cuda')
    ep_ts = ep_ts.unsqueeze(0)

    for idx_seq in range(1):
            if idx_seq: return
            pcds, semantics = create_fvdb_grid_w_semantic_from_points(
                [list_lcur_cropped_pts.to('cuda')], 
                [list_lcur_cropped_sem.to('cuda')], 
                {'voxel_sizes': target_grid_size,
                    'origins': center_target},
                {'voxel_sizes': finest_grid_size,
                    'origins': center_finest},
                extra_meshes=None, 
            )
            semantics = semantics[0].squeeze()
            pose = torch.from_numpy(pose_save).to('cuda')
            # generate a dense voxel for ray tracing
            dense_grid = fvdb.gridbatch_from_dense(num_grids=1,
                                                dense_dims=[Nx, Ny, Nz],
                                                ijk_min=[Minx, Miny, Minz],
                                                voxel_sizes=target_grid_size,
                                                origins=center_target,
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
            del vox_coord_jt
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

            labels[labels<0] = 0

            ijk = dense_grid.ijk.jdata
            ijk += torch.tensor([128, 128, 16], device=ijk.device, dtype=torch.int32)

            reshape_labels = torch.zeros((256, 256, 32), dtype=labels.dtype, device='cpu')
            reshape_invalid = torch.zeros((256, 256, 32), dtype=torch.bool, device='cpu')
            reshape_valid = torch.zeros((256, 256, 32), dtype=torch.bool, device='cpu')

            reshape_labels[ijk[:, 0], ijk[:, 1], ijk[:, 2]]     = labels.cpu()
            reshape_invalid[ijk[:, 0], ijk[:, 1], ijk[:, 2]]    = dense_invalid.cpu()
            reshape_valid[ijk[:, 0], ijk[:, 1], ijk[:, 2]]      = dense_valid.cpu()
            
            out_mask = (reshape_labels != 0)
            pts = torch.nonzero(out_mask, as_tuple=False) * 0.2  # shape: [N, 4] (b, y, x, z)
            sem = reshape_labels[out_mask]

            # all_pcd = o3d.t.geometry.PointCloud()
            # all_pcd.point["positions"] = o3d.core.Tensor(pts.cpu().numpy())
            # all_pcd.point["label"] = o3d.core.Tensor(sem.cpu().numpy().astype(np.uint16))
            # o3d.t.io.write_point_cloud((os.getcwd() + "all.pcd"), all_pcd)

            asdasdasd = 1


            save_semantic_kitti(path, pose,
                                valid=reshape_valid.numpy(), 
                                invalid=reshape_invalid.numpy(), 
                                labels=reshape_labels.numpy())


def data_extractor(split, output_path):
    seq_list = sorted(os.listdir(split))
    for seq_file in tqdm(seq_list):
        if '.tfrecord' not in seq_file:
            continue
        # extract raw data from seq_file
        core = seq_file.replace('segment-', '').replace('_with_camera_labels.tfrecord', '')
        scen_dir = os.path.join(output_path, core)
        if not os.path.exists(scen_dir):
            os.makedirs(scen_dir)
        seq_data = tf.data.TFRecordDataset(os.path.join(split, seq_file), compression_type='')
        pts, sems, poses, pts_all, sems_all = extract_pc_img(seq_data, scen_dir)
        torch.cuda.empty_cache()

        frameIdx = 0

        for pt, sem, pos in zip(pts, sems, poses):
            file_name = f"{frameIdx:06d}.pcd"
            path = os.path.join(scen_dir, file_name)
            ray_tracing(pt, sem, pos, path)
            frameIdx += 1

        pcds, semantics = create_fvdb_grid_w_semantic_from_points(
            [pts_all.to('cuda')], 
            [sems_all.to('cuda')], 
            {'voxel_sizes': target_grid_size,
                'origins': center_target},
            {'voxel_sizes': finest_grid_size,
                'origins': center_finest},
            extra_meshes=None, 
        )
        
        pcds = pcds.ijk.jdata * 0.2

        ts_dict = {'points': pcds, 'semantic': semantics[0]}
        all_points = os.path.join(scen_dir, 'all.pcd.pth')
        torch.save(ts_dict, all_points)

        # all_pcd = o3d.t.geometry.PointCloud()
        # all_pcd.point["positions"] = o3d.core.Tensor(pcds.cpu().numpy())
        # all_pcd.point["label"] = o3d.core.Tensor(semantics[0].cpu().numpy().astype(np.uint16))
        # o3d.t.io.write_point_cloud((os.getcwd() + "all.pcd"), all_pcd)


    print("Complete data extraction!")
    return True




def parse_args():
    parser = argparse.ArgumentParser(description='Waymo Data Extraction')
    parser.add_argument('--task', default="MoPA", type=str, help='task name')
    parser.add_argument('--raw_waymo_dir',
                        default="/mnt/data_nas/data/xingyu/waymo/validation",
                        type=str, help='path to directory with raw waymo training/val')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    
    waymo_root_dir = args.raw_waymo_dir
    outpath = os.getcwd() + '/dataset/semantic_kitti'
    # split = os.path.join(waymo_root_dir, "training")
    # output_path = os.path.join(waymo_root_dir, "waymo_extracted_test/training")
    data_extractor(waymo_root_dir, outpath)

    # split = split = os.path.join(waymo_root_dir, "validation")
    # output_path = os.path.join(waymo_root_dir, "waymo_extracted_test/validation")
    # data_extractor(split=split, output_path=output_path)
