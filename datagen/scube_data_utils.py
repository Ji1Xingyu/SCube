# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import numpy as np
import torch
import io
import gc
import time

from tqdm import tqdm
from pyquaternion import Quaternion

############ Segmentation Related ############
@torch.inference_mode()
def inference_mmseg(video_numpy_list, inferenecer):
    """
    Inference the video numpy list using mmseg

    Args:
        video_numpy_list: list, 
            list of numpy array with shape [H, W, 3], 
            each numpy array is a frame of the video

        inferencer: MMSegInferencer

    Returns:
        list: list of semantic segmentation results for each frame
    """
    pred_list = []

    # video_numpy_list can be too long for one inference, so we split it into several parts
    chunking_num = 15
    chunk_index = torch.linspace(0, len(video_numpy_list), chunking_num + 1).long()
    print("Inference mmseg...")

    for idx in range(chunking_num):
        # print(f"Processing segformer chunk {idx}/{chunking_num}")
        if chunk_index[idx] == chunk_index[idx+1]:
            continue
        video_numpy_chunk = video_numpy_list[chunk_index[idx]:chunk_index[idx+1]]
        semantic_map = inferenecer(video_numpy_chunk)['predictions']
        if isinstance(semantic_map, np.ndarray):
            semantic_map = [semantic_map]

        pred_list.extend(semantic_map)

    return pred_list

############ Encoding Related ############
def encode_dict_to_npz_bytes(data_dict):
    buffer = io.BytesIO()
    np.savez(buffer, **data_dict)
    buffer.seek(0)
    
    return buffer.getvalue()

def imageencoder_imageio_png16(image):
    """Compress an image using PIL and return it as a string.

    Can handle 16-bit images.

    :param image: ndarray representing an image

    """
    import imageio.v2 as imageio

    with io.BytesIO() as result:
        imageio.imwrite(result, image, format='png')
        return result.getvalue()


############ Object Related ############
def object_info_to_canonical_cuboid(object_info):
    """
    Do not transform the cuboid to the world coordinate, just return the cuboid in the object coordinate
    """
    size = np.array(object_info["size"])
    corners_obj = np.array([
        [1, 1, 1],
        [1, 0, 1],
        [0, 0, 1],
        [0, 1, 1],
        [1, 1, 0],
        [1, 0, 0],
        [0, 0, 0],
        [0, 1, 0],
    ])
    corners_obj = corners_obj * size
    corners_obj = corners_obj - size / 2

    return corners_obj

def object_info_to_cuboid(object_info):
    """

    bbox format:
        h
        ^  w 
        | /
        |/
        o -------> l (heading)

   3 ---------------- 0
  /|                 /|
 / |                / |
2 ---------------- 1  |
|  |               |  |
|  7 ------------- |- 4
| /                | /
6 ---------------- 5 


    Args:
    - object_info: dict, object information, with the following keys:
        - translation: list, x, y, z
        - size: list, l, w, h
        - rotation: list, w, x, y, z

    Returns:
    - corners_world: np.ndarray, shape=(8, 3), the 8 corners of the object in the world coordinate
    """
    try:
        size = np.array(object_info["size"])
    except:
        size = np.array(object_info['object_lwh'])

    corners_obj = np.array([
        [1, 1, 1],
        [1, 0, 1],
        [0, 0, 1],
        [0, 1, 1],
        [1, 1, 0],
        [1, 0, 0],
        [0, 0, 0],
        [0, 1, 0],
    ])
    corners_obj = corners_obj * size
    corners_obj = corners_obj - size / 2
    # pad 1 for homogeneous coordinates
    corners_obj = np.concatenate([corners_obj, np.ones((8, 1))], axis=1)

    # quaternion to rotation matrix
    object_to_world = object_info_to_object2world(object_info)

    corners_world = np.einsum("ij,kj->ki", object_to_world, corners_obj)[:, :3]

    return corners_world

def object_info_to_object2world(object_info):
    try:
        object_to_world = object_info["object_to_world"]
    except:
        # quaternion to rotation matrix
        T = Quaternion(object_info["rotation"]).transformation_matrix
        T[:3, 3] = object_info["translation"]
        object_to_world = T

    return object_to_world


def get_points_in_cuboid(lidar_points, lidar_to_world, object_info):
    """
    Args:
    - lidar_points: np.ndarray, shape=(N, 3), lidar points in the ego car coordinate
    - lidar_to_world: np.ndarray, shape=(4, 4), the transformation matrix from lidar coordinate to world
    - object_info: dict, object information, with the following keys:
        - translation: list, x, y, z
        - size: list, l, w, h
        - rotation: list, w, x, y, z

        we use it to construct world-to-object transformation matrix

    Returns:
    - points_in_cuboid: np.ndarray, shape=(M, 3), the points in the cuboid
    """
    box_l, box_w, box_h = object_info["size"]
    
    object_to_world = object_info_to_object2world(object_info)
    world_to_object = np.linalg.inv(object_to_world)

    # transform lidar points to world coordinate then to object coordinate
    lidar_to_object = world_to_object @ lidar_to_world

    # transform lidar points to object coordinate
    lidar_points_padded = np.concatenate([lidar_points, np.ones((lidar_points.shape[0], 1))], axis=1)
    points_in_object = np.einsum("ij,kj->ki", lidar_to_object, lidar_points_padded)[:, :3] # shape=(N, 3)

    # keep points in the cuboid
    points_in_cuboid = points_in_object[
        (points_in_object[:, 0] >= -box_l / 2) & (points_in_object[:, 0] <= box_l / 2) &
        (points_in_object[:, 1] >= -box_w / 2) & (points_in_object[:, 1] <= box_w / 2) &
        (points_in_object[:, 2] >= -box_h / 2) & (points_in_object[:, 2] <= box_h / 2)
    ]

    return points_in_cuboid


def load_model_with_retries(model_name, retries=10, delay=5):
    for attempt in range(retries):
        try:
            # model = torch.hub.load('yvanyin/metric3d', model_name, pretrain=True).cuda().eval()
            model = torch.hub.load('../cached_model/yvanyin_metric3d_main', model_name, source = 'local', pretrain=True).cuda().eval()
            return model
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                raise

############ Depth Related ############
@torch.inference_mode()
def inference_isolated(input_chunk, metric3d_model='metric3d_vit_small'):
    torch.cuda.empty_cache()
    model = load_model_with_retries(metric3d_model) # can not continue with different batchsize
    pred_depth, _, _ = model.inference({'input': input_chunk})

    del model
    torch.cuda.empty_cache()

    return pred_depth


@torch.inference_mode()
def inference_metric3dv2(video_tensor, max_depth=199.9, metric3d_model='metric3d_vit_small', chunking_num = 100):
    """
    Args:
        video_tensor: torch.Tensor, shape (T, C, H, W), normalized to [0, 1]
        max_depth: float, the maximum depth value, used to filter out invalid depth values to be 0. Note that Metric3D itself set it to 200!
        metric3d_model: str, the name of the metric3d model, e.g. metric3d_vit_small, metric3d_vit_large, metric3d_vit_giant2

    Returns:
        depth: torch.Tensor, shape (T, 1, H, W)
        normal: torch.Tensor, shape (T, 3, H, W)
    """
    chunk_index = torch.linspace(0, video_tensor.shape[0], chunking_num + 1).long()

    pred_depth_chunks = []
    pred_normal = None
    torch.cuda.empty_cache()

    print(f"**********************************Inference metric3d with maximum batch size {(chunk_index[1:] - chunk_index[:-1]).max()}")
    # model = torch.hub.load('yvanyin/metric3d', metric3d_model, pretrain=True).cuda().eval() # can not continue with different batchsize

    for idx in tqdm(range(chunking_num)):
        # print(f"Processing metric3d chunk {idx}/{chunking_num}")
        if chunk_index[idx] == chunk_index[idx+1]:
            continue
        # with torch.no_grad():
            
        input_chunk = video_tensor[chunk_index[idx]:chunk_index[idx+1]].cuda()
        pred_depth = inference_isolated(input_chunk, metric3d_model=metric3d_model)
        pred_depth[pred_depth > max_depth] = 0
        pred_depth_chunks.append(pred_depth.cpu()) # shape (T, 1, H, W)
        # pred_depth_chunks.append(pred_depth) # shape (T, 1, H, W)
        del pred_depth
        del input_chunk
        # del confidence
        # del output_dict
        torch.cuda.synchronize()  # Ensure all CUDA operations are complete
        torch.cuda.empty_cache()
        gc.collect()

    pred_depth = torch.cat(pred_depth_chunks, dim=0)
    print("********************************** Finished Inference start interpolate now")
    pred_depth = torch.nn.functional.interpolate(pred_depth, size=video_tensor.shape[2:], mode='bilinear', align_corners=False)
    pass

    return pred_depth, pred_normal

def align_depth_to_depth(
    source_depth: torch.Tensor,
    target_depth: torch.Tensor,
    target_mask: torch.Tensor | None = None,
    return_scale: bool = False
) -> torch.Tensor:
    """
    Apply affine transformation to align source depth to target depth.

    Args:
        source_inv_depth: Depth map to be aligned. Shape: (H, W).
        target_depth: Target depth map. Shape: (H, W).
        target_mask: Mask of valid target pixels. Shape: (H, W).

    Returns:
        Aligned Depth map. Shape: (H, W).
    """
    # source_depth = source_depth.cuda()
    source_invalid = source_depth == 0
    source_mask = source_depth > 0
    target_depth_mask = target_depth > 0

    if target_mask is None:
        target_mask = target_depth_mask
    else:
        target_mask = torch.logical_and(target_mask > 0, target_depth_mask)

    # Remove outliers
    outlier_quantiles = torch.tensor([0.1, 0.9], device=source_depth.device)

    try:
        print("*******************************Aligning depth")
        source_data_low, source_data_high = torch.quantile(
            source_depth[source_mask], outlier_quantiles
        )
        target_data_low, target_data_high = torch.quantile(
            target_depth[target_mask], outlier_quantiles
        )
        source_mask = (source_depth > source_data_low) & (source_depth < source_data_high)
        target_mask = (target_depth > target_data_low) & (target_depth < target_data_high)

        mask = torch.logical_and(source_mask, target_mask)

        source_data = source_depth[mask].view(-1, 1)
        target_data = target_depth[mask].view(-1, 1)

        # TODO: Maybe use RANSAC or M-estimators to make it more robust
        ones = torch.ones((source_data.shape[0], 1), device=source_data.device)
        source_data_h = torch.cat([source_data, ones], dim=1)
        transform_matrix = torch.linalg.lstsq(source_data_h, target_data).solution

        scale, bias = transform_matrix[0, 0], transform_matrix[1, 0]
        aligned_depth = source_depth * scale + bias

        # invalid still invalid
        aligned_depth[source_invalid] = 0

        print(f"Scale: {scale}, Bias: {bias}")

    except Exception as e:
        if return_scale:
            return 1, 0
        else:
            return source_depth

    if return_scale:
        return scale, bias
    else:
        return aligned_depth


def align_depth_to_depth_batch(
    source_depth: torch.Tensor,
    target_depth: torch.Tensor,
    target_mask: torch.Tensor | None = None,
    return_scale: bool = False
) -> torch.Tensor:
    """
    Apply affine transformation to align source depth to target depth.

    Args:
        source_inv_depth: Depth map to be aligned. Shape: (B, H, W).
        target_depth: Target depth map. Shape: (B, H, W).
        target_mask: Mask of valid target pixels. Shape: (B, H, W).

    Returns:
        Aligned Depth map. Shape: (B, H, W).
    """
    assert return_scale == False, "return_scale is not supported for batch version"

    B = source_depth.shape[0]
    aligned_depth = []
    for i in range(B):
        aligned_depth.append(align_depth_to_depth(source_depth[i], target_depth[i], target_mask=target_mask[i]))

    return torch.stack(aligned_depth)


def project_points_to_depth_image(points, points_coordinate_to_camera, camera_intrinsic, width, height):
    """
    Fast implementation to project the 3D points to image plane to get the depth.

    Args:
        points: torch.Tensor, shape (N, 3), the 3D points
        points_coordinate_to_camera: torch.Tensor, shape (4, 4), the transformation matrix from the points coordinate to the camera coordinate
        camera_intrinsic: torch.Tensor, shape (3, 3), the camera intrinsic matrix

    Returns:
        depth_image: torch.Tensor, shape (H, W)
    """
    points = torch.cat([points, torch.ones(points.shape[0], 1).to(points)], dim=1) # shape (N, 4)
    points_cam = points @ points_coordinate_to_camera.T # shape (N, 4)
    points_cam = points_cam[:, :3] # shape (N, 3)

    valid_depth_mask = points_cam[:, 2] > 0
    points_cam = points_cam[valid_depth_mask]

    points_depth = points_cam[:, 2]
    points_cam = points_cam[:, :3] / points_cam[:, 2:3] # shape (N, 3)
    points_uv = points_cam @ camera_intrinsic.T # shape (N, 3)

    u_round = torch.round(points_uv[:, 0]).long()
    v_round = torch.round(points_uv[:, 1]).long()

    valid_uv_mask = (u_round >= 0) & (u_round < width) & (v_round >= 0) & (v_round < height)

    u_valid = u_round[valid_uv_mask]
    v_valid = v_round[valid_uv_mask]
    z_valid = points_depth[valid_uv_mask]

    indices = v_valid * width + u_valid

    depth_image = torch.full((height, width), float('inf')).to(points_depth).flatten()
    depth_image = depth_image.scatter_reduce_(0, indices, z_valid, "amin")
    depth_image = depth_image.view(height, width)
    depth_mask = torch.isfinite(depth_image)

    # change inf to 0
    depth_image[~depth_mask] = 0

    return depth_image, depth_mask