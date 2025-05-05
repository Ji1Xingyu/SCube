import trimesh
import numpy as np
import torch
from pyquaternion import Quaternion

def object_info_to_cuboid(object_lwh, object_to_another_coord):
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
    - object_lwh: list, l, w, h
    - object_to_another_coord: np.ndarray, shape=(4, 4), transformation matrix from object to another coordinate system (e.g. world)

    Returns:
    - corners: np.ndarray, shape=(8, 3), the 8 corners of the object in the world coordinate
    """

    size = np.array(object_lwh)
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
    corners = np.einsum("ij,kj->ki", object_to_another_coord, corners_obj)[:, :3]

    return corners


def build_scene_bounding_boxes_from_all_object_info(all_object_dict, world_transform=np.eye(4), aabb_half_range=None):
    """
    Create a scene consisting all car object, rescale them, transform them and merge them

    Args:
        all_object_dict: dict, dict containing all object information, webdataset's all_object_info
        world_transform: np.array, [4, 4], world transform matrix to transform bounding box from world to another coordinate system
        aabb_half_range: np.array, [3,], half range of the axis aligned bounding box to filter out objects

    Returns:
        all_cuboids: np.ndarray, shape=(N, 8, 3), the 8 corners of the object in the world coordinate
    """
    all_cuboids = []

    for gid, object_info in all_object_dict.items(): 
        object_to_world = np.array(object_info['object_to_world'])
        target_lwh = np.array(object_info['object_lwh'])
        is_car = object_info['object_type'] == 'car'
        is_moving = object_info['object_is_moving'] # we should treat object is moving equally

        if is_car:
            object_to_target_coord = world_transform @ object_to_world
            corners = object_info_to_cuboid(target_lwh, object_to_target_coord)

            if aabb_half_range is not None:
                aabb_half_range_np = np.array(aabb_half_range)
                aabb_range = np.stack([-aabb_half_range_np, aabb_half_range_np])
                aabb = aabb_range.tolist()

                vertices_inside = trimesh.bounds.contains(aabb, corners.reshape(-1, 3))
                if np.all(vertices_inside):
                    all_cuboids.append(corners)

    if len(all_cuboids) == 0:
        return np.zeros((0, 8, 3))
    
    return np.stack(all_cuboids, axis=0)


def get_points_in_cuboid_torch(points, cuboid):
    """
    Args:
    - points: torch.Tensor, shape=(N, 3), lidar points in the ego car coordinate
    - cuboid: dict, object information, with the following keys:
        - object_lwh: list, l, w, h
        - object_to_grid: list, shape=(4, 4), the transformation matrix from object to grid coordinate

        we use it to construct grid-to-object transformation matrix

    Returns:
    - points_in_cuboid: torch.Tensor, shape=(N, 3), the points in the cuboid
    - mask: torch.Tensor, shape=(N,), the mask of points inside the cuboid
    """
    box_l, box_w, box_h = cuboid["object_lwh"]
    
    object_to_grid = torch.tensor(cuboid["object_to_grid"]).to(points.device)
    grid_to_object = torch.inverse(object_to_grid)

    # transform lidar points to world coordinate then to object coordinate
    points = torch.cat([points, torch.ones(points.shape[0], 1).to(points.device)], dim=1)
    points_in_cuboid = torch.matmul(points, grid_to_object.T)
    points_in_cuboid = points_in_cuboid[:, :3]

    # filter points inside the cuboid
    mask = (points_in_cuboid[:, 0] >= -box_l/2) & (points_in_cuboid[:, 0] <= box_l/2) & \
              (points_in_cuboid[:, 1] >= -box_w/2) & (points_in_cuboid[:, 1] <= box_w/2) & \
                (points_in_cuboid[:, 2] >= -box_h/2) & (points_in_cuboid[:, 2] <= box_h/2)

    return points_in_cuboid, mask
    
