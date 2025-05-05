import torch
import numpy as np
import cv2

def vis_depth(depth, minmax=None, valid_farthest=300):
    """
    Args:
        depth: np.ndarray or torch.Tensor,
            shape (H, W)

        minmax: 
            if None, use adaptive minmax according to the depth values
            if [d_min, d_max], use the provided minmax

        valid_farthest: float
            the farthest valid depth value, used to filter out invalid depth values like inf

        cmap: 
            https://docs.opencv.org/3.4/d3/d50/group__imgproc__colormap.html
            e.g. cv2.COLORMAP_JET

    Returns:
        colored_depth: (H, W, 3)
    """
    from matplotlib import cm
    import matplotlib as mpl

    is_tensor = False
    if isinstance(depth, torch.Tensor):
        depth = depth.detach().cpu().numpy()
        is_tensor = True

    depth = np.nan_to_num(depth) # change nan to 0
    depth_valid_count = (depth < valid_farthest).sum()

    if minmax is None:
        constant_max = np.percentile(depth[depth < valid_farthest], 99.5)
        constant_min = np.percentile(depth, 0.5) if np.percentile(depth, 0.5) < constant_max else 0
    else:
        constant_min, constant_max = minmax

    normalizer = mpl.colors.Normalize(vmin=constant_min, vmax=constant_max)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma_r')
    
    colored_depth = mapper.to_rgba(depth)[:, :, :3] # range [0, 1]
    colored_depth = (colored_depth * 255).astype(np.uint8) # range [0, 255]

    if is_tensor:
        colored_depth = torch.from_numpy(colored_depth)

    return colored_depth


def least_square_fit(relative_dense_depth, absolute_sparse_depth, sparse_depth_mask):
    """
    Args:
        relative_dense_depth: torch.Tensor, shape (H, W), the relative depth map
        absolute_sparse_depth: torch.Tensor, shape (H, W), the absolute depth map
        sparse_depth_mask: torch.Tensor, shape (H, W), the mask of the sparse depth map

    Returns:
        absolute_dense_depth: torch.Tensor, shape (H, W), the absolute depth map

    Note that if 3 inputs have the same shape, they can be directly passed to this function.
    """
    relative_depths = relative_dense_depth[sparse_depth_mask]
    absolute_depths = absolute_sparse_depth[sparse_depth_mask]

    numerator = torch.sum(relative_depths * absolute_depths)
    denominator = torch.sum(relative_depths ** 2)

    if denominator == 0:
        raise ValueError("The denominator is zero, cannot perform the least square fit.")

    scale = numerator / denominator
    print(f"Rescaling factor for least square fit: {scale}")
    
    # apply scale to the relative depth
    absolute_dense_depth = relative_dense_depth * scale

    return absolute_dense_depth


def least_square_fit_batch(relative_dense_depth, absolute_sparse_depth, sparse_depth_mask):
    """
    Args:
        relative_dense_depth: torch.Tensor, shape (..., H, W, 1), the relative depth map
        absolute_sparse_depth: torch.Tensor, shape (..., H, W, 1), the absolute depth map
        sparse_depth_mask: torch.Tensor, shape (..., H, W, 1), the mask of the sparse depth map

    Returns:
        absolute_dense_depth: torch.Tensor, shape (..., H, W, 1), the absolute depth map

    Note that ... can be [B, N], means there are B*N samples. least_square_fit_batch has different scale 
    for each sample
    """

    numerator = torch.sum(relative_dense_depth * absolute_sparse_depth * sparse_depth_mask, dim=(-3, -2, -1)) # shape (...,)
    denominator = torch.sum(relative_dense_depth * relative_dense_depth * sparse_depth_mask, dim=(-3, -2, -1)) # shape (...,)

    scale = numerator / (denominator + 1e-6) # avoid divide by zero

    # if divide by zero. set the scale to 1
    scale[denominator == 0] = 1
    absolute_dense_depth = relative_dense_depth * scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

    return absolute_dense_depth


def project_2D_to_3D(depth, camera_intrinsic, camera_to_world):
    """
    Args:
        depth: torch.Tensor, shape (B, 1, H, W)
        camera_intrinsic: torch.Tensor, shape (B, 6), fx, fy, cx, cy, w, h
        camera_to_world: torch.Tensor, shape (B, 4, 4)
    
    Returns:
        points_3d: torch.Tensor, shape (B, H*W, 3)
    """

    B, _, H, W = depth.shape
    assert H == round(camera_intrinsic[0, 5].item()), f"The height of the depth map and camera intrinsic do not match. H is {H} and camera_intrinsic[0, 5] is {camera_intrinsic[0, 5]}"

    # Create a grid of pixel coordinates
    x = torch.arange(W, dtype=torch.float32, device=depth.device)
    y = torch.arange(H, dtype=torch.float32, device=depth.device)
    x, y = torch.meshgrid(x, y, indexing='xy') # must pass indexing='xy'
    
    # Stack pixel coordinates and flatten
    pixel_coords = torch.stack((x.flatten(), y.flatten(), torch.ones_like(x.flatten())), dim=0)  # (3, H*W)

    # Inverse of the intrinsic matrix
    camera_intrinsic_K = torch.eye(3).expand(B, -1, -1).to(depth.device)
    camera_intrinsic_K[:, 0, 0] = camera_intrinsic[:, 0]
    camera_intrinsic_K[:, 1, 1] = camera_intrinsic[:, 1]
    camera_intrinsic_K[:, 0, 2] = camera_intrinsic[:, 2]
    camera_intrinsic_K[:, 1, 2] = camera_intrinsic[:, 3]
    K_inv = torch.linalg.inv(camera_intrinsic_K)  # (B, 3, 3)

    # Convert pixel coordinates to normalized camera coordinates
    normalized_camera_coords = K_inv @ pixel_coords.unsqueeze(0)  # (B, 3, H*W)

    # Get the depth values and reshape
    current_depth = depth.view(B, 1, -1)  # (B, 1, H*W)

    # Scale by depth
    camera_coords = normalized_camera_coords * current_depth  # (B, 3, H*W)

    # Convert to homogeneous coordinates
    camera_coords_homogeneous = torch.cat((camera_coords, torch.ones(B, 1, H * W, device=depth.device)), dim=1)  # (B, 4, H*W)

    # Transform to world coordinates
    world_coords_homogeneous = camera_to_world @ camera_coords_homogeneous  # (B, 4, H*W)

    # Return the 3D points in world coordinates (X, Y, Z)
    points_3d = world_coords_homogeneous[:, :3, :].transpose(1, 2)  # (B, H*W, 3)

    return points_3d
