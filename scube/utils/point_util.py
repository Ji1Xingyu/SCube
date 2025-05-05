import torch
import numpy as np

def uniform_point_sampling(range_min_max, interval, transform=None):
    """
    sample points in a uniform grid_points.
    Args:
        range_min_max: [B, 2, 3]
            minimum / maximum value of the xyz range, B is the batch size
        interval: [B, 3]
            interval of the grid_points
        transform: [B, 4, 4]
            transform function to apply to the points
    """
    # breakpoint()
    B = range_min_max.shape[0]
    i = 0
    x = torch.arange(range_min_max[i, 0, 0], range_min_max[i, 1, 0], interval[i, 0], device=range_min_max.device)
    y = torch.arange(range_min_max[i, 0, 1], range_min_max[i, 1, 1], interval[i, 1], device=range_min_max.device)
    z = torch.arange(range_min_max[i, 0, 2], range_min_max[i, 1, 2], interval[i, 2], device=range_min_max.device)

    grid_points = torch.stack(torch.meshgrid(x, y, z), dim=-1)
    grid_points = grid_points.reshape(-1, 3)
    grid_points = grid_points.unsqueeze(0).repeat(B, 1, 1)

    if transform is not None:
        grid_points = torch.cat([grid_points, torch.ones_like(grid_points[:, :, :1])], dim=-1)
        grid_points = torch.bmm(grid_points, transform.transpose(1, 2))
        grid_points = grid_points[:, :, :3]

    return grid_points
   

def interpolate_polyline_to_points_torch(polyline, interpolate_num=100):
    """
    polyline:
        torch.Tensor, shape (N, 3) or list of points 

    Returns:
        torch.Tensor, shape (interpolate_num*N, 3)
    """
    def interpolate_points(previous_vertex, vertex):
        """
        Args:
            previous_vertex: (x, y, z)
            vertex: (x, y, z)

        Returns:
            points: shape (interpolate_num, 3)
        """
        # interpolate between previous_vertex and vertex
        x = torch.linspace(previous_vertex[0], vertex[0], steps=interpolate_num)
        y = torch.linspace(previous_vertex[1], vertex[1], steps=interpolate_num)
        z = torch.linspace(previous_vertex[2], vertex[2], steps=interpolate_num)
        return torch.stack([x, y, z], dim=1)

    points = []
    previous_vertex = None
    for idx, vertex in enumerate(polyline):
        if idx == 0:
            previous_vertex = vertex
            continue
        else:
            points.extend(interpolate_points(previous_vertex, vertex))
            previous_vertex = vertex

    return torch.stack(points)
