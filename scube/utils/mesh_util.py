import trimesh
import numpy as np
import torch

def build_scene_mesh_from_all_object_info(all_object_dict, world_transform=np.eye(4), aabb_half_range=None, plyfile = 'assets/car.ply'):
    """
    Create a scene consisting all car object, rescale them, transform them and merge them

    Args:
        all_object_dict: dict, dict containing all object information, webdataset's all_object_info
        world_transform: np.array, [4, 4], world transform matrix to transform meshes to another coordinate system
        aabb_half_range: np.array, [3,], half range of the axis aligned bounding box to filter out objects
        plyfile: str, path to the ply file
    """
    mesh = trimesh.load(plyfile)
    mesh_bounds = mesh.bounds
    mesh_lwh = mesh_bounds[1] - mesh_bounds[0] # [3,]

    scene_meshes = []
    for gid, object_info in all_object_dict.items(): 
        object_to_world = np.array(object_info['object_to_world'])
        target_lwh = np.array(object_info['object_lwh'])
        is_car = object_info['object_type'] == 'car'
        rescale = target_lwh / mesh_lwh

        if is_car:
            transformed_mesh = mesh.copy()
            transformed_mesh.apply_scale(rescale)
            transformed_mesh.apply_transform(world_transform @ object_to_world)

            if aabb_half_range is not None:
                aabb_half_range_np = np.array(aabb_half_range)
                aabb_range = np.stack([-aabb_half_range_np, aabb_half_range_np])
                aabb = aabb_range.tolist()

                vertices_inside = trimesh.bounds.contains(aabb, transformed_mesh.vertices)
                if np.all(vertices_inside):
                    scene_meshes.append(transformed_mesh)

            else:
                scene_meshes.append(transformed_mesh)

    scene_meshes = trimesh.util.concatenate(scene_meshes)
    scene_mesh_vertices = np.asarray(scene_meshes.vertices)
    scene_mesh_faces = np.asarray(scene_meshes.faces)
    
    return scene_mesh_vertices, scene_mesh_faces