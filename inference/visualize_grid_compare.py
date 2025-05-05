# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import glob
import os
import pickle
import sys
import time
from pathlib import Path

sys.path.append(Path(__file__).parent.parent.as_posix())

import click
import numpy as np
import torch
import viser
import viser.transforms as tf
from scube.utils.voxel_util import single_semantic_voxel_to_mesh
from scube.utils.vis_util import get_waymo_palette
from termcolor import colored

waymo_palette, waymo_mapping = get_waymo_palette()

def read_fvdb_grid_and_semantic(path):
    """
    Returns:
        vox_ijk: (N, 3) numpy array
        visualization_color_category: (N,) numpy array
        visualizaiton_color: (N, 3) numpy array
    
    """
    data = torch.load(path)

    try:
        grid = data['points']
    except:
        grid = data['grid']
    vox_ijk = grid.ijk.jdata.cpu().numpy()
    semantics = data['semantics']
    semantics = np.array(semantics.numpy()).astype(np.uint8)
    visualization_color_category = waymo_mapping[semantics.tolist()].tolist()
    visualizaiton_color = waymo_palette[visualization_color_category].astype(np.float32)
    
    return vox_ijk, visualization_color_category, visualizaiton_color


def set_kill_key_button(server, gui_kill_button):
    @gui_kill_button.on_click
    def _(_) -> None:
        print(f"{colored('Killing the sample.', 'red', attrs=['bold'])}")
        setattr(server, "alive", False)
        time.sleep(0.3)


def render_point_cloud(server, points, colors=None, point_size=0.1, name="/simple_pc", port=8080):
    """
        points: [N, 3]
        colors: [3,] or [N, 3]
    """
    setattr(server, "alive", True)

    if colors is None:
        colors = (90, 200, 255)
    
    server.add_point_cloud(
        name=name,
        points=points,
        colors=colors,
        point_size=point_size,
    )

    while True:
        clients = server.get_clients()
        for id, client in clients.items():
            camera = client.camera
            # ic(id, camera.position, camera.wxyz)
        time.sleep(1/60)

        if not server.alive:
            server.scene.reset()
            break


def render_multiple_polygon_mesh(server,
                                 vertices_list,
                                 face_list,
                                 color_list,
                                 name="/simple_mesh",
                                 port=8080):
    """
    Render multiple polygon meshes without texture

    Args:
        vertices_list (List[ndarray]) 
            A list of numpy array of vertex positions. Each array should have shape (V, 3).
        faces_list (List[ndarray]) 
            A list of numpy array of face indices. Each array should have shape (F, 3).
        color_list (List[Tuple[int, int, int] | Tuple[float, float, float] | ndarray]) 
            A list of color of the mesh as an RGB tuple.
    """
    setattr(server, "alive", True)

    for i, vertices in enumerate(vertices_list):
        server.add_mesh_simple(
            name=name + f"_{i}",
            vertices=vertices,
            faces=face_list[i],
            color=color_list[i],
            wxyz=tf.SO3.from_x_radians(0.0).wxyz,
            position=(0, 0, 0),
        )

    while True:
        clients = server.get_clients()
        for id, client in clients.items():
            camera = client.camera
            # ic(id, camera.position, camera.wxyz)
        time.sleep(1/60)

        if not server.alive:
            server.scene.reset()
            break

    time.sleep(1)


def visualize_grid_compare(paths, port, type, size):
    assert len(paths) > 0, "No path is provided"
    center = True

    for path in paths:
        assert os.path.isdir(path), f"{path} is not a directory"
    server = viser.ViserServer(port=port)
    gui_kill_button = server.add_gui_button(
        "Kill", hint="Press this button to kill this sample"
    )
    set_kill_key_button(server, gui_kill_button)

    folder_p = Path(paths[0])
    gt_pt_files = [x for x in os.listdir(folder_p) if '_gt.pt' in x]
    gt_pt_files = [os.path.join(folder_p, x) for x in gt_pt_files]
    sample_num = len(gt_pt_files)
    print(f"Total {sample_num} samples")

    pred_pt_files_diff = []
    
    for i in range(sample_num):
        vox_ijk = []
        visualization_color_category = []
        visualization_color = []
        
        pred_pt_files_diff = []

        gt_file = (Path(paths[0]) / f"{i}_gt.pt").as_posix()
        for path in paths:
            folder_p = Path(path)
            pred_file = (folder_p / f"{i}.pt").as_posix()
            pred_pt_files_diff.append(pred_file)
        
        print_info = '\n'.join(pred_pt_files_diff)
        print(f"Visualizing {i+1}/{sample_num}, from \n{gt_file} and \n{print_info}")

        # add GT
        vox_ijk_gt, visualization_color_category_gt, visualizaiton_color_gt = read_fvdb_grid_and_semantic(gt_file)
        vox_ijk.append(vox_ijk_gt)
        visualization_color_category.append(visualization_color_category_gt)
        visualization_color.append(visualizaiton_color_gt)

        interval = int((np.max(vox_ijk_gt[:,1]) - np.min(vox_ijk_gt[:,1])) * 1.1)

        # add preds
        for idx, pred_file in enumerate(pred_pt_files_diff):
            vox_ijk_pred, visualization_color_category_pred, visualizaiton_color_pred = read_fvdb_grid_and_semantic(pred_file)
            vox_ijk_pred[:,1] += (idx + 1) * interval  # pred is on the right (+Y)
            vox_ijk.append(vox_ijk_pred)
            visualization_color_category.append(visualization_color_category_pred)
            visualization_color.append(visualizaiton_color_pred)

        vox_ijk = np.concatenate(vox_ijk, axis=0)
        if center:
            vox_ijk_center = np.round(np.mean(vox_ijk, axis=0))
            vox_ijk = vox_ijk - vox_ijk_center
            vox_ijk = vox_ijk.astype(np.int32)

        visualization_color_category = np.concatenate(visualization_color_category, axis=0)
        visualization_color = np.concatenate(visualization_color, axis=0)
        
        if type == "voxel":
            visualization_types = np.unique(visualization_color_category)
            cube_v_list = []
            cube_f_list = []
            cube_color_list = []
            geometry_list = []

            for visualization_type in visualization_types:
                mask = visualization_color_category == visualization_type
                if sum(mask) == 0:
                    continue

                cube_v_i, cube_f_i = single_semantic_voxel_to_mesh(vox_ijk[mask])
                color_i = visualization_color[mask][0] 

                cube_v_list.append(cube_v_i)
                cube_f_list.append(cube_f_i)
                cube_color_list.append(color_i)

                from pycg import image, render, vis

                geometry = vis.mesh(cube_v_i, cube_f_i, np.array(color_i).reshape(1,3).repeat(cube_v_i.shape[0], axis=0))
                geometry_list.append(geometry)

            save_render = False
            if save_render:
                scene: render.Scene = vis.show_3d(geometry_list, show=False, up_axis='+Z', default_camera_kwargs={"pitch_angle": 80.0, "fill_percent": 0.8, "fov": 80.0, 'plane_angle': 270})
                img = scene.render_filament()
                img = scene.render_pyrender()
                image.write(img, 'tmp.png')

            render_multiple_polygon_mesh(server, cube_v_list, cube_f_list, cube_color_list)

        elif type == "pc":
            vox_ijk = vox_ijk * 0.1
            render_point_cloud(server, vox_ijk, visualization_color, point_size=size, port=port)

@click.command()
@click.option('--paths', '-p', multiple=True, help='directories of .pt files')
@click.option('--port', '-o', default=8080, help='port number')
@click.option("--type", '-t', default="voxel", help="voxel or pc. voxel can not be used for 1024 resolution grid.")
@click.option("--size", '-s', default=0.05, help="point size for point cloud")
def main(paths, port, type, size):
    visualize_grid_compare(paths, port, type, size)

if __name__ == '__main__':
    main()