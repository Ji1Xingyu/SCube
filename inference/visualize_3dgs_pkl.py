# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import numpy as np
import pickle
import time
import torch
import numpy as np
import viser
import click
from viser import transforms as tf

def RGB2SH(rgb):
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0

def load_gaussian(path):
    """
    Read .pkl gaussian splatting file
    It is a dict, with keys: xyz, opacity, scaling, rotation, rgbs

    Note that the sh_degree is specified to be 0
    """
    with open(path, 'rb') as f:
        gaussians = pickle.load(f)

    gaussians['rgbs'] = np.clip(gaussians['rgbs'], 0, 1)

    if 'features' not in gaussians:
        gaussians['features'] = RGB2SH(gaussians['rgbs']).reshape(-1, 1, 3) # (N, 1, 3)
        gaussians['sh_degree'] = 0
    
    gaussians['ply_format'] = False

    return gaussians

def transform2tensor(x):
    for key, value in x.items():
        if isinstance(value, np.ndarray):
            x[key] = torch.from_numpy(value).float()
            if torch.cuda.is_available():
                x[key] = x[key].cuda()
    return x


def rasterization_gsplat_backend(gaussians_tensors, 
                                 image_height, 
                                 image_width, 
                                 tanfovx, 
                                 tanfovy, 
                                 scale_modifier,
                                 world_view_transform, 
                                 render_alpha):
    from gsplat import rasterization

    focal_length_x = image_width / (2 * tanfovx)
    focal_length_y = image_height / (2 * tanfovy)
    K = torch.tensor(
        [
            [focal_length_x, 0, image_width / 2.0],
            [0, focal_length_y, image_height / 2.0],
            [0, 0, 1],
        ],
    ).to(gaussians_tensors['xyz'])

    means3D = gaussians_tensors['xyz']
    opacity = gaussians_tensors['opacity']
    scales = gaussians_tensors['scaling'] * scale_modifier
    rotations = gaussians_tensors['rotation']
    shs = gaussians_tensors['features']
    sh_degree = gaussians_tensors['sh_degree']
    world_view_transform = torch.from_numpy(world_view_transform).to(gaussians_tensors['xyz'])
    
    render_colors, render_alphas, info = rasterization(
        means=means3D,  # [N, 3]
        quats=rotations,  # [N, 4]
        scales=scales,  # [N, 3]
        opacities=opacity.squeeze(-1),  # [N,]
        colors=shs,
        viewmats=world_view_transform[None],  # [1, 4, 4]
        Ks=K[None],  # [1, 3, 3]
        backgrounds=None,
        width=int(image_width),
        height=int(image_height),
        packed=False,
        sh_degree=sh_degree,
        render_mode='RGB'
    )

    # [1, H, W, 3] -> [3, H, W]
    rendered_image = render_colors[0].permute(2, 0, 1)
    rendered_alpha = render_alphas[0].permute(2, 0, 1)

    if render_alpha:
        return {'image': rendered_image, 'alpha': rendered_alpha}
    
    return {'image': rendered_image}


def render_foreground(camera, height, width, gaussians_tensors, scale_modifier, use_window_aspect):
    """
    Args:
        camera: viser.CameraHandle 
            https://viser.studio/client_handles/#viser.CameraHandle
            https://viser.studio/conventions/
                OpenCV/COLMAP convention.
                    Forward: +Z
                    Up: -Y
                    Right: +X

            wxyz: ndarray[Any, dtype[float64]]
                Corresponds to the R in P_world = [R | t] p_camera. Synchronized automatically when assigned.

            position: ndarray[Any, dtype[float64]]
                Corresponds to the t in P_world = [R | t] p_camera. Synchronized automatically when assigned.
                The look_at point and up_direction vectors are maintained when updating position, which means that updates to position will often also affect wxyz.

            fov: float
                Vertical field of view of the camera, in radians. Synchronized automatically when assigned.

            aspect: float
                Canvas width divided by height. Not assignable.

        gaussians_tensors: dict
            dictionary storing cuda tensor of gaussians

        scale_modifier: float
            scale modifier for the gaussian splatting

        window_aspect: bool
            True for real-time renderer, use camera.aspect 
            False for recording, use width/height

    """
    cam_wxyz = camera.wxyz
    cam_pos = camera.position
    cam_fov = camera.fov # vertical
    cam_aspect = camera.aspect if use_window_aspect else width / height

    # we need provide world-to-camera transformation's R and T
    R_c2w = tf.SO3(np.asarray(cam_wxyz)).as_matrix() # camera to world
    T_c2w = cam_pos # camera to world

    camera_to_world = np.eye(4)
    camera_to_world[:3, :3] = R_c2w
    camera_to_world[:3, 3] = T_c2w

    world_to_camera = np.linalg.inv(camera_to_world)

    # calculate the full_proj_transform
    FoVy = cam_fov
    FoVx = 2 * np.arctan(np.tan(FoVy / 2) * cam_aspect)
    
    render_alpha = True

    with torch.no_grad():
        # [3, H, W], range 0-1
        rendered_output = rasterization_gsplat_backend(
            gaussians_tensors, 
            height, 
            width, 
            np.tan(FoVx/2), 
            np.tan(FoVy/2), 
            scale_modifier,
            world_to_camera, 
            render_alpha
        )

        
    rendered_output['image'] = rendered_output['image'].clamp(0, 1).cpu().numpy().transpose(1, 2, 0) # [H, W, 3]
    rendered_output['alpha'] = torch.mean(rendered_output['alpha'], dim=0, keepdim=True).clamp(0, 1).cpu().numpy().transpose(1, 2, 0) # [H, W, 1]

    return rendered_output


def client_render_and_set_background(client, height, width, gaussians_tensors, 
                                     gui_fovy_modifier, gui_scale_modifier,
                                     use_window_aspect):
    camera = client.camera
    camera.fov = np.deg2rad(gui_fovy_modifier.value)
    render_pkg = render_foreground(camera, height, width, gaussians_tensors, gui_scale_modifier.value, use_window_aspect)
    gs_image = render_pkg['image']
    client.set_background_image(gs_image)
    
    return (gs_image * 255).astype(np.uint8)


def visualize_gaussian(gaussians, center):
    server = viser.ViserServer()
    server.configure_theme(dark_mode=True)
    server.world_axes.visible = True
    
    gaussians_tensors = transform2tensor(gaussians)
    height = 1280 
    width = 1920

    if center:
        gaussians_tensors['xyz'] -= torch.mean(gaussians_tensors['xyz'], dim=0, keepdim=True)

    # add a slider to control the scale_modifier
    with server.add_gui_folder("Control", expand_by_default=True):
        gui_scale_modifier = server.add_gui_slider(
            "Scale", min=0, max=2.0, step=0.05, initial_value=1.0
        )
        gui_fovy_modifier = server.add_gui_slider(
            "Vertical FoV (degree)", min=25, max=120, step=1, initial_value=30
        )
            
    while True:
        clients = server.get_clients()
        for id, client in clients.items():
            render_func_kwargs = {'client': client, 'height': height, 'width': width, 'gaussians_tensors': gaussians_tensors, 
                                  'gui_fovy_modifier': gui_fovy_modifier, 'gui_scale_modifier': gui_scale_modifier,
                                  'use_window_aspect': True}
            client_render_and_set_background(**render_func_kwargs)
        time.sleep(1/20)


@click.command()
@click.option('--path', '-p', help='path to the 3DGS .ply or .splat file')
@click.option('--center', '-c', is_flag=True, help='whether to center the gaussians. default is false.')
def main(path, center):
    gaussians = load_gaussian(path)
    visualize_gaussian(gaussians, center=center)

if __name__ == "__main__":
    main()
