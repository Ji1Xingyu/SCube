# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import sys
from pathlib import Path

sys.path.append(Path(__file__).parent.parent.as_posix())

import click
import numpy as np
import viser
from time import sleep
from scube.utils.vis_util import get_waymo_palette
from webdataset import get_sample 

waymo_palette, waymo_mapping = get_waymo_palette()

def visualize_wds_pc(path):
    data = get_sample(path)['pcd.vs01.pth']
    points = data['points'].cpu().numpy()
    semantics = data['semantics'].cpu().numpy()
    visualization_color_category = waymo_mapping[semantics.tolist()].tolist()
    visualizaiton_color = waymo_palette[visualization_color_category].astype(np.float32)

    server = viser.ViserServer()
    server.scene.add_point_cloud(
        name="ground_truth_pc",
        points=points,
        colors=visualizaiton_color
    )

    while True:
        sleep(1)

@click.command()
@click.option("--path", "-p", 
              default="~/gs_ws/SemCity/dataset/15942468615931009553_1243_190_1263_190.tar",  
              help="path to the point cloud file (webdataset .tar)")
def main(path):
    visualize_wds_pc(path)

if __name__ == "__main__":
    main()