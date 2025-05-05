# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

from .skybox_panorama_full import SkyboxPanoramaFull
from .skybox_null import SkyboxNull

def convert_to_camel_case(string):
    return ''.join(word.capitalize() for word in string.split('_'))

__all__ = ['SkyboxPanoramaFull',
           'SkyboxNull' 
           'convert_to_camel_case']