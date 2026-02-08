# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: kmaninis@google.com (Kevis-Kokitsi Maninis)
"""Useful functions for interfacing NAVI data."""

import json
import os
import trimesh
from typing import Text, Optional
import mediapy as media
import numpy as np
from PIL import Image
from PIL import ImageOps
import transformations


def read_image(image_path: Text) -> Image.Image:
    """Reads a NAVI image (and rotates it according to the metadata)."""
    return ImageOps.exif_transpose(Image.open(image_path))


def decode_depth(depth_encoded: Image.Image, scale_factor: float = 10.):
    """Decodes depth (disparity) from an encoded image (with encode_depth).

    Args:
      depth_encoded: The encoded PIL uint16 image of the depth
      scale_factor: float, factor to reduce quantization error. MUST BE THE SAME
        as the value used to encode the depth.

    Returns:
      depth: float[h, w] image with decoded depth values.
    """
    max_val = (2 ** 16) - 1
    disparity = np.array(depth_encoded).astype('uint16')
    disparity = disparity.astype(np.float32) / (max_val * scale_factor)
    disparity[disparity == 0] = np.inf
    depth = 1 / disparity
    return depth


def read_depth_from_png(depth_image_path: str) -> np.ndarray:
    """Reads encoded depth image from an uint16 png file."""
    if not depth_image_path.endswith('.png'):
        raise ValueError(f'Path {depth_image_path} is not a valid png image path.')

    depth_image = Image.open(depth_image_path)
    # Don't change the scale_factor.
    depth = decode_depth(depth_image, scale_factor=10)
    return depth


def convert_to_triangles(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Converts vertices and faces to triangle format float32[N, 3, 3]."""
    faces = faces.reshape([-1])
    tri_flat = vertices[faces, :]
    return tri_flat.reshape((-1, 3, 3)).astype(np.float32)


def camera_matrices_from_annotation(annotation):
    """Convert camera pose and intrinsics to 4x4 matrices."""
    translation = transformations.translate(annotation['camera']['t'])
    rotation = transformations.quaternion_to_rotation_matrix(
        annotation['camera']['q'])
    object_to_world = translation @ rotation
    h, w = annotation['image_size']
    focal_length_pixels = annotation['camera']['focal_length']
    intrinsics = transformations.gl_projection_matrix_from_intrinsics(
        w, h, focal_length_pixels, focal_length_pixels, w // 2, h // 2, zfar=1000)
    return object_to_world, intrinsics


def load_scene_data(query: str, navi_release_root: str,
                    max_num_images: Optional[int] = None, load_video: bool = False):
    """Loads the data of a certain scene from a query."""
    query_data = query.split('-')
    video_id = None
    if len(query_data) == 5:
        object_id, scene_type, scene_idx, camera_model, video_id = query_data
        scene_name = f'{scene_type}-{scene_idx}'
        scene = f'{scene_name}-{camera_model}-{video_id}'
    elif len(query_data) == 4:
        object_id, scene_type, scene_idx, camera_model = query_data
        scene_name = f'{scene_type}-{scene_idx}'
        scene = f'{scene_name}-{camera_model}'
    elif len(query_data) == 2:
        object_id, scene_name = query_data
        scene = scene_name
        assert scene_name == 'wild_set'
    else:
        raise ValueError(f'Query {query} is not valid.')

    annotation_json_path = os.path.join(
        navi_release_root, object_id, scene,
        'annotations.json')
    with open(annotation_json_path, 'r') as f:
        annotations = json.load(f)

    # Load the 3D mesh.
    mesh_path = os.path.join(
        navi_release_root, object_id, '3d_scan', f'{object_id}.obj')
    mesh = trimesh.load(mesh_path)

    # Load the images.
    images = []
    for i_anno, anno in enumerate(annotations):
        if max_num_images is not None and i_anno >= max_num_images:
            break
        image_path = os.path.join(
            navi_release_root, object_id, scene, 'images', anno['filename'])
        images.append(read_image(image_path))

    # Load the video, for video scenes.
    video = None
    if video_id and load_video:
        video_path = os.path.join(
            navi_release_root, object_id, scene, 'video.mp4')
        video = media.read_video(video_path)
    return annotations, mesh, images, video


from typing import Tuple, List
import os
import json
import trimesh


def load_pair_data_for_scene(
    query: str,
    navi_release_root: str,
    image_pair: Tuple[str, str],
):
    """Loads filtered annotations, 3D mesh, and a specific pair of images for a scene.

    This function loads only the annotations and images corresponding to the
    provided image filename pair. All other annotations in the scene are ignored.

    Args:
        query: Scene query string. Supported formats are:
            - "{object_id}-{scene_type}-{scene_idx}-{camera_model}-{video_id}"
            - "{object_id}-{scene_type}-{scene_idx}-{camera_model}"
            - "{object_id}-{scene_name}" (only for 'wild_set')
        navi_release_root: Root directory of the NAVI release.
        image_pair: Tuple of two image filenames to load
            (e.g. ("000.jpg", "001.jpg")).

    Returns:
        annotations: List of annotation dictionaries corresponding to image_pair,
            in the same order as image_pair.
        mesh: Trimesh object of the 3D scan.
        images: List containing the loaded images, in the same order as image_pair.

    Raises:
        ValueError: If the query format is invalid or if any filename in
            image_pair is not found in the annotations.
    """
    query_data = query.split('-')

    if len(query_data) == 5:
        object_id, scene_type, scene_idx, camera_model, _ = query_data
        scene_name = f'{scene_type}-{scene_idx}'
        scene = f'{scene_name}-{camera_model}'
    elif len(query_data) == 4:
        object_id, scene_type, scene_idx, camera_model = query_data
        scene_name = f'{scene_type}-{scene_idx}'
        scene = f'{scene_name}-{camera_model}'
    elif len(query_data) == 2:
        object_id, scene_name = query_data
        scene = scene_name
        assert scene_name == 'wild_set'
    else:
        raise ValueError(f'Query {query} is not valid.')

    annotation_json_path = os.path.join(
        navi_release_root, object_id, scene, 'annotations.json'
    )
    with open(annotation_json_path, 'r') as f:
        all_annotations = json.load(f)

    # Load the 3D mesh.
    mesh_path = os.path.join(
        navi_release_root, object_id, '3d_scan', f'{object_id}.obj'
    )
    mesh = trimesh.load(mesh_path)

    # Build a lookup table for fast access.
    anno_by_filename = {anno['filename']: anno for anno in all_annotations}

    filtered_annotations: List[dict] = []
    images = []

    for filename in image_pair:
        if filename not in anno_by_filename:
            raise ValueError(
                f'Image filename "{filename}" not found in annotations.'
            )

        anno = anno_by_filename[filename]
        filtered_annotations.append(anno)

        image_path = os.path.join(
            navi_release_root, object_id, scene, 'images', filename
        )
        images.append(read_image(image_path))

    return filtered_annotations, mesh, images

