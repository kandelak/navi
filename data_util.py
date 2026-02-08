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

import mesh_util
import transformations
import torch as t
from gl import scene_renderer
from gl import camera_util


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


# Type aliases for readability.
Pixel = tuple[int, int]
VisibleSamples = dict[int, Pixel]
VisiblePair = tuple[VisibleSamples, VisibleSamples]
IntersectedVisiblePair = dict[int, tuple[Pixel, Pixel]]

def sample_and_project_on_image_pair(
        *,
        triangles: np.ndarray,
        annotations: tuple[dict, dict],
        images: tuple[np.ndarray, np.ndarray],
        num_samples: int = 100,
) -> VisiblePair:
    """Samples 3D points on a mesh and projects them into a pair of images, keeping only visible points.

    The function:
      1) Samples `num_samples` 3D points on the mesh surface defined by `triangles`.
      2) For each of the two views (annotation + image), projects the sampled 3D points into the
         2D pixel grid and filters out points that are not visible (e.g., outside the image,
         occluded, or failing any validity checks implemented by `project_and_filter_sample_coordinates`).

    Notes:
      - This function assumes `mesh_util.sample_points_from_mesh` returns sampled points in a stable
        order and that `project_and_filter_sample_coordinates` returns a dictionary mapping
        sample indices -> (x, y) pixel coordinates.

    Args:
        triangles: Mesh triangles used for sampling and projection. Expected to be compatible with
            `mesh_util.sample_points_from_mesh` and `project_and_filter_sample_coordinates`.
            Commonly this is an array shaped (T, 3, 3) containing triangle vertex coordinates.
        annotations: A tuple of exactly two annotation dicts (one per image). Each annotation must
            contain the camera parameters expected by `project_and_filter_sample_coordinates`.
        images: A tuple of exactly two images (one per view), aligned with `annotations`.
            Each image is typically a numpy array shaped (H, W, C) or (H, W).
        num_samples: Number of 3D surface samples to draw from the mesh.

    Returns:
        samples_visible: A pair of dictionaries `(samples_visible_1, samples_visible_2)`.

            Each dictionary maps:
              - key: `int` sample index in `[0, num_samples - 1]`, referring to the position of the
                corresponding 3D point in the internally sampled array returned by
                `mesh_util.sample_points_from_mesh`.
              - value: `Tuple[int, int]` pixel coordinate `(x, y)` on the *dense* 2D image grid where
                that 3D sample projects and is considered visible.

            If a sample index is **absent** from a dictionary, that means the corresponding 3D point
            was *not* visible in that view (e.g., projected outside bounds, occluded, invalid depth, etc.).

            Concretely:
              - `samples_visible[0]` contains visibility + 2D locations for the first image.
              - `samples_visible[1]` contains visibility + 2D locations for the second image.

            Return type:
              Tuple[
                Dict[int, Tuple[int, int]],
                Dict[int, Tuple[int, int]]
              ]

    """
    if len(annotations) != 2 or len(images) != 2:
        raise ValueError(
            "annotations and images must both be tuples of length 2 "
            "(one per view in the pair)."
        )

    # 1) Sample the mesh surface.
    sampled_points, _ = mesh_util.sample_points_from_mesh(
        triangles, num_sample_points=num_samples
    )

    # 2) Project/filter for each of the two views.
    samples_visible_1 = _project_and_filter_sample_coordinates(
        triangles, annotations[0], sampled_points, images[0]
    )
    samples_visible_2 = _project_and_filter_sample_coordinates(
        triangles, annotations[1], sampled_points, images[1]
    )

    return samples_visible_1, samples_visible_2



def _project_and_filter_sample_coordinates(
    mesh_triangles: t.tensor, annotation, sampled_points: t.tensor,
    image: Image.Image) -> dict[int, tuple[int, int]]:
  """Returns the sampled points, projected on the image, that are visible from the current view."""

  object_to_world, intrinsics = camera_matrices_from_annotation(annotation)

  # Render the 3D model alignment.
  mesh_triangles_aligned = transformations.transform_mesh(
      mesh_triangles, object_to_world)
  rend = scene_renderer.render_scene(
      mesh_triangles_aligned, view_projection_matrix=intrinsics,
      output_type=t.float32, clear_color=(0,0,0,0),
      image_size=image.size[::-1], cull_back_facing=False, return_rgb=False)
  depth = rend[:, :, 3].numpy()

  # Align the sampled points.
  sampled_points_world = transformations.transform_points(
      sampled_points, object_to_world)
  sampled_points_screen = transformations.transform_points(
      sampled_points_world, intrinsics)

  # Convert from OpenGL space to image space.
  sampled_points_screen += t.tensor([1., 1., 0])
  sampled_points_screen *= t.tensor([image.size[0]/2, image.size[1]/2, 1])
  samples = t.concat(
      (sampled_points_screen[:, :2], sampled_points_world[:, 2:3]),
      dim=1).numpy()

  # Discard points where the depth doesn't match the OpenGL depth buffer.
  coords = {}
  for i_sample, sample in enumerate(samples):
    y = round(sample[1])
    x = round(sample[0])
    z = sample[2]
    if abs(depth[y, x] - z) < 1:
      coords[i_sample] = (x, y)
  return coords


def intersect_visible_samples(
    samples_visible_pair: VisiblePair,
) -> IntersectedVisiblePair:
    """Computes the intersection of visible 3D samples across two views.

    This function finds the set of 3D sample indices that are visible in
    *both* images of a pair and returns their corresponding 2D projections
    in each image.

    Args:
        samples_visible_pair: A tuple `(samples_visible_1, samples_visible_2)`
            where each element is a dictionary mapping:
              - key: `int` index of a sampled 3D point
              - value: `(x, y)` pixel coordinates where that 3D point projects
                in the corresponding image.

    Returns:
        intersected_samples: A dictionary mapping:
          - key: `int` sample index that is visible in *both* images
          - value: `((x1, y1), (x2, y2))`, where:
              - `(x1, y1)` is the pixel location in the first image
              - `(x2, y2)` is the pixel location in the second image

        Return type:
            Dict[int, Tuple[Tuple[int, int], Tuple[int, int]]]

        Semantics:
          - Only sample indices present in *both* input dictionaries
            are included.
          - The ordering of keys is not guaranteed.
          - If a sample index is missing from either view, it is excluded.

    """
    samples_visible_1, samples_visible_2 = samples_visible_pair

    common_sample_indices = (
        samples_visible_1.keys() & samples_visible_2.keys()
    )

    intersected_samples: IntersectedVisiblePair = {
        idx: (samples_visible_1[idx], samples_visible_2[idx])
        for idx in common_sample_indices
    }

    return intersected_samples
