from __future__ import annotations

import json
import os
import random
from typing import Dict, List, Optional, Tuple
import numpy as np
from PIL import Image
import cv2
import colour
import matplotlib.pyplot as plt

def show_correspondences(image_1: Image.Image, image_2: Image.Image,
                         corresp_dict_1: Dict[int, Tuple[int, int]],
                         corresp_dict_2: Dict[int, Tuple[int, int]], resize_factor=1) -> None:
  """Display the intersection of valid correspondences between two images."""
  image_1 = np.array(image_1)
  image_2 = np.array(image_2)
  h1, w1 = image_1.shape[:2]
  h2, w2 = image_2.shape[:2]

  # Handle images of different shapes (in the wild_set images).
  if h1 != h2:
    h_max = max(h1, h2)
    image_1 = np.pad(image_1, [[0, h_max-h1], [0, 0], [0, 0]])
    image_2 = np.pad(image_2, [[0, h_max-h2], [0, 0], [0, 0]])

  # Concatenate the two images to display the correspondences.
  img_corresp = np.concatenate((image_1, image_2), axis=1)
  img_corresp = cv2.resize(
      img_corresp,
      (img_corresp.shape[1] // resize_factor, img_corresp.shape[0] // resize_factor))

  # Sort the correspondences of the left images by Y-coordinate
  corresp_1_as_list = [(k, *v) for k, v in corresp_dict_1.items()]
  corresp_1_as_list = sorted(corresp_1_as_list, key=lambda x: x[1])

  # Create the color gradient.
  red = colour.Color("red")
  colors = list(red.range_to(colour.Color("blue"), len(corresp_1_as_list)))
  
  plt.figure(figsize=(12, 17))
  plt.axis('off')
  plt.imshow(img_corresp)
  for color_idx, (corresp_idx, y1, x1) in enumerate(corresp_1_as_list):
    if corresp_idx in corresp_dict_2:
      y2, x2 = corresp_dict_2[corresp_idx]
      x = [x1 / resize_factor, (x2 + w1) / resize_factor]
      y = [y1 / resize_factor, y2 / resize_factor]
      plt.plot(x, y, color=colors[color_idx].rgb, marker='o')

def debug_plot_random_scene_correspondences(
    correspondences_json_path: str,
    navi_release_root: str,
    output_png_path: str,
    max_points: int = 150,
    seed: Optional[int] = None,
    resize_factor: int = 1,
) -> None:
    """Saves a correspondence visualization PNG for one randomly chosen image pair from a JSON.

    This function reuses the provided `show_correspondences(...)` implementation (interface unchanged)
    to render correspondences between two images from the JSON output of
    `compute_correspondences_for_image_pairs(...)`.

    It:
      1) Loads the JSON (a list of correspondence records).
      2) Randomly selects one (view_1_image_path, view_2_image_path) pair.
      3) Loads both images using PIL from `navi_release_root + relative_path`.
      4) Builds `corresp_dict_1` and `corresp_dict_2` where keys are correspondence indices
         (0..K-1) and values are (y, x) tuples, matching the coordinate convention expected by
         your `show_correspondences` code.
      5) Calls `show_correspondences(...)` and saves the resulting matplotlib figure to PNG.

    Args:
        correspondences_json_path: Path to the JSON file written by
            `compute_correspondences_for_image_pairs(...)`.
        navi_release_root: Root directory of the NAVI release (used to resolve image paths).
        output_png_path: Where to save the resulting PNG.
        max_points: Maximum number of correspondences to visualize (randomly subsampled if more).
        seed: Optional random seed for reproducible selection/subsampling.
        resize_factor: Passed through to `show_correspondences` to downscale the stitched view.

    Raises:
        FileNotFoundError: If the JSON file or selected images are missing.
        ValueError: If the JSON is empty or malformed.
    """
    if not os.path.isfile(correspondences_json_path):
        raise FileNotFoundError(f"JSON not found: {correspondences_json_path}")

    rng = random.Random(seed)

    with open(correspondences_json_path, "r") as f:
        records = json.load(f)

    if not isinstance(records, list) or not records:
        raise ValueError("JSON must be a non-empty list of correspondence records.")

    # Group by image pair.
    groups: Dict[Tuple[str, str], List[dict]] = {}
    for r in records:
        if "view_1_image_path" not in r or "view_2_image_path" not in r:
            raise ValueError("Record missing view_1_image_path/view_2_image_path keys.")
        key = (r["view_1_image_path"], r["view_2_image_path"])
        groups.setdefault(key, []).append(r)

    # Pick one pair at random.
    view1_rel, view2_rel = rng.choice(list(groups.keys()))
    pair_records = groups[(view1_rel, view2_rel)]

    # Optional subsampling of correspondences for clarity.
    if max_points is not None and max_points > 0 and len(pair_records) > max_points:
        pair_records = rng.sample(pair_records, max_points)

    view1_abs = os.path.join(navi_release_root, view1_rel)
    view2_abs = os.path.join(navi_release_root, view2_rel)

    if not os.path.isfile(view1_abs):
        raise FileNotFoundError(f"View 1 image not found: {view1_abs}")
    if not os.path.isfile(view2_abs):
        raise FileNotFoundError(f"View 2 image not found: {view2_abs}")

    image_1 = Image.open(view1_abs).convert("RGB")
    image_2 = Image.open(view2_abs).convert("RGB")

    # Build dicts expected by show_correspondences:
    #   Dict[int, Tuple[int, int]] where value is (y, x).
    # The JSON stores x/y separately, so we convert to (y, x).
    corresp_dict_1: Dict[int, Tuple[int, int]] = {}
    corresp_dict_2: Dict[int, Tuple[int, int]] = {}
    for idx, r in enumerate(pair_records):
        corresp_dict_1[idx] = (int(r["view_1_corresp_y"]), int(r["view_1_corresp_x"]))
        corresp_dict_2[idx] = (int(r["view_2_corresp_y"]), int(r["view_2_corresp_x"]))

    # Reuse your existing plotting routine.
    show_correspondences(
        image_1=image_1,
        image_2=image_2,
        corresp_dict_1=corresp_dict_1,
        corresp_dict_2=corresp_dict_2,
        resize_factor=resize_factor,
    )
    
    out_dir = os.path.dirname(output_png_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    plt.tight_layout()
    plt.savefig(output_png_path, dpi=200)
    plt.close()
