from __future__ import annotations

import json
import os
import random
from typing import Optional, Tuple, Dict, List

import data_util
import visualization
from tqdm import tqdm


def compute_correspondences_for_image_pairs(
    navi_release_root: str,
    pairs_txt_path: str,
    num_samples_per_scene: int,
    output_path: str,
    random_subsample_size: Optional[int] = None,
    seed: int = 0,
) -> None:
    """Computes 2D-2D correspondences for image pairs listed in a text file and writes them to JSON.

    The input text file must contain one pair per line, with three whitespace-separated fields:
        <view_1_image_path> <view_2_image_path> <angular_rot>

    Example line:
        3d_dollhouse_sink/multiview-00-pixel_5/images/000.jpg
        3d_dollhouse_sink/multiview-00-pixel_5/images/012.jpg
        177.80534415031153

    For each line, the function:
      1) Parses object_id, scene folder, and the two image filenames.
      2) Builds a query string of the form "{object_id}-{scene_type}-{scene_idx}-{camera_model}".
      3) Loads the mesh and the two images using `data_util.load_pair_data_for_scene`.
      4) Samples 3D points on the mesh and projects them into both images using
         `data_util.sample_and_project_on_image_pair`.
      5) Intersects visible samples across both views.
      6) Writes one JSON entry per correspondence.

    Args:
        navi_release_root: Root directory of the NAVI release.
        pairs_txt_path: Path to the .txt file listing image pairs and angular rotation.
        num_samples_per_scene: Number of 3D samples drawn per scene.
        output_path: Path to the output .json file.
        random_subsample_size: If provided, randomly samples this many rows from the txt file
            before processing.
        seed: Random seed used for subsampling.

    Output JSON format:
        A list of dictionaries, each with the following keys:
          - view_1_image_path (str)
          - view_2_image_path (str)
          - angular_rot (float)
          - view_1_corresp_x (int)
          - view_1_corresp_y (int)
          - view_2_corresp_x (int)
          - view_2_corresp_y (int)

        Each input row can generate zero or many output entries depending on
        how many 3D samples are visible in both views.
    """
    if not os.path.isfile(pairs_txt_path):
        raise FileNotFoundError(f"pairs_txt_path not found: {pairs_txt_path}")

    # ---- Read and parse txt rows ----
    rows: List[Tuple[str, str, float]] = []
    with open(pairs_txt_path, "r") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) != 3:
                raise ValueError(
                    f"Invalid line {line_no}: expected 3 fields, got {len(parts)}"
                )

            p1, p2, rot_str = parts
            try:
                rot = float(rot_str)
            except ValueError as e:
                raise ValueError(
                    f"Invalid rotation value on line {line_no}: {rot_str}"
                ) from e

            rows.append((p1, p2, rot))

    # ---- Optional subsampling ----
    if random_subsample_size is not None and random_subsample_size > 0:
        rng = random.Random(seed)
        rows = rng.sample(rows, min(random_subsample_size, len(rows)))

    # ---- Helpers ----
    def _parse_rel_image_path(rel_path: str) -> Tuple[str, str, str]:
        """Returns (object_id, scene_folder, filename)."""
        parts = rel_path.replace("\\", "/").split("/")
        if len(parts) < 4 or parts[-2] != "images":
            raise ValueError(
                f"Unexpected image path format: {rel_path}"
            )
        return parts[0], parts[1], parts[-1]

    def _scene_folder_to_query(object_id: str, scene_folder: str) -> str:
        """Builds NAVI query string from scene folder."""
        segs = scene_folder.split("-")
        if len(segs) != 3:
            raise ValueError(
                f"Unexpected scene folder format: {scene_folder}"
            )
        scene_type, scene_idx, camera_model = segs
        return f"{object_id}-{scene_type}-{scene_idx}-{camera_model}"

    # ---- Compute correspondences ----
    output_records: List[Dict[str, object]] = []
    last_completed_index = -1

    output_dir = os.path.dirname(output_path) or "."
    error_file_path = os.path.join(output_dir, "error_file.txt")

    try:
        for idx, (view1_path, view2_path, angular_rot) in enumerate(
            tqdm(rows, desc="Processing image pairs")
        ):
            obj1, scene1, fname1 = _parse_rel_image_path(view1_path)
            obj2, scene2, fname2 = _parse_rel_image_path(view2_path)

            if obj1 != obj2 or scene1 != scene2:
                raise ValueError(
                    "Image pair must belong to the same object and scene:\n"
                    f"  {view1_path}\n  {view2_path}"
                )

            query = _scene_folder_to_query(obj1, scene1)

            annotations, mesh, images = data_util.load_pair_data_for_scene(
                query=query,
                navi_release_root=navi_release_root,
                image_pair=(fname1, fname2),
            )

            triangles, _, _ = visualization.prepare_mesh_rendering_info(mesh)

            samples_visible_1, samples_visible_2 = (
                data_util.sample_and_project_on_image_pair(
                    triangles=triangles,
                    annotations=(annotations[0], annotations[1]),
                    images=(images[0], images[1]),
                    num_samples=num_samples_per_scene,
                )
            )

            intersected = data_util.intersect_visible_samples(
                (samples_visible_1, samples_visible_2)
            )

            for (_, ((x1, y1), (x2, y2))) in intersected.items():
                output_records.append({
                    "view_1_image_path": view1_path,
                    "view_2_image_path": view2_path,
                    "angular_rot": float(angular_rot),
                    "view_1_corresp_x": int(x1),
                    "view_1_corresp_y": int(y1),
                    "view_2_corresp_x": int(x2),
                    "view_2_corresp_y": int(y2),
                })

            last_completed_index = idx

    except Exception:
        # Persist partial progress and index of last successful row.
        os.makedirs(output_dir, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output_records, f, indent=2)
        with open(error_file_path, "w") as ef:
            ef.write(str(last_completed_index))
        raise
    else:
        # Clean successful run output.
        os.makedirs(output_dir, exist_ok=True)
        if os.path.exists(error_file_path):
            os.remove(error_file_path)
        with open(output_path, "w") as f:
            json.dump(output_records, f, indent=2)
