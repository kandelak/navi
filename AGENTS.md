# Repository Guidelines

## Project Structure & Modules
- `NAVI Dataset Tutorial.ipynb` is the main walkthrough for downloading and exploring the dataset.
- Root Python helpers: `data_util.py` (annotation/data loading), `mesh_util.py` (mesh IO and coordinate transforms), `transformations.py` (camera/math helpers), `misc_util.py` (shared utilities), `visualization.py` (rendering/display helpers for notebooks).
- Rendering stack lives in `gl/` (`scene_renderer.py`, `egl_context.py`, `camera_util.py`, plus GLSL shaders under `gl/shaders/`).
- Dataset artifacts live outside the repo; point scripts to your local `navi_v1.x` extract (e.g., `/path/to/navi_v1.5/`).

## Environment & Setup
- Python 3 with NumPy, PyTorch, matplotlib, PIL, trimesh, and EGL/GL drivers for rendering.
- Create an isolated env, then install deps:
  - `conda create --name navi python=3`
  - `conda activate navi`
  - `python -m pip install -r requirements.txt`
- Download data (v1.5 default): `wget https://storage.googleapis.com/gresearch/navi-dataset/navi_v1.5.tar.gz` then `tar -xzf navi_v1.5.tar.gz`.

## Build, Test, and Run
- No compile step; run scripts directly after installing deps.
- Quick import check:
  ```bash
  python - <<'PY'
  import data_util, visualization, gl.scene_renderer
  print("imports ok")
  PY
  ```
- Notebook workflow: `jupyter notebook NAVI Dataset Tutorial.ipynb`, set your dataset root in the first cell, and run cells in order to validate loaders and rendering.

## Coding Style & Naming Conventions
- Match existing Python style: 2-space indentation, type hints where practical, and concise helper functions.
- Use `snake_case` for variables/functions, `CapWords` for classes, and `UPPER_SNAKE` for constants.
- Add docstrings for new public helpers; keep logs via the stdlib `logging` module instead of prints.

## Testing Guidelines
- No formal test suite yet; add targeted checks near the code you touch (e.g., minimal render call in `visualization.py`, dataset loader round-trip).
- Keep tests/data-paths configurable via environment variables or function args so they run without bundling data.
- Name new tests `test_<feature>.py` in the repo root or a `gl/tests/` subfolder if you add one.

## Commit & Pull Request Guidelines
- Ensure the Google CLA is signed (see `CONTRIBUTING.md`); all changes go through PR review.
- Write imperative, scoped commit messages (e.g., `Add v1.5 split loader`, `Fix EGL fallback`).
- PRs should summarize intent, note dataset path assumptions, and describe how you validated (import check, notebook cell references, screenshots for visual changes).
- Do not commit dataset artifacts; add local paths to `.gitignore` if needed.
