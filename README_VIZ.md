# PIDS Embedding Visualizer

## How it Works

The visualization system consists of three components:

1. **`embedding_viz.py`**
   - Runs automatically at the end of the PIDSMaker pipeline.
   - Extracts Word2Vec and trained GNN Encoder embeddings.
   - Reduces embeddings to 3D spatial coordinates using UMAP.
   - Outputs coordinates and metadata to `.json` files.

2. **`viz_manifest.json`**
   - Generated during the evaluation phase.
   - Stores paths to the `.pth` model weights and evaluation statistics required by the visualizer.

3. **`native_viz.py`**
   - Native desktop GUI.
   - Renders the `.json` output using PyQt5 and VisPy (OpenGL).
   - Features dynamic filtering and swapping between Word2Vec and Encoder embedding spaces.

## Dependencies

The visualization module runs entirely within the Docker container. No host environment setup is required.

Required dependencies (automatically installed in the Docker image):
- **System:** `libgl1-mesa-glx`, `libglib2.0-0`, `libxcb-*`, `libxkbcommon-x11-0`, `libdbus-1-3`, `qtwayland5`
- **Python:** `PyQt5`, `vispy`, `PyOpenGL`, `umap-learn`

### Host Requirements

While the code runs in Docker, the GUI window must render on your host machine's display. 
You **must** configure your host to accept X11 connections from Docker.

On your host terminal (outside of Docker), run this once per session:
```bash
xhost +local:docker
```
*(If you see `qt.qpa.xcb: could not connect to display` errors, it means this step was skipped).*

## Commands

### Run Full Pipeline & Generate Visualization
Executes the standard pipeline (featurization -> training -> evaluation) and automatically generates the `.json` coordinates at the end:
```bash
docker compose --env-file .env.local -f compose-pidsmaker.yml run pids python pidsmaker/main.py velox CADETS_E3 --restart_from_scratch
```

### Regenerate Visualization Artifacts
If the model is already trained and you only need to re-run UMAP coordinate generation:
```bash
docker compose --env-file .env.local -f compose-pidsmaker.yml run pids python pidsmaker/main.py velox CADETS_E3 --force_restart viz
```

### Launch Interactive GUI
Starts the native 3D viewer (requires X11 forwarding from the container to your host):
```bash
docker compose --env-file .env.local -f compose-pidsmaker.yml run pids python scripts/native_viz.py velox CADETS_E3
```
