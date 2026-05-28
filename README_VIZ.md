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

## Quickstart / How to Run

### Step 1: Prepare the Host Environment
While the code runs in Docker, the GUI window must render on your host machine's display. You **must** configure your host to accept X11 connections. On your host terminal, run this once per session:
```bash
xhost +local:docker
```
*(If you see `qt.qpa.xcb: could not connect to display` errors, it means this step was skipped).*

### Step 2: Build and Start the Container
From your host machine, build the image (which installs dependencies like PyTorch and RAPIDS) and start the persistent container:
```bash
docker compose -f compose-pidsmaker.yml build pids
docker compose -f compose-pidsmaker.yml up -d
```

### Step 3: Enter the Container
Drop into the running container to execute commands:
```bash
docker exec -it pidsmaker-pids-1 bash
```

### Step 4: Run the Pipeline (Inside Container)
To run the full end-to-end pipeline (featurization -> training -> evaluation -> viz) from scratch:
```bash
python pidsmaker/main.py velox THEIA_E3 --restart_from_scratch
```

If the model is already trained and you only need to re-run the UMAP coordinate generation:
```bash
python pidsmaker/main.py velox THEIA_E3 --force_restart viz
```

### Step 5: Launch Interactive GUI (Inside Container)
Starts the native 3D viewer to explore the generated embeddings:
```bash
python scripts/native_viz.py velox THEIA_E3
```
