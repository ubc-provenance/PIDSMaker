# Install on NVIDIA DGX Spark

The [NVIDIA DGX Spark](https://www.nvidia.com/en-us/products/workstations/dgx-spark/) differs from a standard x86 GPU server in two ways that affect the installation:

- **ARM CPU (aarch64)**: x86-only binaries used in the default `Dockerfile` (Anaconda, `gosu`, JDK paths) do not run.
- **GB10 Blackwell GPU (compute capability 12.1) with CUDA 13**: the default PyTorch 2.1 + CUDA 12.1 stack does not support this GPU.

PIDSMaker therefore ships a dedicated `Dockerfile.spark` with a compose override `compose-pidsmaker-spark.yml`. The resulting environment differs from the default one as follows:

| | Default (`Dockerfile`) | DGX Spark (`Dockerfile.spark`) |
|---|---|---|
| Base image | `ubuntu:22.04` | `nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04` |
| Conda | Anaconda 2023.03 | Miniforge 25.3.1 |
| Python | 3.9 | 3.11 |
| PyTorch | 2.1.2 + CUDA 12.1 | 2.14.1 + CUDA 13.0 |
| `torch_scatter` | Prebuilt PyG wheel | Compiled from source for `sm_121` |
| `torch.load` default | `weights_only=False` | Restored to `weights_only=False` via `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` |

All other Python packages are installed from the same `requirements.txt` as the default image, with identical pinned versions. Everything else in the [Docker installation guide](./ten-minute-install.md) (dataset download, database loading, W&B) applies unchanged, as the `postgres:17` image is multi-architecture.

## 1. Enable Docker with GPU support

DGX OS ships Docker and the NVIDIA Container Toolkit preinstalled, but your user may not be allowed to use Docker, and Docker may not be configured to use the NVIDIA runtime. Run once:

```sh
sudo usermod -aG docker $USER
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Log out and back in (or run `newgrp docker`) for the group change to take effect, then check that containers see the GPU:

```sh
docker run --rm --gpus all nvidia/cuda:13.0.2-base-ubuntu24.04 nvidia-smi -L
# GPU 0: NVIDIA GB10 (UUID: ...)
```

## 2. Download datasets and load databases

Follow the [Download datasets](./ten-minute-install.md#download-datasets) and [Load databases](./ten-minute-install.md#load-databases) sections of the Docker installation guide. No change is needed for the Spark.

## 3. Build and start the PIDSMaker container

Pass the Spark override on top of the default compose file:

```sh
docker compose -f compose-pidsmaker.yml -f compose-pidsmaker-spark.yml up -d --build
docker compose exec pids bash
```

The first build takes a while, mostly because `torch_scatter` is compiled for the GB10.

!!! note "VSCode Dev Container"
    `.devcontainer/devcontainer.json` builds from the default `Dockerfile`. To use the Dev Container on a Spark, add the override to its compose files:
    ```json
    "dockerComposeFile": ["../compose-pidsmaker.yml", "../compose-pidsmaker-spark.yml"],
    ```

## 4. Check the installation

Inside the `pids` container:

```sh
python -c "import torch, torch_scatter, torch_geometric; print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
# 2.14.1+cu130 NVIDIA GB10 (12, 1)
```

You can then run PIDSMaker as described in the [Docker installation guide](./ten-minute-install.md#weights-biases-interface) and the [tutorials](./tuned_systems.md).
