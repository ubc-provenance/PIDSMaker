---
title: Installation (Docker)
description: Install PIDSMaker with Docker, load the preprocessed provenance databases (DARPA TC, OpTC, ATLASv2, CARBANAKv2) and the network databases (LANL, OpTC flows).
---

# Installation with Docker

Ten minutes from a fresh machine to your first run. PIDSMaker runs in two containers: a **Postgres** database that stores the datasets, provenance or network events, and a **pids** container with the Python environment where you run experiments.
{ .lead }

!!! info "Requirements"
    One of the two setups we recommend:

    - :fontawesome-brands-linux: **Linux** with [Docker](https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository), ideally with an :simple-nvidia: **NVIDIA GPU**, for all datasets. Avoid running Docker as root: [manage Docker as a non-root user](https://docs.docker.com/engine/install/linux-postinstall/#manage-docker-as-a-non-root-user). Without a GPU, PIDSMaker [falls back to CPU](quickstart.md#gpu-or-cpu).
    - :fontawesome-brands-apple: **macOS** (tested on Apple Silicon) with [Docker Desktop](https://docs.docker.com/desktop/setup/install/mac-install/), on CPU (Docker has no access to Apple GPUs), for most datasets: 32 GB of RAM is ideal. Tested end to end on `CADETS_E3` and `LANL`; the largest datasets (`TRACE_E3` and the E5 datasets) need more memory and disk than a laptop has.

    In both cases, you need enough disk space for the datasets you pick (see the table below) and for the artifacts of your runs (a single run can produce 15 GB or more on E3 datasets). No Docker, e.g. on an HPC cluster? Use [Apptainer](apptainer.md) instead.

## 1. Download the datasets

PIDSMaker reads its [datasets](../guide/datasets.md) from Postgres databases, one per dataset. DARPA TC and OpTC are very large and hard to process: we provide **preprocessed versions** of all datasets as Postgres database dumps. The dumps of the network datasets are built from the preprocessed files of [LARES](https://github.com/TristanBilot/lares).

| Dataset | Family | Download (GB) | Loaded in Postgres (GB) |
|---|---|--:|--:|
| `CADETS_E3` | <span class="pm-family pm-family--prov">Provenance</span> | 1.4 | 10 |
| `THEIA_E3` | <span class="pm-family pm-family--prov">Provenance</span> | 1.1 | 12 |
| `CLEARSCOPE_E3` | <span class="pm-family pm-family--prov">Provenance</span> | 0.6 | 4.8 |
| `FIVEDIRECTIONS_E3` | <span class="pm-family pm-family--prov">Provenance</span> | 3.2 | 22 |
| `TRACE_E3` | <span class="pm-family pm-family--prov">Provenance</span> | 11 | 100 |
| `CADETS_E5` | <span class="pm-family pm-family--prov">Provenance</span> | 36 | 276 |
| `THEIA_E5` | <span class="pm-family pm-family--prov">Provenance</span> | 5.8 | 36 |
| `CLEARSCOPE_E5` | <span class="pm-family pm-family--prov">Provenance</span> | 6.2 | 49 |
| `FIVEDIRECTIONS_E5` | <span class="pm-family pm-family--prov">Provenance</span> | 39 | 280 |
| `TRACE_E5` | <span class="pm-family pm-family--prov">Provenance</span> | 91 | 710 |
| `optc_h201` | <span class="pm-family pm-family--prov">Provenance</span> | 2 | 9 |
| `optc_h501` | <span class="pm-family pm-family--prov">Provenance</span> | 1.5 | 6.7 |
| `optc_h051` | <span class="pm-family pm-family--prov">Provenance</span> | 1.7 | 7.7 |
| `ATLASV2_EDR` | <span class="pm-family pm-family--prov">Provenance</span> | 0.2 | 1 |
| `CARBANAKV2_EDR` | <span class="pm-family pm-family--prov">Provenance</span> | 1.1 | 6.6 |
| `LANL` | <span class="pm-family pm-family--net">Network</span> | 0.7 | 8.5 |
| `OPTC_NET` | <span class="pm-family pm-family--net">Network</span> | 21 | 275 |

**Download** is the size of the dump, **loaded in Postgres** the size of the database once loaded.

!!! tip "New to PIDSMaker?"
    Start with `CADETS_E3`, `THEIA_E3` or `CLEARSCOPE_E3` for provenance-based detection, and `LANL` for network detection: they are small, quick to load, and the most widely used in papers.

### Download the dumps { #download-the-dumps }

=== "Option 1: Google Drive"

    Open the [shared folder](https://drive.google.com/drive/folders/1hqfz8__zVqb3QzBuOI2SxrW4lLIdYqFr) and download the `.dump` files of the datasets you want into a `data/` folder at the root of the repository.

=== "Option 2: From the command line"

    On a server, it is easier to download from the command line. You first need a Google API access token ([details](https://stackoverflow.com/a/67550427/10183259)):

    1. Open the [OAuth 2.0 Playground](https://developers.google.com/oauthplayground/).
    2. In the **Select the Scope** box, paste `https://www.googleapis.com/auth/drive.readonly`.
    3. Click **Authorize APIs**, log into your Google account, then click **Exchange authorization code for tokens**.
    4. Copy the **access_token**.

    Then download the dumps (dataset names in lowercase) into `./data`:

    ```shell
    # A comma-separated list of datasets
    ./download_datasets.sh cadets_e3,optc_h201,lanl YOUR_ACCESS_TOKEN

    # Or all datasets
    ./download_datasets.sh all YOUR_ACCESS_TOKEN
    ```

    Downloads resume where they stopped if you re-run the command.

You can also build these databases yourself, the provenance databases from the official DARPA files and the network databases from the preprocessed files of LARES: see [Building datasets from scratch](from-scratch.md).

<span id="macos"></span>

## 2. Set up Docker

=== ":fontawesome-brands-linux: Linux + :simple-nvidia: NVIDIA GPU"

    Install the NVIDIA Container Toolkit, so that the containers can use the GPU. Skip this step to run on CPU only.

    ```shell
    # Add the NVIDIA package repository
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
      sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
      sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

    # Update and install
    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit

    # Restart services
    sudo systemctl restart docker
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
    ```

=== ":fontawesome-brands-apple: macOS"

    1. **Give memory to Docker.** On Linux, containers use all the RAM of the machine. On macOS, they run in the virtual machine of Docker Desktop, limited to 8 GB by default, which the compose files cannot raise. Runs on provenance datasets can take more than 15 GB, and are killed when the VM runs out of memory. Give the VM about 75% of your RAM, e.g. 24 GB on a 32 GB laptop:

        ```shell
        # Memory and swap of the VM, in MiB (restarts Docker Desktop, which stops the running containers)
        plutil -replace MemoryMiB -integer 24576 ~/Library/Group\ Containers/group.com.docker/settings-store.json
        plutil -replace SwapMiB -integer 4096 ~/Library/Group\ Containers/group.com.docker/settings-store.json
        docker desktop restart

        # Check the memory of the VM, in GB
        docker info --format '{{.MemTotal}}' | awk '{printf "%.1f GB\n", $1 / 1024^3}'
        ```

        PIDSMaker prints a warning at the beginning of a run when less than 16 GB are available.

    2. **Let Docker read the repository.** If the repository is in a protected folder such as `~/Documents` or `~/Desktop`, `docker compose` fails with `mkdir /host_mnt/Users/...: operation not permitted`. Allow Docker in **System Settings > Privacy & Security > Files and Folders** (or **Full Disk Access**), then restart Docker Desktop.

    On Apple Silicon, the image is built for ARM (`aarch64`), with the CPU build of PyTorch, and PIDSMaker limits PyTorch to 4 CPU threads (`OMP_NUM_THREADS=4`): the VM exposes the performance and efficiency cores of the Mac as identical CPUs, and with one thread per CPU, operations on small tensors are up to 100 times slower. Set `OMP_NUM_THREADS` yourself to override it.

## 3. Configure the paths

Copy the environment template at the root of the repository:

```shell
cp .env.local .env
```

Then edit `.env`:

| Variable | Description |
|---|---|
| `INPUT_DIR` | Absolute path to the folder that contains the `.dump` files (e.g. the `data/` folder). Mounted as `/data` in the Postgres container. |
| `ARTIFACTS_DIR` | *(optional)* Where all files generated by the pipeline are written (multiple GBs). Mounted as `/home/artifacts` in the `pids` container. |
| `DOCKER_PORT` | *(optional)* Host port of Postgres, `5432` by default. |
| `VIZ_PORT`, `PIDS_PORT` | *(optional)* Host ports of the visualization server (`5000` by default) and of port 8000 of the `pids` container (`8000` by default). On macOS, set `VIZ_PORT=5001`: port 5000 is used by AirPlay Receiver (or turn it off in **System Settings > General > AirDrop & Handoff**). |
| `COMPOSE_PROJECT_NAME` | *(optional)* Prefix of the `pids` container name, useful if several users share a machine. |

!!! warning
    Each time you modify `.env`, run `source .env` before running `docker compose` again.

## 4. Load the databases

Build and start the Postgres container:

```shell
docker compose -p postgres -f compose-postgres.yml up -d --build
```

Open a shell into it:

```shell
docker compose -p postgres exec postgres bash
```

=== "Load all downloaded dumps"

    If you have enough space to load every dump in your `data/` folder:

    ```shell
    ./scripts/load_dumps.sh
    ```

    Databases that already contain tables are skipped, so the script can safely be re-run after downloading more datasets.

=== "Load one dataset at a time"

    With limited disk space, load the databases one by one (`DATASET_NAME` in lowercase, e.g. `cadets_e3`):

    ```shell
    pg_restore -U postgres -h localhost -p 5432 -d DATASET_NAME /data/DATASET_NAME.dump
    ```

Once the databases are loaded, you won't need this container's shell anymore:

```shell
exit
```

## 5. Open the PIDSMaker container

All coding and experiments happen in the `pids` container. The repository is mounted at `/home/pids`, so code changes are seen without rebuilding the image.

=== ":material-microsoft-visual-studio-code: VS Code Dev Container (recommended)"

    Install the [Dev Containers](https://code.visualstudio.com/docs/devcontainers/create-dev-container) extension, then press ++ctrl+shift+p++ and run **Dev Containers: Open Folder in Container**.
    You get a full IDE inside the container, including the debugger.

=== ":lucide-terminal: Terminal"

    ```shell
    docker compose -f compose-pidsmaker.yml up -d --build
    docker compose exec pids bash
    ```

!!! note ":fontawesome-brands-apple: macOS, or Linux without an NVIDIA GPU"
    `up` fails with `could not select device driver "nvidia"`. Add the CPU override, which removes the GPU reservation (requires Docker Compose 2.24 or later, see `docker compose version`):

    ```shell
    docker compose -f compose-pidsmaker.yml -f compose-pidsmaker.cpu.yml up -d --build
    ```

    With VS Code, add `"../compose-pidsmaker.cpu.yml"` to `dockerComposeFile` in `.devcontainer/devcontainer.json`.

## 6. Log in to Weights & Biases

PIDSMaker uses [Weights & Biases](https://wandb.ai/site/) (W&B) to log, plot and keep the history of experiments. We **highly** recommend it.
Create an account, then log in from the `pids` container with the API key from your W&B dashboard:

```shell
wandb login
```

Runs started with `--wandb`, or with `./run.sh`, then send their logs and results to W&B.

## Check your installation

From the `pids` container, run a first system on a dataset you loaded:

<div class="pm-families" markdown>

=== "Provenance"

    ```shell
    python pidsmaker/main.py velox CADETS_E3
    ```

=== "Network"

    ```shell
    python pidsmaker/main.py euler LANL
    ```

</div>

If the pipeline starts building the graphs without database connection errors, you are all set. Continue with the [Quickstart](quickstart.md).

??? question "Something went wrong?"
    See [FAQ & troubleshooting](../community/faq.md), or ask on [Discord](https://discord.gg/jecmkTABUg).
