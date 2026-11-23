---
title: Installation (Docker)
description: Install PIDSMaker with Docker and load the preprocessed DARPA TC, OpTC, ATLASv2 and CARBANAKv2 databases.
---

# Installation with Docker

Ten minutes from a fresh machine to your first run. PIDSMaker runs in two containers: a **Postgres** database that stores the provenance data, and a **pids** container with the Python environment where you run experiments.
{ .lead }

!!! info "Requirements"
    - Linux with [Docker](https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository) installed. Avoid running it as root: [manage Docker as a non-root user](https://docs.docker.com/engine/install/linux-postinstall/#manage-docker-as-a-non-root-user).
    - An NVIDIA GPU is recommended but optional: PIDSMaker [falls back to CPU](quickstart.md#gpu-or-cpu).
    - Enough disk space for the datasets you pick (see the table below) and for the artifacts of your runs (a single run can produce 15 GB or more on E3 datasets).
    - No Docker, e.g. on an HPC cluster? Use [Apptainer](apptainer.md) instead.

## 1. Download the datasets

DARPA TC and OpTC are very large and hard to process. We provide **preprocessed versions** as Postgres database dumps.
**Compressed** is the size of the download, **uncompressed** the size once loaded into Postgres.

| Dataset             | Compressed (GB) | Uncompressed (GB) |
|---------------------|----------------:|------------------:|
| `CADETS_E3`         | 1.4             | 10                |
| `THEIA_E3`          | 1.1             | 12                |
| `CLEARSCOPE_E3`     | 0.6             | 4.8               |
| `FIVEDIRECTIONS_E3` | 3.2             | 22                |
| `TRACE_E3`          | 11              | 100               |
| `CADETS_E5`         | 36              | 276               |
| `THEIA_E5`          | 5.8             | 36                |
| `CLEARSCOPE_E5`     | 6.2             | 49                |
| `FIVEDIRECTIONS_E5` | 39              | 280               |
| `TRACE_E5`          | 91              | 710               |
| `optc_h201`         | 2               | 9                 |
| `optc_h501`         | 1.5             | 6.7               |
| `optc_h051`         | 1.7             | 7.7               |
| `ATLASV2_EDR`       | 0.2             | 1                 |
| `CARBANAKV2_EDR`    | 1.1             | 6.6               |

!!! tip "New to PIDSMaker?"
    Start with `CADETS_E3`, `THEIA_E3` or `CLEARSCOPE_E3`: they are small, quick to load, and the most widely used in papers.

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
    ./download_datasets.sh cadets_e3,optc_h201,clearscope_e5 YOUR_ACCESS_TOKEN

    # Or all datasets
    ./download_datasets.sh all YOUR_ACCESS_TOKEN
    ```

    Downloads resume where they stopped if you re-run the command.

!!! note
    You can also build the databases yourself from the official DARPA files: see [Building datasets from scratch](from-scratch.md).

## 2. Enable GPU support in Docker

Skip this step to run on CPU only.

??? example "Install the NVIDIA Container Toolkit"

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

## 6. Log in to Weights & Biases

PIDSMaker uses [Weights & Biases](https://wandb.ai/site/) (W&B) to log, plot and keep the history of experiments. We **highly** recommend it.
Create an account, then log in from the `pids` container with the API key from your W&B dashboard:

```shell
wandb login
```

Runs started with `--wandb`, or with `./run.sh`, then send their logs and results to W&B.

## Check your installation

From the `pids` container, run a first system:

```shell
python pidsmaker/main.py velox CADETS_E3
```

If the pipeline starts building the graphs without database connection errors, you are all set. Continue with the [Quickstart](quickstart.md).

??? question "Something went wrong?"
    See [FAQ & troubleshooting](../community/faq.md), or ask on [Discord](https://discord.gg/jecmkTABUg).
