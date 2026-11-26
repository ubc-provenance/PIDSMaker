---
title: Installation
description: Install PIDSMaker with pip, and download the preprocessed provenance datasets (DARPA TC, OpTC, ATLASv2, CARBANAKv2) and network datasets (LANL, OpTC flows) from Hugging Face.
---

# Installation

A few minutes from a fresh machine to your first run. PIDSMaker is a Python package: install it with pip, download a dataset, and run a system. Datasets are Parquet files that PIDSMaker reads in place with [DuckDB](https://duckdb.org/): nothing else to set up.
{ .lead }

!!! info "Requirements"
    - :fontawesome-brands-linux: **Linux**, ideally with an :simple-nvidia: **NVIDIA GPU**, for all datasets. Without a GPU, PIDSMaker [falls back to CPU](quickstart.md#gpu-or-cpu).
    - :fontawesome-brands-apple: **macOS** (Apple Silicon), on CPU, for most datasets: 32 GB of RAM is ideal. See [macOS](#macos).
    - **Python 3.9 to 3.11**: PyTorch 2.1.2, which PIDSMaker pins, has no wheels for Python 3.12 and later. We create the environment with [conda](https://docs.conda.io/en/latest/miniconda.html).
    - Disk space for the datasets you pick (see the [table below](#2-download-the-datasets)) and for the artifacts of your runs (a single run can produce 15 GB or more on E3 datasets).

    On an HPC cluster, see [HPC clusters](#hpc-clusters).

## 1. Install PIDSMaker

=== "From PyPI"

    ```shell
    conda create -n pids python=3.11
    conda activate pids
    pip install pidsmaker
    ```

=== "From source"

    To change the code or the configs of PIDSMaker, e.g. to [add a system](../guide/tutorial.md), install it from a clone of the repository, in editable mode: your changes are used without reinstalling.

    ```shell
    git clone https://github.com/ubc-provenance/PIDSMaker.git && cd PIDSMaker
    conda create -n pids python=3.11
    conda activate pids
    pip install -e .
    ```

This installs the `pidsmaker` command, with the configs of all systems and the ground truth of all datasets. Check it with `pidsmaker --help`.

!!! note "PyTorch with CUDA, or CPU only"
    PIDSMaker pins `torch==2.1.2`. On Linux, pip installs it from PyPI with CUDA 12.1, which runs on an NVIDIA GPU when there is one, and on CPU otherwise. On a machine without a GPU, you can install the much smaller CPU-only build of PyTorch first:

    ```shell
    pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cpu
    pip install pidsmaker
    ```

??? tip "Optional: compiled extensions of PyTorch Geometric"
    PIDSMaker does not need the compiled extensions of PyTorch Geometric (`pyg_lib`, `torch_scatter`, `torch_sparse`...): without them, it falls back to pure PyTorch, with the same results. With an NVIDIA GPU, you can install them for CUDA 12.1:

    ```shell
    pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
    ```

Optional dependencies are installed as extras, e.g. `pip install "pidsmaker[viz]"`, or `pip install -e ".[viz,test]"` from source:

| Extra | Installs | For |
|---|---|---|
| `viz` | `umap-learn` | The UMAP projections of the [embedding visualizer](../features/visualization.md) |
| `test` | `pytest`, `pytest-xdist`... | The [tests](../community/contributing.md#before-opening-a-pull-request) |

<span id="1-download-the-datasets"></span>
<span id="download-the-dumps"></span>
<span id="4-load-the-databases"></span>

## 2. Download the datasets

DARPA TC and OpTC are very large and hard to process: we provide **preprocessed versions** of all the [datasets](../guide/datasets.md), as Parquet files in the Hugging Face repository [`pidsmaker/datasets`](https://huggingface.co/datasets/pidsmaker/datasets). The network datasets are built from the preprocessed files of [LARES](https://github.com/TristanBilot/lares).

| Dataset | Family | Size (GB) |
|---|---|--:|
| `CADETS_E3` | <span class="pm-family pm-family--prov">Provenance</span> | 1.3 |
| `THEIA_E3` | <span class="pm-family pm-family--prov">Provenance</span> | 0.71 |
| `CLEARSCOPE_E3` | <span class="pm-family pm-family--prov">Provenance</span> | 0.51 |
| `FIVEDIRECTIONS_E3` | <span class="pm-family pm-family--prov">Provenance</span> | 2.3 |
| `TRACE_E3` | <span class="pm-family pm-family--prov">Provenance</span> | 9.4 |
| `CADETS_E5` | <span class="pm-family pm-family--prov">Provenance</span> | 32.9 |
| `THEIA_E5` | <span class="pm-family pm-family--prov">Provenance</span> | 3.9 |
| `CLEARSCOPE_E5` | <span class="pm-family pm-family--prov">Provenance</span> | 5.0 |
| `FIVEDIRECTIONS_E5` | <span class="pm-family pm-family--prov">Provenance</span> | 30.3 |
| `TRACE_E5` | <span class="pm-family pm-family--prov">Provenance</span> | 74.3 |
| `optc_h201` | <span class="pm-family pm-family--prov">Provenance</span> | 1.6 |
| `optc_h501` | <span class="pm-family pm-family--prov">Provenance</span> | 1.2 |
| `optc_h051` | <span class="pm-family pm-family--prov">Provenance</span> | 1.3 |
| `ATLASV2_EDR` | <span class="pm-family pm-family--prov">Provenance</span> | 0.16 |
| `CARBANAKV2_EDR` | <span class="pm-family pm-family--prov">Provenance</span> | 1.1 |
| `LANL` | <span class="pm-family pm-family--net">Network</span> | 0.40 |
| `OPTC_NET` | <span class="pm-family pm-family--net">Network</span> | 11.8 |

Datasets keep this size when they are used: DuckDB reads the files in place, nothing is loaded into a database. (PIDSMaker 4.x read its datasets from a Postgres server, with Docker: it stays available at the tag [`4.0.0`](https://github.com/ubc-provenance/PIDSMaker/releases/tag/4.0.0).) `PROVENANCE_BENIGN` (2 MB), the corpus used to [pretrain SPIDER](../features/pretrained_encoders.md#pretraining), is downloaded the same way.

!!! tip "New to PIDSMaker?"
    Start with `CADETS_E3`, `THEIA_E3` or `CLEARSCOPE_E3` for provenance-based detection, and `LANL` for network detection: they are small, quick to download, and the most widely used in papers.

Download one or more datasets by name (case-insensitive):

```shell
pidsmaker download CADETS_E3 LANL   # some datasets
pidsmaker download all              # all datasets (about 180 GB)
pidsmaker download --list           # the datasets, their sizes, and which are downloaded
```

Datasets are stored in `~/.cache/pidsmaker/datasets/`, one folder per dataset. To store them elsewhere, e.g. on a larger disk, set `PIDSMAKER_DATASETS_DIR` (e.g. in your `~/.bashrc`), or pass `--datasets_dir` to both `pidsmaker download` and your runs. Interrupted downloads resume where they stopped when you run the command again, and datasets already downloaded are skipped (`--force` downloads them again). The other options are listed in the [command line reference](../reference/cli.md#downloading-datasets).

You can also skip this step: a run downloads its dataset the first time it needs it (set `PIDSMAKER_AUTO_DOWNLOAD=0` to get an error instead). Datasets can also be downloaded from Python:

```python
from pidsmaker.datasets import download

download("CADETS_E3", "LANL")  # returns the folders of the datasets
```

To build the datasets yourself, from the official DARPA files or the preprocessed files of LARES, see [Building datasets from scratch](from-scratch.md).

<span id="6-log-in-to-weights-biases"></span>

## 3. Log in to Weights & Biases

PIDSMaker uses [Weights & Biases](https://wandb.ai/site/) (W&B) to log, plot and keep the history of experiments. We **highly** recommend it, but it is optional.
Create an account, then log in with the API key from your W&B dashboard:

```shell
wandb login
```

Runs started with `--wandb` then send their logs and results to W&B.

## Check your installation

Run a first system on a dataset (downloaded first if needed):

<div class="pm-families" markdown>

=== "Provenance"

    ```shell
    pidsmaker velox CADETS_E3
    ```

=== "Network"

    ```shell
    pidsmaker euler LANL
    ```

</div>

If the pipeline starts building the graphs, you are all set. From a clone of the repository, `python pidsmaker/main.py velox CADETS_E3` does the same. Continue with the [Quickstart](quickstart.md).

??? question "Something went wrong?"
    See [FAQ & troubleshooting](../community/faq.md), or ask on [Discord](https://discord.gg/jecmkTABUg).

## Where files are stored { #paths }

| Files | Default | Change it with |
|---|---|---|
| Datasets | `~/.cache/pidsmaker/datasets/` | `--datasets_dir`, or `$PIDSMAKER_DATASETS_DIR` |
| Artifacts: everything a run produces (graphs, models, results...) | `/home/artifacts/` if this folder exists, else `artifacts/` in the folder where you run `pidsmaker` | `--artifact_dir`, or `$PIDSMAKER_ARTIFACT_DIR` |

Artifacts take much more space than datasets: point them to a large disk. Set `PIDSMAKER_ARTIFACT_DIR` rather than relying on the working directory, so that all your runs share the [cache of their tasks](../guide/pipeline.md#caching-of-tasks). All the environment variables are listed in the [command line reference](../reference/cli.md#environment-variables).

## macOS

PyTorch 2.1.2 has wheels for Apple Silicon, so PIDSMaker installs [as above](#1-install-pidsmaker). It runs on CPU: PIDSMaker does not use the GPU of Macs.

- **Memory.** Runs on provenance datasets can take more than 15 GB of RAM, and the largest datasets (`TRACE_E3` and the E5 datasets) need more memory and disk than a laptop has. PIDSMaker prints a warning at the beginning of a run when less than 16 GB are available. Graph construction uses less memory with fewer processes, e.g. `--num_workers=2`.
- **Port 5000** is used by AirPlay Receiver. Start the [embedding visualizer](../features/visualization.md) on another port (`--port 5001`), or turn AirPlay Receiver off in **System Settings > General > AirDrop & Handoff**.

## HPC clusters { #hpc-clusters }

PIDSMaker needs nothing more on a cluster than on any Linux machine. On a login node, install it as [above](#1-install-pidsmaker) (load your cluster's conda module first if it has one). Home directories are often small: store the datasets and the artifacts on a large filesystem, e.g. your project or scratch space, in your `~/.bashrc` or job scripts:

```shell
export PIDSMAKER_DATASETS_DIR=/path/to/large/disk/pidsmaker/datasets
export PIDSMAKER_ARTIFACT_DIR=/path/to/large/disk/pidsmaker/artifacts
```

Compute nodes often have no internet access. From a login node, download the datasets, and the NLTK tokenizer data that PIDSMaker otherwise downloads the first time it is imported:

```shell
pidsmaker download CADETS_E3 THEIA_E3
python -m nltk.downloader punkt
```

In your jobs, activate the environment, set `PIDSMAKER_AUTO_DOWNLOAD=0` (a missing dataset then stops the run with an error, instead of a download attempt), and run `pidsmaker SYSTEM DATASET`.
