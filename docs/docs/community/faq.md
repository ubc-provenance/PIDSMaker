---
title: FAQ & troubleshooting
description: Answers to common questions and fixes for common problems when installing and running PIDSMaker.
---

# FAQ & troubleshooting

Answers to the questions that come up most often. Can't find yours? Ask on [Discord](https://discord.gg/jecmkTABUg) or open an [issue](https://github.com/ubc-provenance/PIDSMaker/issues).
{ .lead }

## Installation

??? question "Do I need Docker or Postgres?"
    No. PIDSMaker [installs with pip](../getting-started/installation.md), and reads its datasets from Parquet files with DuckDB, in the Python process. [Docker](../getting-started/docker.md) is optional, and a Postgres server is only needed by the [legacy Postgres backend](../getting-started/installation.md#postgres).

??? question "pip refuses to install PIDSMaker, or can't find `torch==2.1.2`."
    Check the version of Python: PIDSMaker needs Python 3.9 to 3.11, as PyTorch 2.1.2 has no wheels for Python 3.12 and later. Create the environment with `conda create -n pids python=3.11`.

??? question "Which datasets should I start with?"
    For provenance-based detection, `CADETS_E3`, `THEIA_E3` and `CLEARSCOPE_E3`. They are small (0.5 to 1.3 GB), quick to process, and the most used in papers. For network detection, `LANL`. E5 datasets and `TRACE_E3` are much larger: see the sizes in the [installation guide](../getting-started/installation.md#2-download-the-datasets).

??? question "A dataset download stopped halfway."
    Run the same `pidsmaker download` command again: downloads resume where they stopped. A run whose dataset is incomplete also downloads the missing files.

??? question "A run doesn't find a dataset I downloaded, or downloads it again."
    The run looks for datasets in another folder than the one you downloaded them to. Both use `--datasets_dir`, else `$PIDSMAKER_DATASETS_DIR`, else `~/.cache/pidsmaker/datasets`: set `PIDSMAKER_DATASETS_DIR` once, e.g. in your `~/.bashrc`, rather than passing `--datasets_dir` to some commands only. See [Where files are stored](../getting-started/installation.md#paths).

??? question "With the legacy Postgres backend, the run can't connect to the database."
    - Check that the run uses this backend: `--database_backend postgres`, or `PIDSMAKER_DATABASE_BACKEND=postgres`.
    - Check that the Postgres container is running: `docker compose -p postgres ps`.
    - From the `pids` container, the database host is `postgres` (the default). From the host, or with [Apptainer](../getting-started/apptainer.md#legacy-postgres-with-apptainer), pass `--database_host localhost` (and `--database_port`, if Postgres does not listen on 5432).
    - Check that the dataset's database was loaded: from the Postgres container, `psql -U postgres -l` lists the databases.

??? question "I changed `.env` but nothing changed (Docker)."
    Run `source .env` before `docker compose`, then recreate the containers with `docker compose ... up -d --build`.

??? question "On macOS, `docker compose` fails with `operation not permitted`."
    Docker Desktop is not allowed to read the folder of the repository (e.g. `~/Documents`). Allow it in **System Settings > Privacy & Security > Files and Folders** (or **Full Disk Access**), then restart Docker Desktop. See [Set up Docker](../getting-started/docker.md#1-set-up-docker).

??? question "`docker compose up` fails with `network shared_network declared as external, but could not be found`."
    The `pids` container joins the Docker network `shared_network`. Create it once with `docker network create shared_network`.

??? question "`docker compose up` fails with `address already in use` on port 5000 or 8000."
    Another program uses this port on the host, e.g. AirPlay Receiver on macOS for port 5000. Set other host ports in `.env`, e.g. `VIZ_PORT=5001` and `PIDS_PORT=8001`, then `source .env` and run `up` again. Without Docker, start the [embedding visualizer](../features/visualization.md) on another port with `--port`.

??? question "`docker compose up` fails with `could not select device driver` (nvidia)."
    The machine has no NVIDIA GPU (or no NVIDIA Container Toolkit). Add the CPU override: `docker compose -f compose-pidsmaker.yml -f compose-pidsmaker.cpu.yml up -d --build`.

??? question "A run is killed (`Killed`, exit code 137) on a laptop or with Docker Desktop."
    The run ran out of memory. On macOS, Docker Desktop limits its virtual machine to 8 GB by default: raise it as explained in [macOS](../getting-started/installation.md#docker-desktop). Graph construction also uses less memory with fewer processes, e.g. `--num_workers=2`.

??? question "Training is very slow on CPU in Docker on a Mac."
    With Docker Desktop on Apple Silicon, PIDSMaker limits PyTorch to 4 threads (`OMP_NUM_THREADS=4`), as more threads make operations on small tensors up to 100 times slower in the VM. If you set `OMP_NUM_THREADS` yourself (e.g. `docker compose exec -e OMP_NUM_THREADS=6 pids bash`), keep it at most at the number of performance cores of your Mac.

## Running experiments

??? question "I don't have a GPU."
    PIDSMaker falls back to the CPU automatically, with a warning. You can also force it with `--cpu`. Expect much longer training times on large datasets.

??? question "A system fails on a dataset of the other family."
    - Flash and ThreaTrace predict node types (`predict_node_type`), which needs at least two node types, and R-Caid predicts the executables of processes (`predict_process_tag`): they can't run on network datasets, which only have hosts.
    - Argus, LARES, E-GraphSAGE and Anomal-E use the attributes of events, which provenance datasets don't have: add `--batching.edge_features=edge_type`.

    See [Running across families](../guide/systems.md#running-across-families).

??? question "I changed the code, but the task is skipped."
    Tasks are cached under a hash of their **arguments**, not of the code. After changing the code of a task, re-run it with `--force_restart=TASK` (e.g. `--force_restart=training`). See [Forcing a restart](../guide/pipeline.md#forcing-a-restart).

??? question "Where are the outputs of my runs?"
    In the artifact folder: `$PIDSMAKER_ARTIFACT_DIR` if it is set, else `/home/artifacts/` if this folder exists, else `artifacts/` in the folder where you started the run. In the Docker container, `/home/artifacts/` is `ARTIFACTS_DIR` from `.env` on the host. Use `--artifact_dir` to change it for a run. See [Where files are stored](../getting-started/installation.md#paths).

??? question "My disk is full."
    Each run caches the outputs of every task, which can exceed 15 GB per run on E3 datasets. Point `PIDSMAKER_ARTIFACT_DIR` (with Docker, `ARTIFACTS_DIR` in `.env`) to a large disk, and delete the folders of old experiments. Runs started with `--restart_from_scratch` delete their outputs automatically at the end.

??? question "Can I run several experiments in parallel?"
    Yes. With simple architectures, 3 to 4 runs fit on a single A100 GPU without major overhead. Use `./run.sh` from the `scripts/` folder to run them in the background. Avoid starting several runs **with the same early tasks** at the same time: they would write to the same folder. See [Caching of tasks](../guide/pipeline.md#caching-of-tasks).

??? question "The GPU runs out of memory."
    Reduce the size of the graphs or batches (e.g. `construction.time_window_size`, or the batching arguments in [Batching & sampling](../features/batching.md)), or the embedding sizes (`training.node_hid_dim`, `training.node_out_dim`).

??? question "I don't want to use Weights & Biases."
    Run `pidsmaker SYSTEM DATASET` without `--wandb`, or `./run_local.sh SYSTEM DATASET` from `scripts/`. Metrics are printed in the logs.

## Results

??? question "I get very different results from one run to the next, or from the papers."
    This is expected: self-supervised detectors, PIDSs and NIDSs alike, are **unstable**, and small changes of seed or hyperparameters change their results a lot. Run each configuration several times with `--experiment=run_n_times` and report the mean and standard deviation; see [Instability](../features/instability.md). Also check that you use the same [ground truth](../guide/evaluation.md#ground-truth) as the paper you compare with.

??? question "Which configuration should I use to compare with an existing system?"
    The non-snooped variants, such as `orthrus_non_snooped`: some original configs use test data during training. See [Systems](../guide/systems.md#variants). For network systems, also read [Comparing NIDSs with published results](../guide/systems.md#results-and-reproducibility).

## Features

??? question "`predict_edge_supervised` complains that attack embeddings are missing."
    The attack labels are embedded by the `feat_inference` task. If it was already run with the same settings (e.g. by `pretrained_velox`), it is skipped: add `--force_restart=feat_inference`. See [Supervised fine-tuning](../features/supervised.md).

??? question "Llama pretrained encoders fail to download."
    Llama models are gated on Hugging Face: accept the license on the model page, then run `huggingface-cli login` where PIDSMaker runs.

??? question "The embedding visualizer page doesn't load on `localhost`."
    If `127.0.0.1` works but `localhost` doesn't, the server listens on IPv4 only while `localhost` resolves to IPv6. Start it with `--host ::`. See [Embedding visualizer](../features/visualization.md).

## Community

??? question "How do I cite PIDSMaker?"
    See [Citing PIDSMaker](../index.md#citing-pidsmaker) on the home page.

??? question "How can I add my system to PIDSMaker?"
    Follow the [tutorial](../guide/tutorial.md), then the [contribution guide](contributing.md). We are happy to help on [Discord](https://discord.gg/jecmkTABUg).
