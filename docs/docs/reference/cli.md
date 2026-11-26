---
title: Command line
description: The pidsmaker command, its flags, the download of datasets, the environment variables of PIDSMaker, and the helper scripts used to run experiments.
---

# Command line

The `pidsmaker` command is the single entry point of the framework: it runs a system on a dataset, and [downloads datasets](#downloading-datasets). Besides the [task arguments](tasks.md), a run accepts the flags below, which control how it is executed rather than what it computes.
{ .lead }

```shell
pidsmaker SYSTEM DATASET [FLAGS] [--task.argument=value ...]
```

From a clone of the repository, `python pidsmaker/main.py SYSTEM DATASET ...` is equivalent.

| Positional | Description |
|---|---|
| `SYSTEM` | Name of a YAML config in `config/`, without the extension (e.g. `velox`, `pretrained_velox`, `euler`). See [Systems](../guide/systems.md). |
| `DATASET` | Name of a dataset in `DATASET_DEFAULT_CONFIG` (e.g. `CADETS_E3`, `optc_h201`, `LANL`, `OPTC_NET`). See [Datasets](../guide/datasets.md). |

Any argument from the [Reference](tasks.md) can be passed as `--task.argument=value`; it overrides the YAML config.

## Pipeline control

| Flag | Default | Description |
|---|---|---|
| `--force_restart TASK` | | Re-runs the pipeline from `TASK` (e.g. `featurization`), overwriting the cached outputs. See [Forcing a restart](../guide/pipeline.md#forcing-a-restart). |
| `--restart_from_scratch` | off | Runs every task in a new folder with a random hash. Its outputs are deleted at the end of the run. |
| `--cpu` | off | Runs on CPU even if a GPU is available. |
| `--save_for_viz` | off | Keeps the per-epoch checkpoints and test graphs needed by the [embedding visualizer](../features/visualization.md). |
| `--save_graph_preprocessing` | off | Saves the preprocessed graphs on disk (heavy). |

## Performance

These flags don't change the results, nor the task hashes. See [Performance](../guide/pipeline.md#performance) and [Disk usage](../guide/pipeline.md#disk-usage).

| Flag | Default | Description |
|---|---|---|
| `--num_workers N` | `min(8, #CPUs)` | Number of processes for CPU-bound preprocessing such as graph construction. |
| `--no_shared_node_features` | off | Stores node features per batch instead of in a shared table rebuilt for each batch. Uses much more memory; for debugging. |
| `--keep_best_epoch_only` | off | At the end of the pipeline, deletes the loss files, evaluation pickles and weights of all epochs except the best one (stats and plots are kept). Re-running evaluation afterwards only considers the kept epoch. |

## Experiment tracking

| Flag | Default | Description |
|---|---|---|
| `--wandb` | off | Sends logs, metrics and figures to Weights & Biases. Always on with `./run.sh`. |
| `--project NAME` | `PIDSMaker` | W&B project of the run. |
| `--exp NAME` | | Name of the experiment, to group and find runs in W&B. |
| `--tags TAGS` | | W&B tags of the run. |

## Experiments and tuning

| Flag | Default | Description |
|---|---|---|
| `--experiment NAME` | `none` | Runs an experiment defined in `config/experiments/uncertainty/`, e.g. `run_n_times` to [measure instability](../features/instability.md). |
| `--tuning_mode MODE` | `none` | `hyperparameters` starts a W&B sweep. See [Hyperparameter tuning](../features/tuning.md). |
| `--tuning_file_path PATH` | | Sweep config to use instead of the dataset-specific one, relative to `config/experiments/tuning/` (e.g. `systems/default/tuning_my_system`). |
| `--sweep_id ID` | | Joins an existing W&B sweep, to run it with several agents. |
| `--tuned` | off | Loads the tuned hyperparameters of the system from `config/tuned_baselines/<dataset>/tuned_<system>.yml` (e.g. LARES and Argus on `OPTC_NET`). |

## Datasets and storage

| Flag | Default | Description |
|---|---|---|
| `--datasets_dir PATH` | `$PIDSMAKER_DATASETS_DIR`, else `~/.cache/pidsmaker/datasets` | Folder of the Parquet datasets, one subfolder per dataset. |
| `--artifact_dir PATH` | `$PIDSMAKER_ARTIFACT_DIR`, else `/home/artifacts/` if it exists, else `./artifacts/` | Folder where all task outputs are written. |
| `--database_backend BACKEND` | `$PIDSMAKER_DATABASE_BACKEND`, else `parquet` | Where datasets are read from: `parquet` (Parquet files read by DuckDB) or `postgres` (a Postgres server, the [legacy backend](../getting-started/installation.md#postgres)). Results are the same on both. |
| `--database_host HOST` | `postgres` | Postgres backend only: host of the server. The default is the Postgres container, seen from the `pids` container; use `localhost` for a server on the same machine. |
| `--database_port PORT` | `5432` | Postgres backend only: port of the server. |
| `--database_user USER` | `postgres` | Postgres backend only: user. |
| `--database_password PASSWORD` | `postgres` | Postgres backend only: password of the user. |

## Downloading datasets

`pidsmaker download` downloads datasets from Hugging Face (see [Download the datasets](../getting-started/installation.md#2-download-the-datasets)):

```shell
pidsmaker download DATASET [DATASET ...]   # or `all`
pidsmaker download --list                  # the datasets, their sizes, and which are downloaded
```

Dataset names are case-insensitive, and the names of their databases work too (e.g. `optc_051` for `optc_h051`). Datasets already downloaded are skipped, and interrupted downloads resume where they stopped.

| Flag | Default | Description |
|---|---|---|
| `--list` | | Lists the datasets, their sizes, and which are downloaded. |
| `--datasets_dir PATH` | `$PIDSMAKER_DATASETS_DIR`, else `~/.cache/pidsmaker/datasets` | Where datasets are stored. |
| `--repo REPO` | `$PIDSMAKER_HF_REPO`, else `tbilot/pidsmaker` | Hugging Face dataset repository. |
| `--revision REVISION` | | Branch, tag or commit of the repository. |
| `--force` | off | Downloads the datasets again, even if they are already downloaded. |

The same from Python: `from pidsmaker.datasets import download; download("CADETS_E3")`.

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `PIDSMAKER_DATASETS_DIR` | `~/.cache/pidsmaker/datasets` | Folder of the datasets (default of `--datasets_dir`). |
| `PIDSMAKER_ARTIFACT_DIR` | `/home/artifacts/` if it exists, else `./artifacts/` | Folder of the artifacts (default of `--artifact_dir`). |
| `PIDSMAKER_AUTO_DOWNLOAD` | `1` | A run downloads its dataset if it is missing. With `0`, it stops with an error instead. |
| `PIDSMAKER_HF_REPO` | `tbilot/pidsmaker` | Hugging Face dataset repository of the datasets (default of `--repo`). |
| `PIDSMAKER_DUCKDB_THREADS` | `8` | Threads of each DuckDB connection. Graph construction opens several connections at the same time, one per day. |
| `PIDSMAKER_DATABASE_BACKEND` | `parquet` | `postgres` selects the [legacy Postgres backend](../getting-started/installation.md#postgres) (default of `--database_backend`). |
| `OMP_NUM_THREADS` | | CPU threads of PyTorch. If it is not set, PIDSMaker sets it to 4 in Docker Desktop on Apple Silicon (see [macOS](../getting-started/installation.md#docker-desktop)). |

## Helper scripts

The scripts in `scripts/` wrap `pidsmaker/main.py`. Run them from the `scripts/` folder of a clone of the repository. `run.sh` and `run_local.sh` also set deterministic CUDA and Python hashing settings (`CUBLAS_WORKSPACE_CONFIG=:4096:8`, `PYTHONHASHSEED=0`).

| Script | Description |
|---|---|
| `./run.sh SYSTEM DATASET [args]` | Runs in the background with `nohup` and `--wandb`. Logs go to `scripts/nohup.out`. |
| `./run_local.sh SYSTEM DATASET [args]` | Same settings, in the foreground and without W&B. |
| `./run_all_datasets.sh SYSTEM [args]` | Runs a system on CLEARSCOPE, CADETS and THEIA (E3 and E5) one after the other, in the background, with W&B. |
| `./run_serial.sh SYSTEM DATASET [args]` | Runs in the foreground with `--wandb` and `--restart_from_scratch`. Used by `run_all_datasets.sh`. |
