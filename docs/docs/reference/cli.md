---
title: Command line
description: The command-line flags of PIDSMaker's main.py, and the helper scripts used to run experiments.
---

# Command line

`pidsmaker/main.py` is the single entry point of the framework. Besides the [task arguments](tasks.md), it accepts the flags below, which control how a run is executed rather than what it computes.
{ .lead }

```shell
python pidsmaker/main.py SYSTEM DATASET [FLAGS] [--task.argument=value ...]
```

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

## Database and storage

| Flag | Default | Description |
|---|---|---|
| `--database_host HOST` | `postgres` | Host of the Postgres server. Use `localhost` outside Docker. |
| `--database_port PORT` | `5432` | Port of the Postgres server. |
| `--database_user USER` | `postgres` | Postgres user. |
| `--database_password PASSWORD` | `postgres` | Password of the Postgres user. |
| `--artifact_dir PATH` | `/home/artifacts/` | Folder where all task outputs are written. |

## Helper scripts

The scripts in `scripts/` wrap `main.py`. Run them from the `scripts/` folder. `run.sh` and `run_local.sh` also set deterministic CUDA and Python hashing settings (`CUBLAS_WORKSPACE_CONFIG=:4096:8`, `PYTHONHASHSEED=0`).

| Script | Description |
|---|---|
| `./run.sh SYSTEM DATASET [args]` | Runs in the background with `nohup` and `--wandb`. Logs go to `scripts/nohup.out`. |
| `./run_local.sh SYSTEM DATASET [args]` | Same settings, in the foreground and without W&B. |
| `./run_all_datasets.sh SYSTEM [args]` | Runs a system on CLEARSCOPE, CADETS and THEIA (E3 and E5) one after the other, in the background, with W&B. |
| `./run_serial.sh SYSTEM DATASET [args]` | Runs in the foreground with `--wandb` and `--restart_from_scratch`. Used by `run_all_datasets.sh`. |
