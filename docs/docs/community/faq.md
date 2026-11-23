---
title: FAQ & troubleshooting
description: Answers to common questions and fixes for common problems when installing and running PIDSMaker.
---

# FAQ & troubleshooting

Answers to the questions that come up most often. Can't find yours? Ask on [Discord](https://discord.gg/jecmkTABUg) or open an [issue](https://github.com/ubc-provenance/PIDSMaker/issues).
{ .lead }

## Installation

??? question "Which datasets should I start with?"
    `CADETS_E3`, `THEIA_E3` and `CLEARSCOPE_E3`. They are small (5 to 12 GB once loaded), quick to process, and the most used in papers. E5 datasets and `TRACE_E3` are much larger: see the sizes in the [installation guide](../getting-started/installation.md#1-download-the-datasets).

??? question "A dataset download stopped halfway."
    Re-run the same `./download_datasets.sh` command: downloads resume where they stopped. With `gdown`, add `--continue` to the command.

??? question "I changed `.env` but nothing changed."
    Run `source .env` before `docker compose`, then recreate the containers with `docker compose ... up -d --build`.

??? question "The run can't connect to the database."
    - Check that the Postgres container is running: `docker compose -p postgres ps`.
    - Inside Docker, the database host is `postgres` (the default). Outside Docker, e.g. with [Apptainer](../getting-started/apptainer.md), pass `--database_host localhost`.
    - Check that the dataset's database was loaded: from the Postgres container, `psql -U postgres -l` lists the databases.

??? question "Can I use PIDSMaker without Docker?"
    Yes, with [Apptainer and conda](../getting-started/apptainer.md), which is convenient on HPC clusters.

## Running experiments

??? question "I don't have a GPU."
    PIDSMaker falls back to the CPU automatically, with a warning. You can also force it with `--cpu`. Expect much longer training times on large datasets.

??? question "I changed the code, but the task is skipped."
    Tasks are cached under a hash of their **arguments**, not of the code. After changing the code of a task, re-run it with `--force_restart=TASK` (e.g. `--force_restart=training`). See [Forcing a restart](../guide/pipeline.md#forcing-a-restart).

??? question "Where are the outputs of my runs?"
    In the artifact folder: `/home/artifacts/` in the container by default, mapped to `ARTIFACTS_DIR` from `.env` on the host. Use `--artifact_dir` to change it for a run.

??? question "My disk is full."
    Each run caches the outputs of every task, which can exceed 15 GB per run on E3 datasets. Point `ARTIFACTS_DIR` to a large disk, and delete the folders of old experiments. Runs started with `--restart_from_scratch` delete their outputs automatically at the end.

??? question "Can I run several experiments in parallel?"
    Yes. With simple architectures, 3 to 4 runs fit on a single A100 GPU without major overhead. Use `./run.sh` from the `scripts/` folder to run them in the background. Avoid starting several runs **with the same early tasks** at the same time: they would write to the same folder. See [Caching of tasks](../guide/pipeline.md#caching-of-tasks).

??? question "The GPU runs out of memory."
    Reduce the size of the graphs or batches (e.g. `construction.time_window_size`, or the batching arguments in [Batching & sampling](../features/batching.md)), or the embedding sizes (`training.node_hid_dim`, `training.node_out_dim`).

??? question "I don't want to use Weights & Biases."
    Run `python pidsmaker/main.py SYSTEM DATASET` without `--wandb`, or `./run_local.sh SYSTEM DATASET` from `scripts/`. Metrics are printed in the logs.

## Results

??? question "I get very different results from one run to the next, or from the papers."
    This is expected: PIDSs are **unstable**, and small changes of seed or hyperparameters change their results a lot. Run each configuration several times with `--experiment=run_n_times` and report the mean and standard deviation; see [Instability](../features/instability.md). Also check that you use the [tuned hyperparameters](../guide/tuned-systems.md) and the same [ground truth](../guide/datasets.md#ground-truth) as the paper you compare with.

??? question "Which configuration should I use to compare with an existing system?"
    The non-snooped variants, such as `orthrus_non_snooped`: some original configs use test data during training. See [Systems](../guide/systems.md#variants).

## Features

??? question "`predict_edge_supervised` complains that attack embeddings are missing."
    The attack labels are embedded by the `feat_inference` task. If it was already run with the same settings (e.g. by `pretrained_velox`), it is skipped: add `--force_restart=feat_inference`. See [Supervised fine-tuning](../features/supervised.md).

??? question "Llama pretrained encoders fail to download."
    Llama models are gated on Hugging Face: accept the license on the model page, then run `huggingface-cli login` in the container.

??? question "The embedding visualizer page doesn't load on `localhost`."
    If `127.0.0.1` works but `localhost` doesn't, the server listens on IPv4 only while `localhost` resolves to IPv6. Start it with `--host ::`. See [Embedding visualizer](../features/visualization.md).

## Community

??? question "How do I cite PIDSMaker?"
    See [Citing PIDSMaker](../index.md#citing-pidsmaker) on the home page.

??? question "How can I add my system to PIDSMaker?"
    Follow the [tutorial](../guide/tutorial.md), then the [contribution guide](contributing.md). We are happy to help on [Discord](https://discord.gg/jecmkTABUg).
