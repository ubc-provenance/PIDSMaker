---
title: Quickstart
description: Run the existing provenance-based and network IDSs on the supported datasets, override their arguments and read their results.
---

# Quickstart

This page shows how to run the existing systems, provenance-based (PIDSs) and network (NIDSs), change their arguments from the command line, and find their results.
{ .lead }

!!! note "Before you start"
    Complete the [installation](installation.md). The `pidsmaker` command runs from any folder.

## Your first run

Every run takes the same two positional arguments, a **system** and a **dataset**:

```shell
pidsmaker SYSTEM DATASET [--arg=value ...]
```

From a clone of the repository, `python pidsmaker/main.py SYSTEM DATASET [--arg=value ...]` is equivalent.

`SYSTEM`
:   The name of a YAML file in `config/`, which describes the full pipeline of a system:

    - provenance: `velox`, `orthrus`, `orthrus_non_snooped`, `kairos`, `magic`, `flash`, `nodlink`, `rcaid`, `threatrace`, `ocrapt`;
    - network: `euler`, `argus`, `jbeil`, `lares`, `egraphsage`, `anomal_e`;
    - or [your own](../guide/tutorial.md).

    See [Systems](../guide/systems.md) for a comparison.

`DATASET`
:   A dataset defined in `DATASET_DEFAULT_CONFIG` in `pidsmaker/config/config.py`, which maps it to its database, a folder of Parquet files [downloaded](installation.md#2-download-the-datasets) the first time the dataset is used:

    - provenance: `CADETS_E3`, `THEIA_E3`, `CLEARSCOPE_E3`, `FIVEDIRECTIONS_E3`, `TRACE_E3`, `CADETS_E5`, `THEIA_E5`, `CLEARSCOPE_E5`, `FIVEDIRECTIONS_E5`, `TRACE_E5`, `optc_h201`, `optc_h501`, `optc_h051`, `ATLASV2_EDR`, `CARBANAKV2_EDR`;
    - network: `LANL`, `OPTC_NET`.

    See [Datasets](../guide/datasets.md).

Any system runs on both kinds of datasets, with [a few exceptions](../guide/systems.md#running-across-families).

For example:

<div class="pm-families" markdown>

=== "Provenance"

    Velox on CADETS E3, which looks for the processes, files and netflows of the attacks:

    ```shell
    pidsmaker velox CADETS_E3
    ```

=== "Network"

    Euler on LANL, which looks for the authentications of the red team:

    ```shell
    pidsmaker euler LANL
    ```

</div>

The content of the config (e.g. `config/velox.yml`) is parsed and checked against the arguments available in `TASK_ARGS` (`pidsmaker/config/config.py`), then the [seven stages of the pipeline](../guide/pipeline.md) run one after the other.

## Ways to run

=== "Shell"

    Logs are printed to the terminal. Good for quick tests and debugging.

    ```shell
    pidsmaker SYSTEM DATASET
    ```

=== "Shell + W&B"

    Same, but logs, metrics and figures are also sent to [Weights & Biases](installation.md#3-log-in-to-weights-biases).

    ```shell
    pidsmaker SYSTEM DATASET --wandb
    ```

=== "Background + W&B"

    Recommended for research and for running several experiments in parallel. The run continues after you close the terminal or log out, and logs to W&B. The command prints the log file of the run, in the current folder:

    ```shell
    pidsmaker SYSTEM DATASET --wandb --background
    tail -f pidsmaker_SYSTEM_DATASET_*.log   # follow the logs
    ```

All runs of `pidsmaker` are reproducible: it sets `PYTHONHASHSEED=0` (sets of strings are iterated in the same order) and `CUBLAS_WORKSPACE_CONFIG=:4096:8` (for the deterministic algorithms of CUDA), unless you set them yourself. `python pidsmaker/main.py` does not.

## Change arguments

Every argument of the YAML config can be overridden from the command line, using its dotted path. CLI arguments always take precedence over the YAML file:

<div class="pm-families" markdown>

=== "Provenance"

    ```shell
    pidsmaker kairos CADETS_E3 \
      --training.node_hid_dim=128 \
      --batching.intra_graph_batching.tgn_last_neighbor.tgn_neighbor_size=10
    ```

=== "Network"

    ```shell
    pidsmaker euler LANL \
      --construction.time_window_size=60.0 \
      --training.decoder.predict_edge_contrastive.neg_ratio=5
    ```

</div>

Components can be swapped the same way. For example, Orthrus with doc2vec instead of word2vec, and 3 GraphSAGE layers instead of 2 attention layers:

```shell
pidsmaker orthrus CADETS_E3 \
  --featurization.used_method=doc2vec \
  --featurization.emb_dim=128 \
  --training.encoder.used_methods=tgn,sage \
  --training.encoder.sage.num_layers=3
```

All arguments are listed in the [Reference](../reference/tasks.md). The flags that are not part of the YAML config (`--wandb`, `--cpu`, `--force_restart`...) are listed in [Command line](../reference/cli.md).

!!! tip "Re-running is cheap"
    Each task of the pipeline is cached on disk under a hash of its arguments. When you change `--training.lr`, graph construction and featurization are reused and only training and evaluation run again. See [Pipeline](../guide/pipeline.md#caching-of-tasks).

## GPU or CPU

By default, PIDSMaker runs on the GPU `cuda:0`. If no GPU is found, it falls back to the CPU and prints a warning. To force the CPU:

```shell
pidsmaker velox CADETS_E3 --cpu
```

## Results

At the end of the `evaluation` task, metrics are printed in the logs and, with `--wandb`, sent to W&B with the figures (score distributions, detected attacks...). The main metrics are:

| Metric | Meaning |
|---|---|
| `precision`, `recall`, `fscore`, `mcc` | Classification of nodes (or edges, time windows) as malicious, with the threshold chosen on the validation set |
| `tp`, `fp`, `tn`, `fn` | Confusion matrix |
| `ap`, `auc` | Average precision and ROC AUC of the anomaly scores, independent of the threshold |
| `adp_score` | Attack Detection Precision: area under the curve of the percentage of attacks detected against precision. It rewards systems that detect every attack with few false positives |
| `discrimination` | Margin between the highest score of each attack and the mean of the 10 highest benign scores (normalized scores), averaged over attacks. Positive when attacks stand out |

Provenance systems classify nodes by default, network systems edges (authentications or flows). How samples are labeled, how thresholds are chosen and how each metric is computed is explained in [Evaluation & labeling](../guide/evaluation.md).

All files produced by a run are written in the artifact folder: `$PIDSMAKER_ARTIFACT_DIR` if it is set, else `/home/artifacts/` if this folder exists, else `artifacts/` in the working directory. Use `--artifact_dir` to change it for a run (see [Where files are stored](installation.md#paths)).

## Next steps

<div class="grid cards" markdown>

-   :lucide-workflow:{ .lg .middle } __Understand the pipeline__

    ---

    How the seven stages fit together, and how caching speeds up experiments.

    [:octicons-arrow-right-24: Pipeline](../guide/pipeline.md)

-   :lucide-shield:{ .lg .middle } __Compare the systems__

    ---

    The provenance and network systems side by side, and which datasets each one runs on.

    [:octicons-arrow-right-24: Systems](../guide/systems.md)

-   :lucide-graduation-cap:{ .lg .middle } __Build your own system__

    ---

    Write a new encoder and a new YAML config, and evaluate them.

    [:octicons-arrow-right-24: Tutorial](../guide/tutorial.md)

-   :lucide-flask-conical:{ .lg .middle } __Measure instability__

    ---

    Run each system several times and report the mean and standard deviation.

    [:octicons-arrow-right-24: Instability](../features/instability.md)

-   :lucide-sliders-horizontal:{ .lg .middle } __Tune a system__

    ---

    Grid, random or Bayesian sweeps with W&B.

    [:octicons-arrow-right-24: Hyperparameter tuning](../features/tuning.md)

</div>
