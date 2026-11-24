---
title: Tasks
description: All configuration arguments of each task of the PIDSMaker pipeline, generated from pidsmaker/config/config.py.
---

# Tasks

Every argument of each [pipeline](../guide/pipeline.md) task, as accepted in a system's YAML config or on the command line (e.g. `--training.lr=0.001`).
{ .lead }

!!! tip "How to read this page"
    - Keys are nested exactly as in the YAML config. On the CLI, join them with dots: `training.encoder.dropout` → `--training.encoder.dropout=0.3`.
    - Click the :material-plus-circle: markers to see a description and the allowed values of an argument.
    - Arguments of interchangeable methods (featurization methods, encoders, decoders, objectives, cleaning) are documented on their own pages: [Featurization](featurization.md), [Encoders](encoders.md), [Decoders](decoders.md), [Objectives](objectives.md), [Cleaning](cleaning.md).

!!! info "Generated content"
    The argument lists below are generated from `TASK_ARGS` in `pidsmaker/config/config.py` by `docs/scripts/get_arguments.py`. Don't edit them by hand.

## Stage 1: Construction

Builds the time-window graphs of provenance or network events from the database. The raw data can be cleaned before the graphs are built with `construction.cleaning`: see [Cleaning](cleaning.md).

--8<-- "scripts/args/args_construction.md"

## Stage 2: Transformation

Optional transformations applied to the graphs after construction.

--8<-- "scripts/args/args_transformation.md"

## Stage 3: Featurization

Trains the model that embeds the textual labels of entities. The methods and their arguments are listed in [Featurization](featurization.md).

--8<-- "scripts/args/args_featurization.md"

### Feature inference

`feat_inference` runs right after `featurization` and embeds every node of the graphs with the trained model. Its arguments mostly apply to [pretrained encoders](../features/pretrained_encoders.md): continued pretraining on the target dataset, and the [renaming attack](../features/pretrained_encoders.md#renaming-attack).

--8<-- "scripts/args/args_feat_inference.md"

## Stage 4: Batching

Selects the node and edge features and builds the batches given to the model. See [Batching & sampling](../features/batching.md).

--8<-- "scripts/args/args_batching.md"

## Stage 5: Training

Trains the encoder and the objectives, then computes the losses on the validation and test sets. Encoders, decoders and objectives have their own pages.

--8<-- "scripts/args/args_training.md"

## Stage 6: Evaluation

Turns the losses into anomaly scores, chooses a threshold and computes the metrics.

--8<-- "scripts/args/args_evaluation.md"

## Stage 7: Triage

Optional post-processing of the detections.

--8<-- "scripts/args/args_triage.md"

## Experiments

Arguments of the experiment modes, such as running a system several times to [measure instability](../features/instability.md). They are usually set in a YAML file under `config/experiments/` and selected with `--experiment`.

--8<-- "scripts/args/args_experiments.md"
