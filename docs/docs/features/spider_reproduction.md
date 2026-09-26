# Reproducing SPIDER

These commands reproduce the SPIDER results of the NeurIPS 2026 paper on CADETS_E3, THEIA_E3, THEIA_E5, CLEARSCOPE_E5 and optc_h051, with the configs of this version of PIDSMaker.
They set the same hyperparameters as the paper, but results can differ slightly because of newer library versions (see [Differences with the paper](#differences-with-the-paper)).
The exact code of the paper is on the [`spider` branch](https://github.com/ubc-provenance/PIDSMaker/tree/spider).

## Setup

Download the [pretrained weights](pretrained_encoders.md#pretrained-weights) and export their absolute path as `SPIDER_WEIGHTS`, then run the commands from `scripts/`:

```shell
cd scripts
```

`./run.sh` runs each command in the background and logs to [Weights & Biases](https://wandb.ai) (run `wandb login` first), so launch them in batches that fit on your GPUs.
`--experiment=run_n_times` trains each detector with 5 seeds and logs the mean and standard deviation of every metric to the W&B project given by `--project` (see [Instability Measurement](instability.md)).
To run in the foreground without W&B, use `./run_local.sh`: the metrics of each seed are then saved to `method_to_metrics.pkl` in the run's evaluation artifacts.

## Frozen encoder

Each `pretrained_<pids>` config replaces the node features of the PIDS with embeddings from the frozen SPIDER encoder.

```shell
# CADETS_E3
./run.sh pretrained_velox CADETS_E3 --training.num_epochs=22 --project=spider_frozen --experiment=run_n_times --featurization.pretrained.weights_path=$SPIDER_WEIGHTS
./run.sh pretrained_orthrus CADETS_E3 --training.encoder.dropout=0.3 --training.lr=0.0001 --training.node_hid_dim=256 --training.node_out_dim=256 --training.num_epochs=20 --project=spider_frozen --experiment=run_n_times --featurization.pretrained.weights_path=$SPIDER_WEIGHTS
./run.sh pretrained_nodlink CADETS_E3 --training.encoder.dropout=0.3 --training.lr=0.0001 --training.node_hid_dim=320 --training.node_out_dim=320 --training.num_epochs=20 --featurization.epochs=20 --construction.time_window_size=15.0 --transformation.used_methods=none --project=spider_frozen --experiment=run_n_times --featurization.pretrained.weights_path=$SPIDER_WEIGHTS
./run.sh pretrained_threatrace CADETS_E3 --project=spider_frozen --experiment=run_n_times --featurization.pretrained.weights_path=$SPIDER_WEIGHTS
./run.sh pretrained_magic CADETS_E3 --project=spider_frozen --experiment=run_n_times --featurization.pretrained.weights_path=$SPIDER_WEIGHTS
./run.sh pretrained_kairos CADETS_E3 --project=spider_frozen --experiment=run_n_times --featurization.pretrained.weights_path=$SPIDER_WEIGHTS

# THEIA_E3
./run.sh pretrained_velox THEIA_E3 --project=spider_frozen --experiment=run_n_times --featurization.pretrained.weights_path=$SPIDER_WEIGHTS
./run.sh pretrained_orthrus THEIA_E3 --training.encoder.dropout=0.3 --training.lr=0.001 --training.node_hid_dim=128 --training.node_out_dim=128 --training.num_epochs=20 --construction.time_window_size=5 --project=spider_frozen --experiment=run_n_times --featurization.pretrained.weights_path=$SPIDER_WEIGHTS
./run.sh pretrained_nodlink THEIA_E3 --training.encoder.dropout=0.3 --training.lr=0.0001 --training.node_hid_dim=320 --training.node_out_dim=320 --training.num_epochs=20 --featurization.epochs=20 --construction.time_window_size=15.0 --transformation.used_methods=none --project=spider_frozen --experiment=run_n_times --featurization.pretrained.weights_path=$SPIDER_WEIGHTS
./run.sh pretrained_threatrace THEIA_E3 --project=spider_frozen --experiment=run_n_times --featurization.pretrained.weights_path=$SPIDER_WEIGHTS
./run.sh pretrained_flash THEIA_E3 --project=spider_frozen --experiment=run_n_times --featurization.pretrained.weights_path=$SPIDER_WEIGHTS

# THEIA_E5
./run.sh pretrained_velox THEIA_E5 --project=spider_frozen --experiment=run_n_times --featurization.pretrained.weights_path=$SPIDER_WEIGHTS
./run.sh pretrained_orthrus THEIA_E5 --training.encoder.dropout=0.3 --training.lr=0.001 --training.node_hid_dim=256 --training.node_out_dim=256 --training.num_epochs=12 --project=spider_frozen --experiment=run_n_times --featurization.pretrained.weights_path=$SPIDER_WEIGHTS
./run.sh pretrained_nodlink THEIA_E5 --training.encoder.dropout=0.3 --training.lr=0.001 --training.node_hid_dim=320 --training.node_out_dim=320 --training.num_epochs=12 --featurization.epochs=20 --construction.time_window_size=15.0 --transformation.used_methods=none --project=spider_frozen --experiment=run_n_times --featurization.pretrained.weights_path=$SPIDER_WEIGHTS
./run.sh pretrained_threatrace THEIA_E5 --project=spider_frozen --training.num_epochs=12 --experiment=run_n_times --featurization.pretrained.weights_path=$SPIDER_WEIGHTS
./run.sh pretrained_flash THEIA_E5 --project=spider_frozen --training.num_epochs=12 --experiment=run_n_times --featurization.pretrained.weights_path=$SPIDER_WEIGHTS

# CLEARSCOPE_E5
./run.sh pretrained_velox CLEARSCOPE_E5 --training.node_hid_dim=128 --training.node_out_dim=128 --project=spider_frozen --experiment=run_n_times --featurization.pretrained.weights_path=$SPIDER_WEIGHTS
./run.sh pretrained_orthrus CLEARSCOPE_E5 --training.encoder.dropout=0.3 --training.lr=0.001 --training.node_hid_dim=256 --training.node_out_dim=256 --training.num_epochs=12 --project=spider_frozen --experiment=run_n_times --featurization.pretrained.weights_path=$SPIDER_WEIGHTS
./run.sh pretrained_nodlink CLEARSCOPE_E5 --training.encoder.dropout=0.3 --training.lr=0.0001 --training.node_hid_dim=128 --training.node_out_dim=128 --training.num_epochs=12 --project=spider_frozen --experiment=run_n_times --featurization.pretrained.weights_path=$SPIDER_WEIGHTS
./run.sh pretrained_threatrace CLEARSCOPE_E5 --training.num_epochs=12 --project=spider_frozen --experiment=run_n_times --featurization.pretrained.weights_path=$SPIDER_WEIGHTS
./run.sh pretrained_flash CLEARSCOPE_E5 --training.num_epochs=12 --project=spider_frozen --experiment=run_n_times --featurization.pretrained.weights_path=$SPIDER_WEIGHTS

# optc_h051
./run.sh pretrained_velox optc_h051 --training.node_hid_dim=256 --training.node_out_dim=256 --project=spider_frozen --experiment=run_n_times --featurization.pretrained.weights_path=$SPIDER_WEIGHTS
./run.sh pretrained_orthrus optc_h051 --training.encoder.dropout=0.3 --training.lr=0.001 --training.node_hid_dim=128 --training.node_out_dim=128 --training.num_epochs=12 --project=spider_frozen --experiment=run_n_times --featurization.pretrained.weights_path=$SPIDER_WEIGHTS
./run.sh pretrained_nodlink optc_h051 --training.encoder.dropout=0.3 --training.lr=0.0001 --training.node_hid_dim=256 --training.node_out_dim=256 --training.num_epochs=12 --project=spider_frozen --experiment=run_n_times --featurization.pretrained.weights_path=$SPIDER_WEIGHTS
./run.sh pretrained_threatrace optc_h051 --training.num_epochs=12 --project=spider_frozen --experiment=run_n_times --featurization.pretrained.weights_path=$SPIDER_WEIGHTS
./run.sh pretrained_flash optc_h051 --training.num_epochs=12 --project=spider_frozen --experiment=run_n_times --featurization.pretrained.weights_path=$SPIDER_WEIGHTS
```

## Continued pretraining

SPIDER first continues pretraining on the training split of the target dataset (`--feat_inference.continue_pretrain=True`, see [Continued pretraining](pretrained_encoders.md#continued-pretraining-on-the-target-dataset)). This is the fine-tuned SPIDER of the paper.

```shell
# CADETS_E3
./run.sh pretrained_velox CADETS_E3 --project=spider_continued --experiment=run_n_times --featurization.pretrained.weights_path=$SPIDER_WEIGHTS --feat_inference.continue_pretrain=True --feat_inference.continue_pretrain_epochs=10
./run.sh pretrained_orthrus CADETS_E3 --training.encoder.dropout=0.3 --training.lr=0.0001 --training.node_hid_dim=256 --training.node_out_dim=256 --training.num_epochs=20 --project=spider_continued --experiment=run_n_times --featurization.pretrained.weights_path=$SPIDER_WEIGHTS --feat_inference.continue_pretrain=True --feat_inference.continue_pretrain_epochs=10
./run.sh pretrained_nodlink CADETS_E3 --training.encoder.dropout=0.3 --training.lr=0.0001 --training.node_hid_dim=320 --training.node_out_dim=320 --training.num_epochs=20 --featurization.epochs=20 --construction.time_window_size=15.0 --transformation.used_methods=none --project=spider_continued --experiment=run_n_times --featurization.pretrained.weights_path=$SPIDER_WEIGHTS --feat_inference.continue_pretrain=True --feat_inference.continue_pretrain_epochs=10
./run.sh pretrained_threatrace CADETS_E3 --project=spider_continued --experiment=run_n_times --featurization.pretrained.weights_path=$SPIDER_WEIGHTS --feat_inference.continue_pretrain=True --feat_inference.continue_pretrain_epochs=10
./run.sh pretrained_magic CADETS_E3 --project=spider_continued --experiment=run_n_times --featurization.pretrained.weights_path=$SPIDER_WEIGHTS --feat_inference.continue_pretrain=True --feat_inference.continue_pretrain_epochs=10
./run.sh pretrained_flash CADETS_E3 --project=spider_continued --experiment=run_n_times --featurization.pretrained.weights_path=$SPIDER_WEIGHTS --feat_inference.continue_pretrain=True --feat_inference.continue_pretrain_epochs=10

# THEIA_E3
./run.sh pretrained_velox THEIA_E3 --project=spider_continued --experiment=run_n_times --featurization.pretrained.weights_path=$SPIDER_WEIGHTS --feat_inference.continue_pretrain=True --feat_inference.continue_pretrain_epochs=10
./run.sh pretrained_orthrus THEIA_E3 --training.encoder.dropout=0.3 --training.lr=0.001 --training.node_hid_dim=128 --training.node_out_dim=128 --training.num_epochs=20 --construction.time_window_size=5 --project=spider_continued --experiment=run_n_times --featurization.pretrained.weights_path=$SPIDER_WEIGHTS --feat_inference.continue_pretrain=True --feat_inference.continue_pretrain_epochs=10
./run.sh pretrained_nodlink THEIA_E3 --training.encoder.dropout=0.3 --training.lr=0.0001 --training.node_hid_dim=320 --training.node_out_dim=320 --training.num_epochs=20 --featurization.epochs=20 --construction.time_window_size=15.0 --transformation.used_methods=none --project=spider_continued --experiment=run_n_times --featurization.pretrained.weights_path=$SPIDER_WEIGHTS --feat_inference.continue_pretrain=True --feat_inference.continue_pretrain_epochs=10
./run.sh pretrained_threatrace THEIA_E3 --project=spider_continued --experiment=run_n_times --featurization.pretrained.weights_path=$SPIDER_WEIGHTS --feat_inference.continue_pretrain=True --feat_inference.continue_pretrain_epochs=10
./run.sh pretrained_flash THEIA_E3 --project=spider_continued --experiment=run_n_times --featurization.pretrained.weights_path=$SPIDER_WEIGHTS --feat_inference.continue_pretrain=True --feat_inference.continue_pretrain_epochs=10

# THEIA_E5
./run.sh pretrained_velox THEIA_E5 --project=spider_continued --experiment=run_n_times --featurization.pretrained.weights_path=$SPIDER_WEIGHTS --feat_inference.continue_pretrain=True --feat_inference.continue_pretrain_epochs=5
./run.sh pretrained_orthrus THEIA_E5 --training.encoder.dropout=0.3 --training.lr=0.001 --training.node_hid_dim=256 --training.node_out_dim=256 --training.num_epochs=12 --project=spider_continued --experiment=run_n_times --featurization.pretrained.weights_path=$SPIDER_WEIGHTS --feat_inference.continue_pretrain=True --feat_inference.continue_pretrain_epochs=5
./run.sh pretrained_nodlink THEIA_E5 --training.encoder.dropout=0.3 --training.lr=0.001 --training.node_hid_dim=320 --training.node_out_dim=320 --training.num_epochs=12 --featurization.epochs=20 --construction.time_window_size=15.0 --transformation.used_methods=none --project=spider_continued --experiment=run_n_times --featurization.pretrained.weights_path=$SPIDER_WEIGHTS --feat_inference.continue_pretrain=True --feat_inference.continue_pretrain_epochs=5
./run.sh pretrained_threatrace THEIA_E5 --project=spider_continued --training.num_epochs=12 --experiment=run_n_times --featurization.pretrained.weights_path=$SPIDER_WEIGHTS --feat_inference.continue_pretrain=True --feat_inference.continue_pretrain_epochs=5
./run.sh pretrained_flash THEIA_E5 --project=spider_continued --training.num_epochs=12 --experiment=run_n_times --featurization.pretrained.weights_path=$SPIDER_WEIGHTS --feat_inference.continue_pretrain=True --feat_inference.continue_pretrain_epochs=5

# CLEARSCOPE_E5
./run.sh pretrained_velox CLEARSCOPE_E5 --project=spider_continued --experiment=run_n_times --featurization.pretrained.weights_path=$SPIDER_WEIGHTS --feat_inference.continue_pretrain=True --feat_inference.continue_pretrain_epochs=1
./run.sh pretrained_orthrus CLEARSCOPE_E5 --training.encoder.dropout=0.3 --training.lr=0.001 --training.node_hid_dim=256 --training.node_out_dim=256 --training.num_epochs=12 --project=spider_continued --experiment=run_n_times --featurization.pretrained.weights_path=$SPIDER_WEIGHTS --feat_inference.continue_pretrain=True --feat_inference.continue_pretrain_epochs=5
./run.sh pretrained_nodlink CLEARSCOPE_E5 --training.encoder.dropout=0.3 --training.lr=0.0001 --training.node_hid_dim=128 --training.node_out_dim=128 --training.num_epochs=12 --project=spider_continued --experiment=run_n_times --featurization.pretrained.weights_path=$SPIDER_WEIGHTS --feat_inference.continue_pretrain=True --feat_inference.continue_pretrain_epochs=5
./run.sh pretrained_threatrace CLEARSCOPE_E5 --training.num_epochs=12 --project=spider_continued --experiment=run_n_times --featurization.pretrained.weights_path=$SPIDER_WEIGHTS --feat_inference.continue_pretrain=True --feat_inference.continue_pretrain_epochs=5
./run.sh pretrained_flash CLEARSCOPE_E5 --training.num_epochs=12 --project=spider_continued --experiment=run_n_times --featurization.pretrained.weights_path=$SPIDER_WEIGHTS --feat_inference.continue_pretrain=True --feat_inference.continue_pretrain_epochs=5

# optc_h051
./run.sh pretrained_velox optc_h051 --project=spider_continued --experiment=run_n_times --featurization.pretrained.weights_path=$SPIDER_WEIGHTS --feat_inference.continue_pretrain=True --feat_inference.continue_pretrain_epochs=10
./run.sh pretrained_orthrus optc_h051 --training.encoder.dropout=0.3 --training.lr=0.001 --training.node_hid_dim=128 --training.node_out_dim=128 --training.num_epochs=12 --project=spider_continued --experiment=run_n_times --featurization.pretrained.weights_path=$SPIDER_WEIGHTS --feat_inference.continue_pretrain=True --feat_inference.continue_pretrain_epochs=10
./run.sh pretrained_nodlink optc_h051 --training.encoder.dropout=0.3 --training.lr=0.0001 --training.node_hid_dim=256 --training.node_out_dim=256 --training.num_epochs=12 --project=spider_continued --experiment=run_n_times --featurization.pretrained.weights_path=$SPIDER_WEIGHTS --feat_inference.continue_pretrain=True --feat_inference.continue_pretrain_epochs=10
./run.sh pretrained_threatrace optc_h051 --training.num_epochs=12 --project=spider_continued --experiment=run_n_times --featurization.pretrained.weights_path=$SPIDER_WEIGHTS --feat_inference.continue_pretrain=True --feat_inference.continue_pretrain_epochs=10
./run.sh pretrained_flash optc_h051 --training.num_epochs=12 --project=spider_continued --experiment=run_n_times --featurization.pretrained.weights_path=$SPIDER_WEIGHTS --feat_inference.continue_pretrain=True --feat_inference.continue_pretrain_epochs=10
```

## Differences with the paper

- The runs use torch 2.1 and the library versions of this release, so they are not bit-identical to the paper.
- MAGIC's GAT layers now follow the original MAGIC implementation (its decoder has 1 layer and 1 head instead of 3 and 4), so the `pretrained_magic` results differ from the paper.
