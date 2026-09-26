<h1 align="center">SPIDER</h1>

SPIDER is a Transformer encoder that embeds provenance-graph entities (processes, files, sockets) from their text attributes alone.
It is pretrained by distillation: a GNN teacher runs over benign provenance graphs to learn behavior-aware entity representations, and the text encoder (the student) learns to reproduce them.
At inference, only the text encoder is used, so SPIDER can replace the node featurizer of existing provenance-based intrusion detection systems (PIDSs).

This branch contains the original code to reproduce the main experiments of our NeurIPS 2026 paper.
It is built on top of [PIDSMaker](https://github.com/ubc-provenance/PIDSMaker).

The [online documentation](https://ubc-provenance.github.io/PIDSMaker/) describes the latest version of PIDSMaker, where SPIDER is available as a [pretrained encoder](https://ubc-provenance.github.io/PIDSMaker/features/pretrained_encoders/). The `docs/` folder of this branch describes the code of this branch.

## Setup

### Clone the repo
```sh
git clone https://github.com/ubc-provenance/PIDSMaker.git -b spider spider
cd spider
```

### Install PIDSMaker
PIDSMaker is a unified framework to experiment with state-of-the-art PIDSs.
It comes with a Docker image and all DARPA TC and OpTC datasets, please follow [these guidelines](docs/docs/ten-minute-install.md) to install it.

### Download the pretrained SPIDER weights
We provide the SPIDER weights used for all results in the paper ([Google Drive](https://drive.google.com/file/d/1NlUPQJTEehhe3cAOjlOq_wnPYjV-MMZ9/view?usp=drive_link), 21 MB).

```sh
gdown 1NlUPQJTEehhe3cAOjlOq_wnPYjV-MMZ9 -O SPIDER-weights.tar.gz
mkdir -p weights && tar -xzf SPIDER-weights.tar.gz -C weights
(cd weights/spider && sha256sum -c SHA256SUMS)
```

This creates `weights/spider/` with the text encoder (`pretrain_mini.pt`) and its tokenizer (`tokenizer.pt`), which are all that the frozen-encoder experiments need.
The fine-tuned experiments also need the GNN teacher (`gnn_teacher_mini_best.pt`), its behavior-label vocabulary (`behavior_vocab.txt`) and its edge-type indices (`edge_type2idx.json`); `spider_log_mini.txt` is the pretraining loss log.
Then export the folder's absolute path, which the commands below pass to `--featurization.feat_training.spider.spider_path` to skip pretraining:

```sh
export SPIDER_WEIGHTS=$(realpath weights/spider)
```

The path must be absolute because the checkpoint files are symlinked into each run's artifact folder.

### Pretrain SPIDER from scratch (optional)
Remove `--featurization.feat_training.spider.spider_path=...` from a command to pretrain SPIDER instead of loading the weights.
SPIDER is pretrained on benign data only, from the datasets listed in `pretrain_datasets` in [config/pretrained/spider/spider.yml](config/pretrained/spider/spider.yml): `PROVENANCE_BENIGN`, `TRACE_E3`, `TRACE_E5`, `CADETS_E5`, `CLEARSCOPE_E3` and `optc_h201`.
`PROVENANCE_BENIGN` is a benign Linux audit corpus: [scripts/provenance_capture/README.md](scripts/provenance_capture/README.md) describes how to collect it, and [create_database_provenance_benign.py](scripts/provenance_capture/create_database_provenance_benign.py) loads it into PostgreSQL.
The encoder comparison and ablation experiments below always pretrain from scratch.

## Reproduce results
All commands are run from `scripts/`:

```sh
cd scripts
```

`./run.sh` starts the pipeline as a background job and logs metrics to [Weights & Biases](https://wandb.ai) (run `wandb login` first), so launch the commands in batches that fit on your GPUs.
`--experiment=run_n_times` trains each detector with 5 seeds and logs the mean and standard deviation of every metric to the W&B project given by `--project`.
To run without W&B, use `./run_local.sh` instead of `./run.sh`: the per-seed metrics are then saved to `method_to_metrics.pkl` in the run's evaluation artifact folder.

### PIDSs without SPIDER
Each PIDS with its original node featurization.

```sh
# CADETS_E3
./run.sh velox CADETS_E3 --detection.gnn_training.encoder.dropout=0.3 --detection.gnn_training.lr=0.0001 --detection.gnn_training.node_hid_dim=256 --detection.gnn_training.node_out_dim=256 --detection.gnn_training.num_epochs=20 --featurization.feat_training.emb_dim=256 --project=spider_baselines --experiment=run_n_times
./run.sh orthrus_non_snooped CADETS_E3 --detection.gnn_training.encoder.dropout=0.3 --detection.gnn_training.lr=0.001 --detection.gnn_training.node_hid_dim=256 --detection.gnn_training.node_out_dim=256 --detection.gnn_training.num_epochs=20 --featurization.feat_training.emb_dim=256 --project=spider_baselines --experiment=run_n_times
./run.sh nodlink CADETS_E3 --detection.gnn_training.encoder.dropout=0.3 --detection.gnn_training.lr=0.001 --detection.gnn_training.node_hid_dim=320 --detection.gnn_training.node_out_dim=320 --detection.gnn_training.num_epochs=20 --featurization.feat_training.emb_dim=128 --featurization.feat_training.epochs=20 --preprocessing.build_graphs.time_window_size=15.0 --preprocessing.transformation.used_methods="none" --project=spider_baselines --experiment=run_n_times
./run.sh kairos CADETS_E3 --detection.gnn_training.encoder.dropout=0.3 --detection.gnn_training.lr=0.001 --detection.gnn_training.node_hid_dim=256 --detection.gnn_training.node_out_dim=256 --detection.gnn_training.num_epochs=20 --featurization.feat_training.emb_dim=256 --project=spider_baselines --experiment=run_n_times
./run.sh flash CADETS_E3 --detection.gnn_training.node_hid_dim=128 --detection.gnn_training.lr=0.0001 --detection.gnn_training.encoder.dropout=0.1 --detection.gnn_training.num_epochs=20 --featurization.feat_training.used_method=word2vec --project=spider_baselines --experiment=run_n_times

# THEIA_E3
./run.sh velox THEIA_E3 --detection.gnn_training.encoder.dropout=0.3 --detection.gnn_training.lr=0.001 --detection.gnn_training.node_hid_dim=320 --detection.gnn_training.node_out_dim=320 --detection.gnn_training.num_epochs=20 --featurization.feat_training.emb_dim=256 --project=spider_baselines --experiment=run_n_times
./run.sh orthrus_non_snooped THEIA_E3 --detection.gnn_training.encoder.dropout=0.3 --detection.gnn_training.lr=0.001 --detection.gnn_training.node_hid_dim=128 --detection.gnn_training.node_out_dim=128 --detection.gnn_training.num_epochs=20 --featurization.feat_training.emb_dim=256 --preprocessing.build_graphs.time_window_size=5 --project=spider_baselines --experiment=run_n_times
./run.sh nodlink THEIA_E3 --detection.gnn_training.encoder.dropout=0.3 --detection.gnn_training.lr=0.0001 --detection.gnn_training.node_hid_dim=320 --detection.gnn_training.node_out_dim=320 --detection.gnn_training.num_epochs=20 --featurization.feat_training.emb_dim=128 --featurization.feat_training.epochs=20 --preprocessing.build_graphs.time_window_size=15.0 --preprocessing.transformation.used_methods="none" --project=spider_baselines --experiment=run_n_times
./run.sh kairos THEIA_E3 --detection.gnn_training.encoder.dropout=0.3 --detection.gnn_training.lr=0.001 --detection.gnn_training.node_hid_dim=320 --detection.gnn_training.node_out_dim=320 --detection.gnn_training.num_epochs=20 --featurization.feat_training.emb_dim=256 --project=spider_baselines --experiment=run_n_times
./run.sh flash THEIA_E3 --detection.gnn_training.node_hid_dim=128 --detection.gnn_training.lr=0.0001 --detection.gnn_training.encoder.dropout=0.1 --detection.gnn_training.num_epochs=20 --featurization.feat_training.used_method=word2vec --project=spider_baselines --experiment=run_n_times

# THEIA_E5
./run.sh velox THEIA_E5 --detection.gnn_training.encoder.dropout=0.3 --detection.gnn_training.lr=0.0001 --detection.gnn_training.node_hid_dim=128 --detection.gnn_training.node_out_dim=128 --detection.gnn_training.num_epochs=12 --featurization.feat_training.emb_dim=128 --preprocessing.build_graphs.time_window_size=15.0 --project=spider_baselines --experiment=run_n_times
./run.sh orthrus_non_snooped THEIA_E5 --detection.gnn_training.encoder.dropout=0.3 --detection.gnn_training.lr=0.001 --detection.gnn_training.node_hid_dim=256 --detection.gnn_training.node_out_dim=256 --detection.gnn_training.num_epochs=12 --featurization.feat_training.emb_dim=64 --project=spider_baselines --experiment=run_n_times
./run.sh nodlink THEIA_E5 --detection.gnn_training.encoder.dropout=0.3 --detection.gnn_training.lr=0.001 --detection.gnn_training.node_hid_dim=320 --detection.gnn_training.node_out_dim=320 --detection.gnn_training.num_epochs=12 --featurization.feat_training.emb_dim=128 --featurization.feat_training.epochs=20 --preprocessing.build_graphs.time_window_size=15.0 --preprocessing.transformation.used_methods="none" --project=spider_baselines --experiment=run_n_times
./run.sh kairos THEIA_E5 --detection.gnn_training.encoder.dropout=0.3 --detection.gnn_training.lr=0.001 --detection.gnn_training.node_hid_dim=256 --detection.gnn_training.node_out_dim=256 --detection.gnn_training.num_epochs=12 --featurization.feat_training.emb_dim=256 --project=spider_baselines --experiment=run_n_times
./run.sh flash THEIA_E5 --detection.gnn_training.node_hid_dim=128 --detection.gnn_training.lr=0.0001 --detection.gnn_training.encoder.dropout=0.1 --detection.gnn_training.num_epochs=20 --featurization.feat_training.used_method=word2vec --project=spider_baselines --experiment=run_n_times

# CLEARSCOPE_E5
./run.sh velox CLEARSCOPE_E5 --detection.gnn_training.encoder.dropout=0.3 --detection.gnn_training.lr=0.001 --detection.gnn_training.node_hid_dim=128 --detection.gnn_training.node_out_dim=128 --detection.gnn_training.num_epochs=12 --featurization.feat_training.emb_dim=128 --project=spider_baselines --experiment=run_n_times
./run.sh orthrus_non_snooped CLEARSCOPE_E5 --detection.gnn_training.encoder.dropout=0.3 --detection.gnn_training.lr=0.001 --detection.gnn_training.node_hid_dim=256 --detection.gnn_training.node_out_dim=256 --detection.gnn_training.num_epochs=12 --featurization.feat_training.emb_dim=128 --project=spider_baselines --experiment=run_n_times
./run.sh nodlink CLEARSCOPE_E5 --detection.gnn_training.encoder.dropout=0.3 --detection.gnn_training.lr=0.0001 --detection.gnn_training.node_hid_dim=128 --detection.gnn_training.node_out_dim=128 --detection.gnn_training.num_epochs=12 --featurization.feat_training.emb_dim=256 --project=spider_baselines --experiment=run_n_times
./run.sh kairos CLEARSCOPE_E5 --detection.gnn_training.encoder.dropout=0.3 --detection.gnn_training.lr=0.001 --detection.gnn_training.node_hid_dim=64 --detection.gnn_training.node_out_dim=64 --detection.gnn_training.num_epochs=12 --featurization.feat_training.emb_dim=32 --project=spider_baselines --experiment=run_n_times
./run.sh flash CLEARSCOPE_E5 --detection.gnn_training.node_hid_dim=128 --detection.gnn_training.lr=0.0001 --detection.gnn_training.encoder.dropout=0.1 --detection.gnn_training.num_epochs=20 --featurization.feat_training.used_method=word2vec --project=spider_baselines --experiment=run_n_times

# optc_h501
./run.sh velox optc_h501 --detection.gnn_training.encoder.dropout=0.3 --detection.gnn_training.lr=0.001 --detection.gnn_training.node_hid_dim=320 --detection.gnn_training.node_out_dim=320 --detection.gnn_training.num_epochs=12 --featurization.feat_training.emb_dim=256 --project=spider_baselines --experiment=run_n_times
./run.sh orthrus_non_snooped optc_h501 --detection.gnn_training.encoder.dropout=0.3 --detection.gnn_training.lr=0.001 --detection.gnn_training.node_hid_dim=256 --detection.gnn_training.node_out_dim=256 --detection.gnn_training.num_epochs=12 --featurization.feat_training.emb_dim=256 --project=spider_baselines --experiment=run_n_times
./run.sh nodlink optc_h501 --detection.gnn_training.encoder.dropout=0.3 --detection.gnn_training.lr=0.0001 --detection.gnn_training.node_hid_dim=128 --detection.gnn_training.node_out_dim=128 --detection.gnn_training.num_epochs=12 --featurization.feat_training.emb_dim=256 --project=spider_baselines --experiment=run_n_times
./run.sh kairos optc_h501 --detection.gnn_training.encoder.dropout=0.3 --detection.gnn_training.lr=0.001 --detection.gnn_training.node_hid_dim=64 --detection.gnn_training.node_out_dim=64 --detection.gnn_training.num_epochs=12 --featurization.feat_training.emb_dim=16 --project=spider_baselines --experiment=run_n_times

# optc_h051
./run.sh velox optc_h051 --detection.gnn_training.encoder.dropout=0.3 --detection.gnn_training.lr=0.001 --detection.gnn_training.node_hid_dim=320 --detection.gnn_training.node_out_dim=320 --detection.gnn_training.num_epochs=12 --featurization.feat_training.emb_dim=256 --project=spider_baselines --experiment=run_n_times
./run.sh orthrus_non_snooped optc_h051 --detection.gnn_training.encoder.dropout=0.3 --detection.gnn_training.lr=0.001 --detection.gnn_training.node_hid_dim=128 --detection.gnn_training.node_out_dim=128 --detection.gnn_training.num_epochs=12 --featurization.feat_training.emb_dim=256 --project=spider_baselines --experiment=run_n_times
./run.sh nodlink optc_h051 --detection.gnn_training.encoder.dropout=0.3 --detection.gnn_training.lr=0.0001 --detection.gnn_training.node_hid_dim=256 --detection.gnn_training.node_out_dim=256 --detection.gnn_training.num_epochs=12 --featurization.feat_training.emb_dim=256 --project=spider_baselines --experiment=run_n_times
./run.sh kairos optc_h051 --detection.gnn_training.encoder.dropout=0.3 --detection.gnn_training.lr=0.001 --detection.gnn_training.node_hid_dim=128 --detection.gnn_training.node_out_dim=128 --detection.gnn_training.num_epochs=12 --featurization.feat_training.emb_dim=32 --project=spider_baselines --experiment=run_n_times
./run.sh flash optc_h051 --detection.gnn_training.node_hid_dim=128 --detection.gnn_training.lr=0.0001 --detection.gnn_training.encoder.dropout=0.1 --detection.gnn_training.num_epochs=20 --featurization.feat_training.used_method=word2vec --project=spider_baselines --experiment=run_n_times
```

### PIDSs with a frozen SPIDER encoder
Each `spider_<pids>` config in [config/pretrained/spider/](config/pretrained/spider/) inherits the PIDS config and replaces its node features with embeddings from the frozen SPIDER encoder.

```sh
# CADETS_E3
./run.sh spider_velox CADETS_E3 --detection.gnn_training.num_epochs=22 --project=spider_frozen --experiment=run_n_times --featurization.feat_training.spider.spider_path=$SPIDER_WEIGHTS
./run.sh spider_orthrus CADETS_E3 --detection.gnn_training.encoder.dropout=0.3 --detection.gnn_training.lr=0.0001 --detection.gnn_training.node_hid_dim=256 --detection.gnn_training.node_out_dim=256 --detection.gnn_training.num_epochs=20 --project=spider_frozen --experiment=run_n_times --featurization.feat_training.spider.spider_path=$SPIDER_WEIGHTS
./run.sh spider_nodlink CADETS_E3 --detection.gnn_training.encoder.dropout=0.3 --detection.gnn_training.lr=0.0001 --detection.gnn_training.node_hid_dim=320 --detection.gnn_training.node_out_dim=320 --detection.gnn_training.num_epochs=20 --featurization.feat_training.epochs=20 --preprocessing.build_graphs.time_window_size=15.0 --preprocessing.transformation.used_methods="none" --project=spider_frozen --experiment=run_n_times --featurization.feat_training.spider.spider_path=$SPIDER_WEIGHTS
./run.sh spider_threatrace CADETS_E3 --project=spider_frozen --experiment=run_n_times --featurization.feat_training.spider.spider_path=$SPIDER_WEIGHTS
./run.sh spider_magic CADETS_E3 --project=spider_frozen --experiment=run_n_times --featurization.feat_training.spider.spider_path=$SPIDER_WEIGHTS
./run.sh spider_kairos CADETS_E3 --project=spider_frozen --experiment=run_n_times --featurization.feat_training.spider.spider_path=$SPIDER_WEIGHTS

# THEIA_E3
./run.sh spider_velox THEIA_E3 --project=spider_frozen --experiment=run_n_times --featurization.feat_training.spider.spider_path=$SPIDER_WEIGHTS
./run.sh spider_orthrus THEIA_E3 --detection.gnn_training.encoder.dropout=0.3 --detection.gnn_training.lr=0.001 --detection.gnn_training.node_hid_dim=128 --detection.gnn_training.node_out_dim=128 --detection.gnn_training.num_epochs=20 --preprocessing.build_graphs.time_window_size=5 --project=spider_frozen --experiment=run_n_times --featurization.feat_training.spider.spider_path=$SPIDER_WEIGHTS
./run.sh spider_nodlink THEIA_E3 --detection.gnn_training.encoder.dropout=0.3 --detection.gnn_training.lr=0.0001 --detection.gnn_training.node_hid_dim=320 --detection.gnn_training.node_out_dim=320 --detection.gnn_training.num_epochs=20 --featurization.feat_training.epochs=20 --preprocessing.build_graphs.time_window_size=15.0 --preprocessing.transformation.used_methods="none" --project=spider_frozen --experiment=run_n_times --featurization.feat_training.spider.spider_path=$SPIDER_WEIGHTS
./run.sh spider_threatrace THEIA_E3 --project=spider_frozen --experiment=run_n_times --featurization.feat_training.spider.spider_path=$SPIDER_WEIGHTS
./run.sh spider_flash THEIA_E3 --project=spider_frozen --experiment=run_n_times --featurization.feat_training.spider.spider_path=$SPIDER_WEIGHTS

# THEIA_E5
./run.sh spider_velox THEIA_E5 --project=spider_frozen --experiment=run_n_times --featurization.feat_training.spider.spider_path=$SPIDER_WEIGHTS
./run.sh spider_orthrus THEIA_E5 --detection.gnn_training.encoder.dropout=0.3 --detection.gnn_training.lr=0.001 --detection.gnn_training.node_hid_dim=256 --detection.gnn_training.node_out_dim=256 --detection.gnn_training.num_epochs=12 --project=spider_frozen --experiment=run_n_times --featurization.feat_training.spider.spider_path=$SPIDER_WEIGHTS
./run.sh spider_nodlink THEIA_E5 --detection.gnn_training.encoder.dropout=0.3 --detection.gnn_training.lr=0.001 --detection.gnn_training.node_hid_dim=320 --detection.gnn_training.node_out_dim=320 --detection.gnn_training.num_epochs=12 --featurization.feat_training.epochs=20 --preprocessing.build_graphs.time_window_size=15.0 --preprocessing.transformation.used_methods="none" --project=spider_frozen --experiment=run_n_times --featurization.feat_training.spider.spider_path=$SPIDER_WEIGHTS
./run.sh spider_threatrace THEIA_E5 --project=spider_frozen --detection.gnn_training.num_epochs=12 --experiment=run_n_times --featurization.feat_training.spider.spider_path=$SPIDER_WEIGHTS
./run.sh spider_flash THEIA_E5 --project=spider_frozen --detection.gnn_training.num_epochs=12 --experiment=run_n_times --featurization.feat_training.spider.spider_path=$SPIDER_WEIGHTS

# CLEARSCOPE_E5
./run.sh spider_velox CLEARSCOPE_E5 --detection.gnn_training.node_hid_dim=128 --detection.gnn_training.node_out_dim=128 --project=spider_frozen --experiment=run_n_times --featurization.feat_training.spider.spider_path=$SPIDER_WEIGHTS
./run.sh spider_orthrus CLEARSCOPE_E5 --detection.gnn_training.encoder.dropout=0.3 --detection.gnn_training.lr=0.001 --detection.gnn_training.node_hid_dim=256 --detection.gnn_training.node_out_dim=256 --detection.gnn_training.num_epochs=12 --project=spider_frozen --experiment=run_n_times --featurization.feat_training.spider.spider_path=$SPIDER_WEIGHTS
./run.sh spider_nodlink CLEARSCOPE_E5 --detection.gnn_training.encoder.dropout=0.3 --detection.gnn_training.lr=0.0001 --detection.gnn_training.node_hid_dim=128 --detection.gnn_training.node_out_dim=128 --detection.gnn_training.num_epochs=12 --project=spider_frozen --experiment=run_n_times --featurization.feat_training.spider.spider_path=$SPIDER_WEIGHTS
./run.sh spider_threatrace CLEARSCOPE_E5 --detection.gnn_training.num_epochs=12 --project=spider_frozen --experiment=run_n_times --featurization.feat_training.spider.spider_path=$SPIDER_WEIGHTS
./run.sh spider_flash CLEARSCOPE_E5 --detection.gnn_training.num_epochs=12 --project=spider_frozen --experiment=run_n_times --featurization.feat_training.spider.spider_path=$SPIDER_WEIGHTS

# optc_h501
./run.sh spider_velox optc_h501 --project=spider_frozen --experiment=run_n_times --featurization.feat_training.spider.spider_path=$SPIDER_WEIGHTS
./run.sh spider_orthrus optc_h501 --detection.gnn_training.encoder.dropout=0.3 --detection.gnn_training.lr=0.001 --detection.gnn_training.node_hid_dim=256 --detection.gnn_training.node_out_dim=256 --detection.gnn_training.num_epochs=12 --project=spider_frozen --experiment=run_n_times --featurization.feat_training.spider.spider_path=$SPIDER_WEIGHTS
./run.sh spider_nodlink optc_h501 --detection.gnn_training.encoder.dropout=0.3 --detection.gnn_training.lr=0.0001 --detection.gnn_training.node_hid_dim=128 --detection.gnn_training.node_out_dim=128 --detection.gnn_training.num_epochs=12 --project=spider_frozen --experiment=run_n_times --featurization.feat_training.spider.spider_path=$SPIDER_WEIGHTS
./run.sh spider_threatrace optc_h501 --detection.gnn_training.num_epochs=12 --project=spider_frozen --experiment=run_n_times --featurization.feat_training.spider.spider_path=$SPIDER_WEIGHTS
./run.sh spider_flash optc_h501 --detection.gnn_training.num_epochs=12 --project=spider_frozen --experiment=run_n_times --featurization.feat_training.spider.spider_path=$SPIDER_WEIGHTS

# optc_h051
./run.sh spider_velox optc_h051 --detection.gnn_training.node_hid_dim=256 --detection.gnn_training.node_out_dim=256 --project=spider_frozen --experiment=run_n_times --featurization.feat_training.spider.spider_path=$SPIDER_WEIGHTS
./run.sh spider_orthrus optc_h051 --detection.gnn_training.encoder.dropout=0.3 --detection.gnn_training.lr=0.001 --detection.gnn_training.node_hid_dim=128 --detection.gnn_training.node_out_dim=128 --detection.gnn_training.num_epochs=12 --project=spider_frozen --experiment=run_n_times --featurization.feat_training.spider.spider_path=$SPIDER_WEIGHTS
./run.sh spider_nodlink optc_h051 --detection.gnn_training.encoder.dropout=0.3 --detection.gnn_training.lr=0.0001 --detection.gnn_training.node_hid_dim=256 --detection.gnn_training.node_out_dim=256 --detection.gnn_training.num_epochs=12 --project=spider_frozen --experiment=run_n_times --featurization.feat_training.spider.spider_path=$SPIDER_WEIGHTS
./run.sh spider_threatrace optc_h051 --detection.gnn_training.num_epochs=12 --project=spider_frozen --experiment=run_n_times --featurization.feat_training.spider.spider_path=$SPIDER_WEIGHTS
./run.sh spider_flash optc_h051 --detection.gnn_training.num_epochs=12 --project=spider_frozen --experiment=run_n_times --featurization.feat_training.spider.spider_path=$SPIDER_WEIGHTS
```

### PIDSs with a fine-tuned SPIDER encoder
Same as above, but SPIDER first continues pretraining with its own objective on the training split of the target dataset (`--featurization.feat_inference.continue_pretrain=True`).

```sh
# CADETS_E3
./run.sh spider_velox CADETS_E3 --project=spider_finetuned --experiment=run_n_times --featurization.feat_training.spider.spider_path=$SPIDER_WEIGHTS --featurization.feat_inference.continue_pretrain=True --featurization.feat_inference.continue_pretrain_epochs=10
./run.sh spider_orthrus CADETS_E3 --detection.gnn_training.encoder.dropout=0.3 --detection.gnn_training.lr=0.0001 --detection.gnn_training.node_hid_dim=256 --detection.gnn_training.node_out_dim=256 --detection.gnn_training.num_epochs=20 --project=spider_finetuned --experiment=run_n_times --featurization.feat_training.spider.spider_path=$SPIDER_WEIGHTS --featurization.feat_inference.continue_pretrain=True --featurization.feat_inference.continue_pretrain_epochs=10
./run.sh spider_nodlink CADETS_E3 --detection.gnn_training.encoder.dropout=0.3 --detection.gnn_training.lr=0.0001 --detection.gnn_training.node_hid_dim=320 --detection.gnn_training.node_out_dim=320 --detection.gnn_training.num_epochs=20 --featurization.feat_training.epochs=20 --preprocessing.build_graphs.time_window_size=15.0 --preprocessing.transformation.used_methods="none" --project=spider_finetuned --experiment=run_n_times --featurization.feat_training.spider.spider_path=$SPIDER_WEIGHTS --featurization.feat_inference.continue_pretrain=True --featurization.feat_inference.continue_pretrain_epochs=10
./run.sh spider_threatrace CADETS_E3 --project=spider_finetuned --experiment=run_n_times --featurization.feat_training.spider.spider_path=$SPIDER_WEIGHTS --featurization.feat_inference.continue_pretrain=True --featurization.feat_inference.continue_pretrain_epochs=10
./run.sh spider_magic CADETS_E3 --project=spider_finetuned --experiment=run_n_times --featurization.feat_training.spider.spider_path=$SPIDER_WEIGHTS --featurization.feat_inference.continue_pretrain=True --featurization.feat_inference.continue_pretrain_epochs=10
./run.sh spider_flash CADETS_E3 --project=spider_finetuned --experiment=run_n_times --featurization.feat_training.spider.spider_path=$SPIDER_WEIGHTS --featurization.feat_inference.continue_pretrain=True --featurization.feat_inference.continue_pretrain_epochs=10

# THEIA_E3
./run.sh spider_velox THEIA_E3 --project=spider_finetuned --experiment=run_n_times --featurization.feat_training.spider.spider_path=$SPIDER_WEIGHTS --featurization.feat_inference.continue_pretrain=True --featurization.feat_inference.continue_pretrain_epochs=10
./run.sh spider_orthrus THEIA_E3 --detection.gnn_training.encoder.dropout=0.3 --detection.gnn_training.lr=0.001 --detection.gnn_training.node_hid_dim=128 --detection.gnn_training.node_out_dim=128 --detection.gnn_training.num_epochs=20 --preprocessing.build_graphs.time_window_size=5 --project=spider_finetuned --experiment=run_n_times --featurization.feat_training.spider.spider_path=$SPIDER_WEIGHTS --featurization.feat_inference.continue_pretrain=True --featurization.feat_inference.continue_pretrain_epochs=10
./run.sh spider_nodlink THEIA_E3 --detection.gnn_training.encoder.dropout=0.3 --detection.gnn_training.lr=0.0001 --detection.gnn_training.node_hid_dim=320 --detection.gnn_training.node_out_dim=320 --detection.gnn_training.num_epochs=20 --featurization.feat_training.epochs=20 --preprocessing.build_graphs.time_window_size=15.0 --preprocessing.transformation.used_methods="none" --project=spider_finetuned --experiment=run_n_times --featurization.feat_training.spider.spider_path=$SPIDER_WEIGHTS --featurization.feat_inference.continue_pretrain=True --featurization.feat_inference.continue_pretrain_epochs=10
./run.sh spider_threatrace THEIA_E3 --project=spider_finetuned --experiment=run_n_times --featurization.feat_training.spider.spider_path=$SPIDER_WEIGHTS --featurization.feat_inference.continue_pretrain=True --featurization.feat_inference.continue_pretrain_epochs=10
./run.sh spider_flash THEIA_E3 --project=spider_finetuned --experiment=run_n_times --featurization.feat_training.spider.spider_path=$SPIDER_WEIGHTS --featurization.feat_inference.continue_pretrain=True --featurization.feat_inference.continue_pretrain_epochs=10

# THEIA_E5
./run.sh spider_velox THEIA_E5 --project=spider_finetuned --experiment=run_n_times --featurization.feat_training.spider.spider_path=$SPIDER_WEIGHTS --featurization.feat_inference.continue_pretrain=True --featurization.feat_inference.continue_pretrain_epochs=5
./run.sh spider_orthrus THEIA_E5 --detection.gnn_training.encoder.dropout=0.3 --detection.gnn_training.lr=0.001 --detection.gnn_training.node_hid_dim=256 --detection.gnn_training.node_out_dim=256 --detection.gnn_training.num_epochs=12 --project=spider_finetuned --experiment=run_n_times --featurization.feat_training.spider.spider_path=$SPIDER_WEIGHTS --featurization.feat_inference.continue_pretrain=True --featurization.feat_inference.continue_pretrain_epochs=5
./run.sh spider_nodlink THEIA_E5 --detection.gnn_training.encoder.dropout=0.3 --detection.gnn_training.lr=0.001 --detection.gnn_training.node_hid_dim=320 --detection.gnn_training.node_out_dim=320 --detection.gnn_training.num_epochs=12 --featurization.feat_training.epochs=20 --preprocessing.build_graphs.time_window_size=15.0 --preprocessing.transformation.used_methods="none" --project=spider_finetuned --experiment=run_n_times --featurization.feat_training.spider.spider_path=$SPIDER_WEIGHTS --featurization.feat_inference.continue_pretrain=True --featurization.feat_inference.continue_pretrain_epochs=5
./run.sh spider_threatrace THEIA_E5 --project=spider_finetuned --detection.gnn_training.num_epochs=12 --experiment=run_n_times --featurization.feat_training.spider.spider_path=$SPIDER_WEIGHTS --featurization.feat_inference.continue_pretrain=True --featurization.feat_inference.continue_pretrain_epochs=5
./run.sh spider_flash THEIA_E5 --project=spider_finetuned --detection.gnn_training.num_epochs=12 --experiment=run_n_times --featurization.feat_training.spider.spider_path=$SPIDER_WEIGHTS --featurization.feat_inference.continue_pretrain=True --featurization.feat_inference.continue_pretrain_epochs=5

# CLEARSCOPE_E5
./run.sh spider_velox CLEARSCOPE_E5 --project=spider_finetuned --experiment=run_n_times --featurization.feat_training.spider.spider_path=$SPIDER_WEIGHTS --featurization.feat_inference.continue_pretrain=True --featurization.feat_inference.continue_pretrain_epochs=1
./run.sh spider_orthrus CLEARSCOPE_E5 --detection.gnn_training.encoder.dropout=0.3 --detection.gnn_training.lr=0.001 --detection.gnn_training.node_hid_dim=256 --detection.gnn_training.node_out_dim=256 --detection.gnn_training.num_epochs=12 --project=spider_finetuned --experiment=run_n_times --featurization.feat_training.spider.spider_path=$SPIDER_WEIGHTS --featurization.feat_inference.continue_pretrain=True --featurization.feat_inference.continue_pretrain_epochs=5
./run.sh spider_nodlink CLEARSCOPE_E5 --detection.gnn_training.encoder.dropout=0.3 --detection.gnn_training.lr=0.0001 --detection.gnn_training.node_hid_dim=128 --detection.gnn_training.node_out_dim=128 --detection.gnn_training.num_epochs=12 --project=spider_finetuned --experiment=run_n_times --featurization.feat_training.spider.spider_path=$SPIDER_WEIGHTS --featurization.feat_inference.continue_pretrain=True --featurization.feat_inference.continue_pretrain_epochs=5
./run.sh spider_threatrace CLEARSCOPE_E5 --detection.gnn_training.num_epochs=12 --project=spider_finetuned --experiment=run_n_times --featurization.feat_training.spider.spider_path=$SPIDER_WEIGHTS --featurization.feat_inference.continue_pretrain=True --featurization.feat_inference.continue_pretrain_epochs=5
./run.sh spider_flash CLEARSCOPE_E5 --detection.gnn_training.num_epochs=12 --project=spider_finetuned --experiment=run_n_times --featurization.feat_training.spider.spider_path=$SPIDER_WEIGHTS --featurization.feat_inference.continue_pretrain=True --featurization.feat_inference.continue_pretrain_epochs=5

# optc_h501
./run.sh spider_velox optc_h501 --project=spider_finetuned --experiment=run_n_times --featurization.feat_training.spider.spider_path=$SPIDER_WEIGHTS --featurization.feat_inference.continue_pretrain=True --featurization.feat_inference.continue_pretrain_epochs=10
./run.sh spider_orthrus optc_h501 --detection.gnn_training.encoder.dropout=0.3 --detection.gnn_training.lr=0.001 --detection.gnn_training.node_hid_dim=256 --detection.gnn_training.node_out_dim=256 --detection.gnn_training.num_epochs=12 --project=spider_finetuned --experiment=run_n_times --featurization.feat_training.spider.spider_path=$SPIDER_WEIGHTS --featurization.feat_inference.continue_pretrain=True --featurization.feat_inference.continue_pretrain_epochs=10
./run.sh spider_nodlink optc_h501 --detection.gnn_training.encoder.dropout=0.3 --detection.gnn_training.lr=0.0001 --detection.gnn_training.node_hid_dim=128 --detection.gnn_training.node_out_dim=128 --detection.gnn_training.num_epochs=12 --project=spider_finetuned --experiment=run_n_times --featurization.feat_training.spider.spider_path=$SPIDER_WEIGHTS --featurization.feat_inference.continue_pretrain=True --featurization.feat_inference.continue_pretrain_epochs=10
./run.sh spider_threatrace optc_h501 --detection.gnn_training.num_epochs=12 --project=spider_finetuned --experiment=run_n_times --featurization.feat_training.spider.spider_path=$SPIDER_WEIGHTS --featurization.feat_inference.continue_pretrain=True --featurization.feat_inference.continue_pretrain_epochs=10
./run.sh spider_flash optc_h501 --detection.gnn_training.num_epochs=12 --project=spider_finetuned --experiment=run_n_times --featurization.feat_training.spider.spider_path=$SPIDER_WEIGHTS --featurization.feat_inference.continue_pretrain=True --featurization.feat_inference.continue_pretrain_epochs=10

# optc_h051
./run.sh spider_velox optc_h051 --project=spider_finetuned --experiment=run_n_times --featurization.feat_training.spider.spider_path=$SPIDER_WEIGHTS --featurization.feat_inference.continue_pretrain=True --featurization.feat_inference.continue_pretrain_epochs=10
./run.sh spider_orthrus optc_h051 --detection.gnn_training.encoder.dropout=0.3 --detection.gnn_training.lr=0.001 --detection.gnn_training.node_hid_dim=128 --detection.gnn_training.node_out_dim=128 --detection.gnn_training.num_epochs=12 --project=spider_finetuned --experiment=run_n_times --featurization.feat_training.spider.spider_path=$SPIDER_WEIGHTS --featurization.feat_inference.continue_pretrain=True --featurization.feat_inference.continue_pretrain_epochs=10
./run.sh spider_nodlink optc_h051 --detection.gnn_training.encoder.dropout=0.3 --detection.gnn_training.lr=0.0001 --detection.gnn_training.node_hid_dim=256 --detection.gnn_training.node_out_dim=256 --detection.gnn_training.num_epochs=12 --project=spider_finetuned --experiment=run_n_times --featurization.feat_training.spider.spider_path=$SPIDER_WEIGHTS --featurization.feat_inference.continue_pretrain=True --featurization.feat_inference.continue_pretrain_epochs=10
./run.sh spider_threatrace optc_h051 --detection.gnn_training.num_epochs=12 --project=spider_finetuned --experiment=run_n_times --featurization.feat_training.spider.spider_path=$SPIDER_WEIGHTS --featurization.feat_inference.continue_pretrain=True --featurization.feat_inference.continue_pretrain_epochs=10
./run.sh spider_flash optc_h051 --detection.gnn_training.num_epochs=12 --project=spider_finetuned --experiment=run_n_times --featurization.feat_training.spider.spider_path=$SPIDER_WEIGHTS --featurization.feat_inference.continue_pretrain=True --featurization.feat_inference.continue_pretrain_epochs=10
```

### Comparison with other entity encoders
Replaces SPIDER with other entity encoders, each trained on the same benign corpus and plugged into Velox (`spider_velox`).

```sh
./run.sh spider_velox CADETS_E3 --project=spider_encoders --experiment=run_n_times --featurization.feat_training.spider.model_type=bert --featurization.feat_training.spider.pretrain_tokens=150000000 --featurization.feat_training.spider.warmup_tokens=15000000 --exp=dataset_bert
./run.sh spider_velox CADETS_E3 --project=spider_encoders --experiment=run_n_times --featurization.feat_training.spider.model_type=graphmae --exp=dataset_graphmae
./run.sh spider_velox CADETS_E3 --project=spider_encoders --experiment=run_n_times --featurization.feat_training.spider.model_type=deepwalk --exp=dataset_deepwalk
./run.sh spider_velox CADETS_E3 --project=spider_encoders --experiment=run_n_times --featurization.feat_training.spider.model_type=dgi --exp=dataset_dgi
./run.sh spider_velox CADETS_E3 --project=spider_encoders --experiment=run_n_times --featurization.feat_training.spider.model_type=node2vec --exp=dataset_node2vec
./run.sh spider_velox CADETS_E3 --project=spider_encoders --experiment=run_n_times --featurization.feat_training.spider.model_type=gae --exp=dataset_gae
./run.sh spider_velox CADETS_E3 --project=spider_encoders --experiment=run_n_times --exp=dataset_word2vec --featurization.feat_training.used_method=word2vec --featurization.feat_training.pretrain_datasets='PROVENANCE_BENIGN,TRACE_E3,TRACE_E5,CADETS_E5,CLEARSCOPE_E3,optc_h201'
./run.sh spider_velox CADETS_E3 --project=spider_encoders --experiment=run_n_times --exp=dataset_fasttext --featurization.feat_training.used_method=fasttext --featurization.feat_training.pretrain_datasets='PROVENANCE_BENIGN,TRACE_E3,TRACE_E5,CADETS_E5,CLEARSCOPE_E3,optc_h201'
./run.sh spider_velox CADETS_E3 --project=spider_encoders --experiment=run_n_times --featurization.feat_training.spider.model_type=llama --featurization.feat_training.spider.pretrain_tokens=150000000 --featurization.feat_training.spider.warmup_tokens=15000000 --exp=dataset_llama
./run.sh spider_velox CADETS_E3 --project=spider_encoders --experiment=run_n_times --featurization.feat_training.spider.model_type=gpt2_pretrained --featurization.feat_training.spider.pretrain_tokens=50000000 --featurization.feat_training.spider.warmup_tokens=5000000 --exp=dataset_gpt2_pretrained
```

### Ablations
Variants of SPIDER's training objective and tokenizer, evaluated with Velox on CADETS_E3.

```sh
# Distillation loss: MSE instead of SCE
./run.sh spider_velox CADETS_E3 --project=spider_ablations --experiment=run_n_times --featurization.feat_training.spider.spider.distill_loss=mse --exp=dataset_ablation_distill_mse

# Teacher loss: BCE (cross-entropy on entity class) instead of SupCon
./run.sh spider_velox CADETS_E3 --project=spider_ablations --experiment=run_n_times --featurization.feat_training.spider.spider.teacher_loss=bce --exp=dataset_ablation_teacher_bce

# Teacher data: signature only (no GNN embedding)
./run.sh spider_velox CADETS_E3 --project=spider_ablations --experiment=run_n_times --featurization.feat_training.spider.spider.teacher_data=signature --exp=dataset_ablation_teacher_sig_only

# Teacher data: GNN embedding only (no signature)
./run.sh spider_velox CADETS_E3 --project=spider_ablations --experiment=run_n_times --featurization.feat_training.spider.spider.teacher_data=gnn_emb --exp=dataset_ablation_teacher_gnn_only

# Student-only: predict signature vector directly (no teacher)
./run.sh spider_velox CADETS_E3 --project=spider_ablations --experiment=run_n_times --featurization.feat_training.spider.spider.student_only_mode=student_signature --exp=dataset_ablation_student_signature

# Student-only: predict entity class directly (no teacher)
./run.sh spider_velox CADETS_E3 --project=spider_ablations --experiment=run_n_times --featurization.feat_training.spider.spider.student_only_mode=student_class --exp=dataset_ablation_student_class

# Tokenizer: pure BPE instead of domain-specific pre-tokenization + BPE
./run.sh spider_velox CADETS_E3 --project=spider_ablations --experiment=run_n_times --featurization.feat_training.spider.tokenizer.mode=bpe_only --exp=dataset_ablation_bpe_only
```

## Code structure
- [pidsmaker/spider/](pidsmaker/spider/): SPIDER and the other entity encoders. [models/spider.py](pidsmaker/spider/models/spider.py) defines the GNN teacher, the T5 text student and the distillation losses, [pretrain/pretrain_spider.py](pidsmaker/spider/pretrain/pretrain_spider.py) the distillation training loop, and [data/](pidsmaker/spider/data/) the tokenizer, neighborhood sampling and behavior signatures.
- [feat_inference_spider.py](pidsmaker/featurization/feat_inference_methods/feat_inference_spider.py): embeds every entity of the target dataset with the pretrained encoder, optionally after continued pretraining.
- [config/pretrained/spider/](config/pretrained/spider/): `spider.yml` holds all SPIDER hyperparameters, and each `spider_<pids>.yml` plugs SPIDER into a PIDS.
- [spider_scripts/](spider_scripts/): analysis scripts, such as the entity clustering evaluation in `eval_spider.py`.
