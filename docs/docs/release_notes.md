# Changelog

````
# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- Pretrained encoders as node featurization for any PIDS (`featurization.used_method: pretrained`, `featurization.pretrained.model_type`): SPIDER (NeurIPS 2026, with pretrained weights), CyberGFM, GPT-2, Llama 3.2 and OPT fine-tuned on entity labels, language models trained on random walks (BERT, RoBERTa, ModernBERT, LogBERT, Llama), self-supervised GNNs (GraphMAE, GAE, DGI), DeepWalk and node2vec. One config per PIDS in `config/pretrained/spider/`, and fine-tuning of the encoder as a detector (`training.used_method: pretrained`).
- `training.stable_optim`: AdamW, warmup and cosine learning rate schedule, and gradient clipping, to reduce the instability between runs.
- `PROVENANCE_BENIGN` pretraining corpus (its collection scripts are on the `spider` branch).
- `predict_edge_supervised` objective: supervised fine-tuning with attack edges (`config/attack_edges/`).
- `hetero_graph_transformer` encoder.
- Options `construction.consistent_edge_types`, `construction.null_label_tokens`, `auto` node label features, `training.fuse_duplicate_edges_training`, `feat_inference.rename_attack`, `percentile` and `fixed_zero` thresholds, and `best_ap@10` model selection.
- `scripts/run_local.sh` to run in the foreground without W&B.

### Changed
- Docker image: torch 1.13.1 → 2.1.2 (CUDA 12.1), PyG 2.5.3 → 2.6.1, gensim 4.4.0, pandas 2.3.3, scipy 1.13.1, networkx 3.2.1, scikit-learn 1.6.1, and `transformers`. The image must be rebuilt.
- Graph construction keeps only printable ASCII characters in labels and stores node ids as strings. Artifact hashes change, and baseline results can shift slightly.
- `run_n_times` reuses existing artifacts instead of deleting them before the first run.

### Fixed
- Time-window and edge-level ground truth were empty on DARPA TC and crashed on OpTC.
- The `rgcn` and `rgcn_per_type` encoders had no default settings.

## [1.0.0] - 2025-06-05
- Initial release
- Systems: Velox, Orthrus, R-Caid, Flash, Kairos, Magic, NodLink, ThreaTrace
- Datasets: CLEARSCOPE_E3, CADETS_E3, THEIA_E3, CLEARSCOPE_E5, THEIA_E5, optc_h201, optc_h501, optc_h051
