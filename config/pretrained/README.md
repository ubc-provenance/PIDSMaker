# Pretrained encoder configs

Configs in this folder use `featurization.used_method: pretrained`: a pretrained encoder
embeds the label of each entity, and these embeddings are the node features of the detector.
See the [documentation](https://ubc-provenance.github.io/PIDSMaker/features/pretrained_encoders/) for a full guide.

| Config | Content |
|---|---|
| `pretrained.yml` | all options of the pretrained encoders (SPIDER by default) |
| `pretrained_<pids>.yml` | a pretrained encoder plugged into a PIDS |
| `cybergfm/` | the CyberGFM recipe: BERT pretraining (`model_type: bert`) with fine-tuned detection (`cybergfm.yml`) |

Configs are invoked by their name. `pretrained.yml` is included by the `pretrained_<pids>.yml` configs, which are the ones to run:

```
python pidsmaker/main.py pretrained_velox CADETS_E3
python pidsmaker/main.py cybergfm CADETS_E3
```

## Config structure

Inside `featurization.pretrained`:

| Key | Used by |
|---|---|
| `model_type`, `model_size`, `graph_context_mode`, `pretrain_datasets`, `weights_path` | all encoders |
| `training` (`pretrain_tokens`, `warmup_tokens`, `batch_size`, `lr`, `scheduler`) | all encoders except `deepwalk`, `node2vec`, `graphmae`, `gae` and `dgi`, which have their own `epochs` |
| `mlm` | masked language models and `llama` |
| `walks` | masked language models, `llama`, `deepwalk` and `node2vec` |
| `tokenizer` | all encoders except `deepwalk`, `node2vec` and the HF models |
| `<model_type>` (e.g. `spider`, `graphmae`, `deepwalk`) | only that `model_type` |

## Model types

### Masked language models

Trained on tokenized random walks over the provenance graph.

- **`bert`**: BERT with masked language modeling, as in CyberGFM.
- **`roberta`**: BERT with RoBERTa's language modeling head.
- **`modernbert`**: ModernBERT (RoPE, alternating global and sliding-window attention). Options: `mlm.modernbert`.
- **`ropebert`**: BERT with RoPE and global attention on every layer. Options: `mlm.ropebert`.
- **`logbert`**: BERT with LogBERT's hypersphere loss. Options: `mlm.logbert`.
- **`llama`**: decoder-only model trained from scratch with next-token prediction. Options: `mlm.llama`.

### Walk embeddings

Word2Vec trained on random walks over entity labels, without tokenizer nor transformer.

- **`deepwalk`**: uniform random walks. Options: `deepwalk`.
- **`node2vec`**: biased walks controlled by `node2vec.p` (return) and `node2vec.q` (in-out). Also uses `deepwalk`.

### Self-supervised GNNs

A T5 encoder embeds each entity and its temporal neighborhood, and a GAT is trained on top of it.

- **`graphmae`**: masked feature reconstruction (GraphMAE). Options: `graphmae`.
- **`gae`**: link prediction (graph autoencoder). Options: `gae`.
- **`dgi`**: mutual information between node and graph embeddings (Deep Graph Infomax). Options: `dgi`.

### Distillation

A T5 encoder is distilled from a GNN teacher or from behavior signatures. Only the T5 encoder is used at inference.

- **`spider`**: SPIDER. A GNN teacher is trained with a supervised contrastive loss on entity classes, and the T5 encoder is distilled from it. Options: `spider`.
- **`behavior_cluster`**, **`gnn_distill`**: earlier variants, from behavior signatures only or from a self-supervised GNN teacher. Options: `behavior_cluster`, `gnn_distill`.

### HF models

A public language model fine-tuned on entity labels with next-token prediction. `emb_dim` is set to the hidden size of the model, and the model's own tokenizer is used.

- **`gpt2_pretrained`**: GPT-2 (`model_size`: `small`, `medium`, `large` or `xl`).
- **`llama3_pretrained`**: Llama 3.2 (`model_size`: `1b` or `3b`). Requires `huggingface-cli login`, since Llama is gated.
- **`opt_pretrained`**: OPT-1.3B.

## Fine-tuning the encoder as a detector

By default, the encoder is a frozen featurizer of the detector configured by the included PIDS config (`training.used_method: default`).
With `training.used_method: pretrained`, the detector is bypassed: the encoder is fine-tuned on the target dataset and scores edges directly (see `cybergfm/cybergfm.yml`).
The options are under `training.pretrained`:

- **`finetune_mode`**:
  - `lp`: link prediction, the score of an edge is its reconstruction error. Always trains the encoder.
  - `mlm`: masked-token reconstruction. The score adds `edge_score_weight` × the edge type loss, which requires `mlm.mask_edge_type: True` during pretraining.
  - `cls`: ranks benign walks against perturbed ones, with margin `finetune_margin`. No labels needed.
  - `cls_attack`: classification with `num_attack_walks` walks per ground-truth attack edge (supervised).
  - `edge_cls`: edge type classification from the endpoint embeddings.
  - `tgn`: a TGN trained on top of the frozen embeddings, with options under `training.pretrained.tgn`.
  - `perplexity`: no fine-tuning, the score is the next-token perplexity of a causal model (`model_type: llama`).
- **`freeze_backbone`**: only train the head (`True`) or also the encoder (`False`). Ignored by `lp`.
- **`finetune_epochs`**, **`finetune_lr`**, **`finetune_walk_len`**: fine-tuning loop.
- **`num_inference_walks`**, **`inference_batch_size`**: each edge is scored as the average over `num_inference_walks` walks.

## Reusing pretrained weights

Set `featurization.pretrained.weights_path` to the absolute path of a folder containing any subset of
`tokenizer.pt`, `corpus.pt`, `behavior_vocab.txt`, `pretrain_<model_size>.pt` and `pretrain_<model_size>_best.pt`.
These files are symlinked into the run's model folder, and pretraining is skipped if a checkpoint is present.
