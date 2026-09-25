# Pretrained encoder configs

Configs in this folder use `featurization.feat_training.used_method: spider`,
which trains (or loads) a graph foundation model that produces static node
embeddings consumed by the downstream GNN detector.

One subfolder per pretrained model:

| Folder | Content |
|---|---|
| `spider/` | `spider.yml` (all SPIDER hyperparameters) and `spider_<pids>.yml` (SPIDER plugged into a PIDS) |
| `cybergfm/` | the CyberGFM recipe: BERT pretraining (`model_type: bert`) with fine-tuned detection (`cybergfm.yml`) |

## Invocation

Any file in this folder is invokable by its bare name — the loader searches
`config/` recursively:

```
python -m pidsmaker.main spider         CADETS_E3   # config/pretrained/spider/spider.yml
python -m pidsmaker.main spider_velox   CADETS_E3   # config/pretrained/spider/spider_velox.yml
```

## Config structure

Inside `featurization.feat_training.spider`:

| Section | Scope |
|---|---|
| top-level keys (`model_type`, `model_size`, `graph_context_mode`, `pretrain_datasets`, `spider_path`) | shared across every `model_type` |
| `training:` | token-budget loop (`pretrain_tokens`, `warmup_tokens`, `batch_size`, `lr`, `scheduler`) — used by MLM, GNN token-budget, and HF pretrained |
| `mlm:` | MLM masking (`mask_rate_*`, `mask_edge_type`) plus per-architecture sub-blocks (`modernbert`, `ropebert`, `llama`, `logbert`) |
| `walks:` | random-walk corpus (`walk_length`, `num_walks`, `time_weight`, …) — drives MLM/walk-embedding; ignored by GNN-SSL / GNN-token-budget / HF |
| `tokenizer:` | BPE tokenizer (`bpe_vocab_size`, `max_seq_len`, `mode`, …) — used by everything except `deepwalk`/`node2vec` and the HF pretrained models (which use HF tokenizers) |
| `<model_name>:` (e.g. `spider`, `graphmae`, `deepwalk`) | only read when `model_type` matches |

The default config `spider.yml` documents every section inline.

## Model types

The `model_type` key picks the pretraining objective and architecture.
Models are grouped below by training mode.

### MLM walk-based

Tokenize random walks over the provenance graph, mask a fraction of tokens,
and train an encoder to predict them. At inference, the encoder produces
contextualized embeddings of each entity's local walks.

- **`bert`** — HuggingFace BERT with MLM. The baseline encoder; absolute
  position embeddings, full global attention, LayerNorm. Matches CyberGFM's
  original pretraining recipe.
- **`roberta`** — Same backbone as BERT but with RoBERTa's LM head
  (dense → GELU → LayerNorm → projection) and offset position embeddings.
  A drop-in MLM variant with a slightly stronger decoder.
- **`modernbert`** — Modernized BERT: RoPE, RMSNorm pre-norm, GeGLU,
  alternating global / local sliding-window attention, no bias terms.
  Better long-context behavior and faster training.
  Tunables: `mlm.modernbert.global_attn_every_n_layers`, `mlm.modernbert.local_attention_window`.
- **`ropebert`** — BERT with RoPE + RMSNorm + GeGLU but full global attention
  on every layer. Strips ModernBERT's sliding-window complexity while keeping
  the modern building blocks. Tunable: `mlm.ropebert.rope_theta`.
- **`llama`** — Decoder-only causal LM (HF `LlamaForCausalLM`, randomly
  initialized). Replaces masked-token prediction with next-token prediction,
  giving signal on every token and enabling perplexity-based scoring at
  inference without fine-tuning. Tunable: `mlm.llama.rope_theta`.
- **`logbert`** — BERT MLM augmented with a hypersphere-volume-minimization
  loss (LogBERT). The encoder is pushed to map all benign sequences inside
  a tight hypersphere; anomalies fall outside it. Tunable: `mlm.logbert.hvm_weight`.

### Walk embedding (shallow)

No tokenizer, no transformer — train Word2Vec directly on walks over
node-label strings. Cheap, transductive at the *label* level (any node whose
`(ntype, nlabel)` was seen at training gets an embedding).

- **`deepwalk`** — Word2Vec Skip-gram on uniform random walks. Tunables under
  `deepwalk:` (`window`, `epochs`, `min_count`, `workers`).
- **`node2vec`** — DeepWalk with biased second-order walks controlled by
  `node2vec.p` (return) and `node2vec.q` (in-out: `q > 1` ≈ BFS,
  `q < 1` ≈ DFS). Shares the `deepwalk:` block for training.

### GNN self-supervised

Sample temporal 1-hop neighborhoods, encode the center + neighbors with a
T5 token encoder, then run a GAT-based GNN. Trained with their own `epochs`
/ `lr` (the top-level `training:` block is ignored). The `walks:` sampler is
built but only `nodes` and `backward_adj` are read.

- **`graphmae`** — Masked Graph Autoencoder (Hou et al., KDD 2022).
  Masks a fraction of node features and reconstructs them via a lightweight
  GAT decoder using Scaled Cosine Error. Tunables under `graphmae:`
  (`num_layers`, `num_heads`, `decoder_num_layers`, `mask_rate`,
  `replace_rate`, `neighborhood_min/max`, `epochs`, `lr`).
- **`gae`** — Graph Autoencoder (Kipf & Welling, 2016). Encodes
  neighborhoods with a GAT, decodes the adjacency via inner-product, trained
  with BCE on positive edges + negative samples. Tunables under `gae:`.
- **`dgi`** — Deep Graph Infomax (Velickovic et al., ICLR 2019). Maximizes
  mutual information between node embeddings and a graph-level summary by
  contrasting against shuffled (corrupted) features via a bilinear
  discriminator. Tunables under `dgi:`.

### GNN token-budget

A T5 student encoder is trained against a structural target derived from
behavior signatures and/or a GNN teacher. Honors the top-level `training:`
block (`pretrain_tokens`, `warmup_tokens`, `batch_size`, `lr`). At inference,
only the T5 student is used.

- **`behavior_cluster`** — T5 encoder + two heads (multi-label BCE on the
  behavior signature vector + InfoNCE contrastive). The "text-only" baseline:
  the encoder sees only entity labels and must predict the *kind* of edges
  the entity has in the graph. Tunables under `behavior_cluster:` (BCE
  weight, contrastive weight, temperature, P×K batching, `bce_mode`,
  `contrastive_target`).
- **`gnn_distill`** — T5 student is aligned via SCE loss to embeddings
  produced by a GNN teacher that runs on neighborhoods of EMA-student
  features. The center node is masked for the GNN's self-supervised signal.
  Tunables under `gnn_distill:`.
- **`spider`** — Successor to `behavior_cluster` and `gnn_distill`.
  A GNN teacher trained with supervised contrastive loss on entity-class
  centers; the T5 student is distilled (SCE) to match the teacher centers.
  Ablation knobs (`teacher_loss`, `teacher_data`, `distill_loss`,
  `student_only_mode`) make it a superset of the other two. Tunables under
  `spider:`.

### HuggingFace pretrained

Load a public causal LM, fine-tune on entity labels with next-token
prediction (no walks, no graph), and use mean-pooled hidden states as
embeddings. `emb_dim` is overridden by the model's hidden size. The
tokenizer is the model's HF tokenizer (the `tokenizer:` block is ignored).

- **`gpt2_pretrained`** — GPT-2 small/medium/large/xl. `model_size` picks
  the variant. Tunable: `gpt2_pretrained.max_seq_len`.
- **`llama3_pretrained`** — Llama 3.2 (1b / 3b). Requires
  `huggingface-cli login` because Llama is gated.
- **`opt_pretrained`** — Meta OPT-1.3b. Useful as a same-size foil to Llama.

## Picking a model_type

- **Strong default**: `spider` — best results on the downstream benchmark; pair
  with a non-trivial `pretrain_datasets`.
- **Fast iteration / sanity baseline**: `deepwalk` (no transformer, minutes
  not hours) or `bert` (the original CyberGFM recipe).
- **Long context / large vocab**: `modernbert` or `ropebert` over `bert`.
- **Streaming or perplexity-based detection**: `llama` (causal) or `logbert`
  (built-in anomaly score).
- **Pure neighborhood SSL**: `graphmae` (masked reconstruction), `gae`
  (link prediction), `dgi` (mutual information). Use these to compare
  against `gnn_distill`/`spider`.
- **Foundation-from-PLM**: `gpt2_pretrained` / `llama3_pretrained` /
  `opt_pretrained` to measure how much benefit pretrained English LM weights
  give on the entity-label task.

## Using the SPIDER model in `gnn_training`

The SPIDER model produced by `feat_training` can be consumed in two ways
by the downstream detector:

1. **As a frozen featurizer** (default) — set
   `detection.gnn_training.used_method: default`. The pretrained encoder
   produces static per-node embeddings (mean-pooled walks for MLM/walk models;
   raw encoder output for GNN-SSL / GNN token-budget / HF). These embeddings
   land in `node_emb` and feed whatever GNN encoder the baseline config
   specifies. This is what every `spider_<baseline>.yml` file does — see
   `spider_velox.yml`, `spider_orthrus.yml`, etc.: they inherit a
   baseline detector (`velox`, `orthrus_non_snooped`, …) and just swap in
   SPIDER embeddings.

2. **As a fine-tuned scorer** — set
   `detection.gnn_training.used_method: spider`. The pretrained encoder
   is fine-tuned on the target dataset and produces *anomaly scores
   directly*; the baseline GNN detector is bypassed. Configure under
   `detection.gnn_training.spider`. See `cybergfm/cybergfm.yml` for a
   full example. Knobs:

   - **`finetune_mode`** — picks the head + loss. Only `bert`-family
     `model_type`s support all modes; `llama` uses `perplexity`; the GNN
     pretrained models (`spider`, `gnn_distill`, `behavior_cluster`,
     `graphmae`, …) are normally used as featurizers (mode 1), not
     fine-tuned here.

     - `cls` — Ranking-margin classification. Train a CLS head to rank
       anomalous walks above benign ones; uses `finetune_margin`. No labeled
       attacks needed (benign vs synthetic perturbations).
     - `cls_attack` — Same head as `cls`, but trains on real attack walks
       extracted from the labeled attack edges. Requires ground-truth
       attacks; uses `num_attack_walks` per malicious edge.
     - `lp` — Link prediction. Predict whether an (src, dst) pair is a real
       walk continuation; score is reconstruction error. Always trains the
       backbone (`freeze_backbone` ignored).
     - `mlm` — Masked-token reconstruction at detection time. Score = node
       CE + `edge_score_weight` × edge-type CE (requires
       `mlm.mask_edge_type: True` at pretrain).
     - `tgn` — Replace the MLM head with a TGN (Temporal Graph Network)
       trained on edge-level objectives. Fully separate pipeline; configure
       under `spider.tgn` (`objective`, `memory_dim`, `edge_emb_dim`,
       `time_dim`, …). Useful when you want temporal memory on top of static
       SPIDER embeddings.
     - `edge_cls` — Edge-type classification, predict the edge type given
       endpoint embeddings.
     - `perplexity` — No fine-tuning. The pretrained causal LM (typically
       `llama`) is used as-is; score = next-token perplexity over each
       walk.

   - **`freeze_backbone`** — `True` trains only the head (fast, best when
     pretraining was strong); `False` updates the encoder too.
     Force-ignored when `finetune_mode=lp`.
   - **`finetune_epochs` / `finetune_lr` / `finetune_walk_len`** — standard
     training knobs for the fine-tune stage.
   - **`num_inference_walks` / `inference_batch_size`** — at inference each
     edge is scored as the average across `num_inference_walks` walks.
   - **`edge_score_weight`** — only meaningful for `lp` and `mlm`; weights
     the edge-type CE component (requires `mlm.mask_edge_type=True`).

### Picking between mode 1 and mode 2

- Use **mode 1** (frozen featurizer) when comparing SPIDER pretraining
  against other featurizers under a fixed detector — i.e. when the question
  is *"do these embeddings help?"*. All `spider_<baseline>.yml` configs
  in `spider/` are mode 1.
- Use **mode 2** (`used_method: spider`) when the SPIDER model is
  itself the detector — i.e. when the question is *"can pretraining replace
  the GNN?"*. Pick `finetune_mode` from the model family:
  encoder MLM → `cls`/`lp`/`mlm`, causal LM → `perplexity`,
  with-attacks-supervision → `cls_attack`, temporal-memory experiments →
  `tgn`.

## Reusing a pretrained SPIDER model

Set `spider_path` to a directory containing any subset of:
`tokenizer.pt`, `corpus.pt`, `behavior_vocab.txt`,
`pretrain_<model_size>.pt`, `pretrain_<model_size>_best.pt`. Present files
are symlinked into the run's model dir; if a checkpoint exists, pretraining
is skipped. Works with any `model_type`.
