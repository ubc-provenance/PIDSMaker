# Pretrained Encoders

Most PIDSs turn the label of each entity (process command line, file path, IP address and port) into a vector with a model trained on the target dataset only, such as `word2vec` or `doc2vec`.
PIDSMaker can instead use an encoder **pretrained** on provenance data: [SPIDER](#spider), [CyberGFM](https://arxiv.org/abs/2601.05988), masked language models, self-supervised GNNs, or general-purpose language models such as GPT-2 and Llama.

All pretrained encoders share one featurization method, `featurization.used_method: pretrained`, and `featurization.pretrained.model_type` picks the encoder.
They can be used in two ways:

- **As a frozen featurizer** (default): the encoder embeds the entities of the target dataset, and these embeddings become the node features of any PIDS. The detector itself is unchanged.
- **As a detector**: the encoder is [fine-tuned on the target dataset](#fine-tuning-the-encoder-as-a-detector) and scores edges directly, replacing the GNN.

## Available encoders

| `model_type` | Encoder |
|---|---|
| `spider` | [SPIDER](#spider) (NeurIPS 2026): a T5 encoder distilled from a GNN teacher. Pretrained weights available |
| `behavior_cluster`, `gnn_distill` | Earlier variants of SPIDER's distillation |
| `bert` | BERT trained with masked language modeling on random walks, as in CyberGFM |
| `roberta`, `modernbert`, `ropebert`, `logbert` | Variants of BERT: RoBERTa head, ModernBERT, RoPE, and LogBERT's hypersphere loss |
| `llama` | Decoder-only model trained from scratch with next-token prediction on random walks |
| `gpt2_pretrained` | GPT-2 (`model_size`: `small`, `medium`, `large` or `xl`), fine-tuned on entity labels |
| `llama3_pretrained` | Llama 3.2 (`model_size`: `1b` or `3b`), fine-tuned on entity labels. Requires `huggingface-cli login`, since Llama is gated |
| `opt_pretrained` | OPT-1.3B, fine-tuned on entity labels |
| `deepwalk`, `node2vec` | Word2Vec on random walks over entity labels (no transformer) |
| `graphmae`, `gae`, `dgi` | Self-supervised GNNs: masked feature reconstruction, link prediction and mutual information |

[`config/pretrained/README.md`](https://github.com/ubc-provenance/PIDSMaker/blob/main/config/pretrained/README.md) describes each encoder and its options, and [`config/pretrained/pretrained.yml`](https://github.com/ubc-provenance/PIDSMaker/blob/main/config/pretrained/pretrained.yml) lists all options with their default values.

## Using an encoder in a PIDS

One config is provided per PIDS. Each one includes the PIDS config and `pretrained.yml`, so the detector stays the same and only its node features change.

| Config | PIDS |
|---|---|
| `pretrained_velox` | Velox |
| `pretrained_orthrus` | Orthrus (non-snooped) |
| `pretrained_kairos` | Kairos |
| `pretrained_magic` | MAGIC |
| `pretrained_flash` | Flash |
| `pretrained_nodlink` | NodLink |
| `pretrained_threatrace` | ThreaTrace |

They use SPIDER by default. Set `featurization.pretrained.model_type` to use another encoder, for example GPT-2 or DeepWalk in Velox:

```shell
python pidsmaker/main.py pretrained_velox CADETS_E3 --featurization.pretrained.model_type=gpt2_pretrained --featurization.pretrained.model_size=small
python pidsmaker/main.py pretrained_velox CADETS_E3 --featurization.pretrained.model_type=deepwalk
```

These configs also enable [`training.stable_optim`](instability.md#reducing-instability).

### Pretraining

Without pretrained weights, the encoder is first pretrained on the benign data of the datasets listed in `featurization.pretrained.pretrain_datasets`: by default `PROVENANCE_BENIGN`, `TRACE_E3`, `TRACE_E5`, `CADETS_E5`, `CLEARSCOPE_E3` and `optc_h201`. Each of them must be installed as a database.
`PROVENANCE_BENIGN` is a Linux audit corpus that you collect yourself (see [Datasets](../datasets.md#provenance_benign)).

The trained encoder is saved in the `stored_models/` folder of the run's featurization artifacts.
Pass that folder with `--featurization.pretrained.weights_path` to reuse it in other runs without pretraining again. The path must be absolute, because its files are symlinked into each run's artifact folder.

The main options are:

- `featurization.pretrained.model_size`: `tiny`, `mini`, `med` or `baseline` for the encoders trained from scratch, or the variant of GPT-2, Llama and OPT.
- `featurization.pretrained.training`: token budget (`pretrain_tokens`, `warmup_tokens`), `batch_size` and `lr`.
- `featurization.pretrained.tokenizer`: BPE vocabulary size and maximum label length (not used by GPT-2, Llama and OPT, which have their own tokenizers).
- `featurization.pretrained.walks`: random walks used by the masked language models and DeepWalk.

All options are listed in the [featurization arguments](../config/featurization.md).

## SPIDER

SPIDER (NeurIPS 2026) is our pretrained entity encoder.
During pretraining, a GNN teacher learns an embedding of each entity from its neighborhood in the graph, and a small T5 encoder is distilled to predict the teacher's embedding from the entity's label alone.
At detection time, only the T5 encoder is used, so it embeds entities from their label only, including entities never seen during pretraining.

### Pretrained weights

Download the weights used in the paper ([Google Drive](https://drive.google.com/file/d/1NlUPQJTEehhe3cAOjlOq_wnPYjV-MMZ9/view?usp=drive_link), 21 MB) from the root of the repository, inside the container:

```shell
gdown 1NlUPQJTEehhe3cAOjlOq_wnPYjV-MMZ9 -O SPIDER-weights.tar.gz
mkdir -p weights && tar -xzf SPIDER-weights.tar.gz -C weights
(cd weights/spider && sha256sum -c SHA256SUMS)
```

Then pass the folder (`/home/pids/weights/spider` in the container) to any `pretrained_<pids>` config:

```shell
python pidsmaker/main.py pretrained_velox CADETS_E3 --featurization.pretrained.weights_path=/home/pids/weights/spider
```

The folder contains:

| File | Content |
|---|---|
| `pretrain_mini.pt` | T5 encoder (`model_size: mini`) |
| `tokenizer.pt` | BPE tokenizer of entity labels |
| `gnn_teacher_mini_best.pt` | GNN teacher, only needed to [continue pretraining](#continued-pretraining-on-the-target-dataset) |
| `behavior_vocab.txt`, `edge_type2idx.json` | Behavior labels and edge types of the teacher, only needed to continue pretraining |
| `spider_log_mini.txt` | Pretraining loss log |

### Continued pretraining on the target dataset

With `feat_inference.continue_pretrain: True`, SPIDER is further pretrained on the benign training data of the target dataset before embedding its entities.
This needs the teacher files of the weights folder.
`feat_inference.continue_pretrain_epochs` (10 in `pretrained.yml`) and `feat_inference.continue_pretrain_lr_factor` (0.1, relative to the pretraining learning rate) control this step.

```shell
python pidsmaker/main.py pretrained_velox CADETS_E3 --featurization.pretrained.weights_path=/home/pids/weights/spider --feat_inference.continue_pretrain=True
```

### Renaming attack

`feat_inference.rename_attack` evaluates robustness to an attacker who renames malicious processes to benign names.
At inference, the test-set entities whose label matches one of `entities` (exact match, or path ending with `/<entity>`) get the embedding of a benign name instead.
The benign name is either drawn from the `top_k` most frequent labels of the training set, or fixed with `target_label` (when `top_k: 0`).

For example, on CADETS_E3, renaming its attack processes to frequent benign names:

```shell
python pidsmaker/main.py pretrained_velox CADETS_E3 --featurization.pretrained.weights_path=/home/pids/weights/spider \
    --feat_inference.rename_attack.enabled=True \
    --feat_inference.rename_attack.entities="main, pEja72mA, XIM, tmux-1002, font, sendmail" \
    --feat_inference.rename_attack.target_node_type=subject \
    --feat_inference.rename_attack.top_k=100
```

### Checkpoint selection

During pretraining, `spider` and `behavior_cluster` are evaluated after each epoch on the 528 labeled entities of `config/eval/eval_set_clustering.json`: a k-nearest-neighbors classifier must recover the role of each entity (browser, database, crypto, …) from its embedding.
The checkpoint with the best accuracy is kept (`pretrain_<size>_best.pt`, and `gnn_teacher_<size>_best.pt` for SPIDER's teacher), and it is the one used at inference.
The other encoders keep their last checkpoint.

### Reproducing the paper

The [`spider` branch](https://github.com/ubc-provenance/PIDSMaker/tree/spider) contains the exact code used for the paper, and its README lists the commands of every experiment.
With this version, the Velox results of the paper are reproduced by the commands below, run from `scripts/` after downloading the [weights](#pretrained-weights).
`--experiment=run_n_times` trains 5 seeds and logs the mean and standard deviation to W&B (`./run_local.sh` runs without W&B).
Results can differ slightly from the paper because of newer library versions.

```shell
cd scripts
ARGS="--experiment=run_n_times --featurization.pretrained.weights_path=/home/pids/weights/spider"

# Frozen encoder
./run.sh pretrained_velox CADETS_E3 $ARGS --training.num_epochs=22
./run.sh pretrained_velox THEIA_E3 $ARGS
./run.sh pretrained_velox THEIA_E5 $ARGS
./run.sh pretrained_velox CLEARSCOPE_E5 $ARGS --training.node_hid_dim=128 --training.node_out_dim=128
./run.sh pretrained_velox optc_h051 $ARGS --training.node_hid_dim=256 --training.node_out_dim=256

# Continued pretraining (fine-tuned SPIDER in the paper)
./run.sh pretrained_velox CADETS_E3 $ARGS --feat_inference.continue_pretrain=True --feat_inference.continue_pretrain_epochs=10
./run.sh pretrained_velox THEIA_E3 $ARGS --feat_inference.continue_pretrain=True --feat_inference.continue_pretrain_epochs=10
./run.sh pretrained_velox THEIA_E5 $ARGS --feat_inference.continue_pretrain=True --feat_inference.continue_pretrain_epochs=5
./run.sh pretrained_velox CLEARSCOPE_E5 $ARGS --feat_inference.continue_pretrain=True --feat_inference.continue_pretrain_epochs=1
./run.sh pretrained_velox optc_h051 $ARGS --feat_inference.continue_pretrain=True --feat_inference.continue_pretrain_epochs=10
```

## Fine-tuning the encoder as a detector

With `training.used_method: pretrained`, the GNN detector is bypassed: the pretrained encoder is fine-tuned on the target dataset and scores edges directly.
`training.pretrained.finetune_mode` picks the fine-tuning objective:

| `finetune_mode` | Objective |
|---|---|
| `lp` | Link prediction: an edge's score is its reconstruction error |
| `mlm` | Masked-token reconstruction of the walks around an edge |
| `cls` | Ranking benign walks against perturbed ones (no labels) |
| `cls_attack` | Classification with walks from ground-truth attack edges (supervised) |
| `edge_cls` | Edge type classification from the endpoint embeddings |
| `tgn` | A TGN trained on top of the frozen embeddings (options under `training.pretrained.tgn`) |
| `perplexity` | No fine-tuning: next-token perplexity of a causal model (`model_type: llama`) |

[`config/pretrained/cybergfm/cybergfm.yml`](https://github.com/ubc-provenance/PIDSMaker/blob/main/config/pretrained/cybergfm/cybergfm.yml) implements the CyberGFM baseline this way, with BERT pretraining followed by link prediction:

```shell
python pidsmaker/main.py cybergfm CADETS_E3
```
