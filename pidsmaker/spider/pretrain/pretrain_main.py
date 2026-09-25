"""SPIDER pretraining on provenance graphs.

Thin dispatcher: loads datasets and samplers, then delegates to the
appropriate pretrain_* module based on model_type.
"""

import os

import torch

from pidsmaker.utils.utils import log, log_start

from .pretrain_common import (
    _load_dataset_for_pretrain,
    _load_and_merge_graphs,
    _build_comprehensive_walk_corpus,
)
from ..data.sampler import ProvenanceWalkSampler

def main(cfg):
    log_start(__file__)

    # ── Config ──────────────────────────────────────────────────────────
    spider_cfg = cfg.featurization.spider

    # Model config
    model_type = spider_cfg.model_type
    model_size = spider_cfg.model_size

    # Tokenizer config (nested)
    tokenizer_cfg = spider_cfg.tokenizer
    bpe_vocab_size = tokenizer_cfg.bpe_vocab_size
    max_seq_len = tokenizer_cfg.max_seq_len
    tokenizer_mode = tokenizer_cfg.mode
    normalize_netflow_ips = tokenizer_cfg.normalize_netflow_ips
    canonicalize_neighbors = tokenizer_cfg.canonicalize_neighbors
    spider_path = getattr(spider_cfg, 'spider_path', None)

    # Walk config — path mode
    walks_cfg = spider_cfg.walks
    walk_length = walks_cfg.walk_length
    num_walks = walks_cfg.num_walks
    time_weight = walks_cfg.time_weight
    half_life = walks_cfg.half_life
    graph_context_mode = spider_cfg.graph_context_mode
    random_walk_start = walks_cfg.random_walk_start
    diversity_weight = walks_cfg.diversity_weight

    # Training hyperparameters (MLM-specific knobs like mask_rate_* and
    # logbert.hvm_weight are read inside pretrain_mlm.py — they have no
    # meaning for the other branches.)
    training_cfg = spider_cfg.training
    total_tokens = int(float(training_cfg.pretrain_tokens))
    warmup_tokens = int(float(training_cfg.warmup_tokens))
    batch_size = training_cfg.batch_size
    lr = training_cfg.lr
    scheduler_type = training_cfg.scheduler

    device = "cuda" if torch.cuda.is_available() and not cfg._use_cpu else "cpu"
    print(device)

    # ── Output directory ────────────────────────────────────────────────
    out_dir = cfg.featurization._model_dir
    os.makedirs(out_dir, exist_ok=True)

    # ── Symlink available pretrained artifacts into out_dir ─────────────
    if spider_path:
        candidate_files = [
            "tokenizer.pt", "corpus.pt", "behavior_vocab.txt",
            f"pretrain_{model_size}.pt", f"pretrain_{model_size}_best.pt",
        ]
        linked = 0
        for fname in candidate_files:
            src = os.path.join(spider_path, fname)
            dst = os.path.join(out_dir, fname)
            if os.path.exists(src) and not os.path.exists(dst):
                os.symlink(src, dst)
                linked += 1
        if linked:
            log(f"Symlinked {linked} artifact(s) from {spider_path} → {out_dir}")

        # If model checkpoint already exists, skip training entirely
        has_checkpoint = (
            os.path.exists(os.path.join(out_dir, f"pretrain_{model_size}_best.pt"))
            or os.path.exists(os.path.join(out_dir, f"pretrain_{model_size}.pt"))
        )
        if has_checkpoint:
            log(f"Pretrained checkpoint found in {spider_path}, skipping training")
            return

    # ── Parse dataset list ────────────────────────────────────────────
    pretrain_datasets = spider_cfg.pretrain_datasets
    if pretrain_datasets:
        dataset_names = [d.strip() for d in pretrain_datasets.split(",")]
    else:
        dataset_names = [cfg.dataset.name]

    if dataset_names:
        log(f"Pretraining on datasets: {dataset_names}")

    # ── Load per-dataset data ────────────────────────────────────────
    combined_indexid2msg = {}
    combined_train_nodes = set()
    sampler_pairs = []  # [(sampler, indexid2msg), ...]
    all_val_data = []   # [(val_paths, indexid2msg), ...]

    if spider_path:
        # ── Load pre-built corpus (samplers + indexid2msg per dataset) ──
        log(f"Loading pre-built corpus from {spider_path}")
        corpus_state = torch.load(os.path.join(spider_path, "corpus.pt"),
                                  map_location="cpu", weights_only=False)
        for ds_entry in corpus_state["datasets"]:
            ds_name = ds_entry["name"]
            ds_indexid2msg = ds_entry["indexid2msg"]
            sampler = ProvenanceWalkSampler.from_state(ds_entry["sampler_state"])
            sampler_pairs.append((sampler, ds_indexid2msg))
            for k, v in ds_indexid2msg.items():
                combined_indexid2msg[f"{ds_name}::{k}"] = v
                combined_train_nodes.add(f"{ds_name}::{k}")
            log(f"  {ds_name}: {len(sampler.nodes):,} nodes restored")
        log(f"Corpus loaded: {len(sampler_pairs)} dataset(s), "
            f"{len(combined_indexid2msg):,} entities")

    else:
        # ── Load graphs from scratch ──────────────────────────────────
        for ds_name in dataset_names:
            ds_indexid2msg, ds_train_nodes, train_paths, val_paths = \
                _load_dataset_for_pretrain(cfg, ds_name)

            for k, v in ds_indexid2msg.items():
                combined_indexid2msg[f"{ds_name}::{k}"] = v
                if k in ds_train_nodes:
                    combined_train_nodes.add(f"{ds_name}::{k}")

            train_graphs = _load_and_merge_graphs(train_paths, graph_context_mode)
            log(f"{ds_name}: loaded {len(train_graphs)} graph(s) from {len(train_paths)} files")

            # Node2Vec: pass p/q to sampler for biased walks
            n2v_kwargs = {}
            if model_type == "node2vec":
                n2v_kwargs = dict(
                    node2vec_p=spider_cfg.node2vec.p,
                    node2vec_q=spider_cfg.node2vec.q,
                )

            for g in train_graphs:
                sampler = ProvenanceWalkSampler(
                    g, walk_length=walk_length, num_walks=num_walks,
                    time_weight=time_weight, half_life=half_life,
                    random_walk_start=random_walk_start,
                    diversity_weight=diversity_weight,
                    **n2v_kwargs,
                )
                sampler_pairs.append((sampler, ds_indexid2msg))

            if val_paths:
                all_val_data.append((val_paths, ds_indexid2msg))

        # ── Save corpus for future reuse ──────────────────────────────
        corpus_save_path = os.path.join(out_dir, "corpus.pt")
        log(f"Saving corpus to {corpus_save_path} ...")
        corpus_datasets = []
        for ds_name, (sampler, ds_indexid2msg) in zip(dataset_names, sampler_pairs):
            corpus_datasets.append({
                "name": ds_name,
                "indexid2msg": ds_indexid2msg,
                "sampler_state": sampler.save_state(),
            })
        torch.save({"datasets": corpus_datasets}, corpus_save_path)
        log(f"Corpus saved ({len(corpus_datasets)} datasets)")

    # ── GraphMAE ──────────────────────────────────────────────────────
    if model_type == "graphmae":
        from .pretrain_graphmae import pretrain_graphmae
        pretrain_graphmae(
            cfg, spider_cfg, sampler_pairs, combined_indexid2msg, out_dir, device, batch_size,
            model_size, max_seq_len, bpe_vocab_size, tokenizer_mode,
            normalize_netflow_ips, canonicalize_neighbors, spider_path,
        )
        return

    # ── GAE (Graph Autoencoder) ──────────────────────────────────────
    if model_type == "gae":
        from .pretrain_gae import pretrain_gae
        pretrain_gae(
            cfg, spider_cfg, sampler_pairs, combined_indexid2msg, out_dir, device, batch_size,
            model_size, max_seq_len, bpe_vocab_size, tokenizer_mode,
            normalize_netflow_ips, canonicalize_neighbors, spider_path,
        )
        return

    # ── DGI (Deep Graph Infomax) ─────────────────────────────────────
    if model_type == "dgi":
        from .pretrain_dgi import pretrain_dgi
        pretrain_dgi(
            cfg, spider_cfg, sampler_pairs, combined_indexid2msg, out_dir, device, batch_size,
            model_size, max_seq_len, bpe_vocab_size, tokenizer_mode,
            normalize_netflow_ips, canonicalize_neighbors, spider_path,
        )
        return

    # ── GNN Distillation ──────────────────────────────────────────────
    if model_type == "gnn_distill":
        from .pretrain_gnn_distill import pretrain_gnn_distill
        pretrain_gnn_distill(
            cfg, spider_cfg, sampler_pairs, combined_indexid2msg, out_dir, device,
            model_size, max_seq_len, bpe_vocab_size, tokenizer_mode,
            normalize_netflow_ips, canonicalize_neighbors, spider_path,
            total_tokens, warmup_tokens, batch_size, lr,
        )
        return

    # ── SPIDER ────────────────────────────────────────────────────────
    if model_type == "spider":
        from .pretrain_spider import pretrain_spider
        pretrain_spider(
            cfg, spider_cfg, sampler_pairs, combined_indexid2msg, out_dir, device,
            model_size, max_seq_len, bpe_vocab_size, tokenizer_mode,
            normalize_netflow_ips, canonicalize_neighbors, spider_path,
            total_tokens, warmup_tokens, batch_size, lr,
        )
        return

    # ── Behavior Cluster ──────────────────────────────────────────────
    if model_type == "behavior_cluster":
        from .pretrain_behavior_cluster import pretrain_behavior_cluster
        pretrain_behavior_cluster(
            cfg, spider_cfg, sampler_pairs, combined_indexid2msg, out_dir, device,
            model_size, max_seq_len, bpe_vocab_size, tokenizer_mode,
            normalize_netflow_ips, canonicalize_neighbors, spider_path,
            total_tokens, warmup_tokens, batch_size, lr,
        )
        return

    # ── HF pretrained models (no walks — fine-tune on entity labels) ─────
    from ..models.hf_pretrained import HF_MODEL_TYPES
    if model_type in HF_MODEL_TYPES:
        from .pretrain_hf import pretrain_hf_model
        log(f"HF pretrained mode: skipping walk sampling, fine-tuning on "
            f"{len(combined_indexid2msg):,} entity labels")
        pretrain_hf_model(
            cfg, model_type, model_size, combined_indexid2msg, out_dir, device,
            total_tokens, warmup_tokens, batch_size, lr, scheduler_type,
            max_seq_len=max_seq_len,
        )
        return

    # ── Build walk corpus (before tokenizer to use walk-frequency distribution) ──
    walk_corpus = _build_comprehensive_walk_corpus(
        sampler_pairs, num_walks=num_walks, walk_length=walk_length,
        time_weight=time_weight, half_life=half_life, max_retries=50,
    )

    # ── DeepWalk / Node2Vec ───────────────────────────────────────────
    if model_type in ("deepwalk", "node2vec"):
        from .pretrain_deepwalk import pretrain_deepwalk
        pretrain_deepwalk(cfg, spider_cfg, walk_corpus, out_dir, model_type)
        return

    # ── MLM / Causal LM ──────────────────────────────────────────────
    from .pretrain_mlm import pretrain_mlm
    pretrain_mlm(
        cfg, spider_cfg, walk_corpus, sampler_pairs, combined_indexid2msg, all_val_data,
        out_dir, device, model_type, model_size, max_seq_len, bpe_vocab_size, tokenizer_mode,
        normalize_netflow_ips, canonicalize_neighbors, spider_path,
        total_tokens, warmup_tokens, batch_size, lr, scheduler_type,
    )
