"""
SPIDER detection: score val/test edges using the fine-tuned model.

Replaces GNN training+inference for SPIDER. Loads the fine-tuned CLS, LP,
or TGN model, scores all edges in val/test graphs, and writes per-edge anomaly
scores as CSVs in the format expected by node_evaluation.py.

CSV format: loss,srcnode,dstnode,time,edge_type
Directory:  {edge_losses_dir}/{split}/model_epoch_0/{time_interval}.csv
"""

import multiprocessing as mp
import os
from typing import List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from pidsmaker.utils.dataset_utils import get_rel2id
from pidsmaker.utils.utils import (
    get_all_files_from_folders,
    get_indexid2msg,
    log,
    log_start,
    log_tqdm,
    ns_time_to_datetime_US,
)

from .finetune import _collect_edges


def _build_context_graph(graph_paths: List[str]) -> nx.MultiDiGraph:
    """Merge multiple snapshot graphs into one for use as sampler context."""
    merged = nx.MultiDiGraph()
    for path in graph_paths:
        g = torch.load(path, weights_only=False)
        merged.add_nodes_from(g.nodes(data=True))
        merged.add_edges_from(g.edges(data=True, keys=True))
    return merged


def _group_paths_by_day(paths: List[str]) -> dict:
    """Group graph file paths by their parent directory (= day folder)."""
    from collections import defaultdict
    groups = defaultdict(list)
    for p in paths:
        day_folder = os.path.basename(os.path.dirname(p))
        groups[day_folder].append(p)
    return groups
from .models.bert import (
    ProvenanceBERTFineTuneCLS,
    ProvenanceBERTFineTuneLP,
    get_bert_config,
)
from .models.modernbert import (
    ProvenanceModernBERTFineTuneCLS,
    ProvenanceModernBERTFineTuneLP,
    get_modernbert_config,
)
from .models.ropebert import (
    ProvenanceRoPEBERTFineTuneCLS,
    ProvenanceRoPEBERTFineTuneLP,
    get_ropebert_config,
)
from .data.sampler import ProvenanceWalkSampler
from .data.tokenizer_bpe import ProvenanceTokenizerBPE as ProvenanceTokenizer


def _load_finetuned_model(
    pretrain_dir: str,
    ft_dir: str,
    tokenizer: ProvenanceTokenizer,
    model_size: str,
    model_type: str,
    finetune_mode: str,
    freeze_backbone: bool,
    device: str,
    pretrain_cfg=None,
):
    """Load the fine-tuned CLS or LP model, falling back to pretrained weights."""
    if model_type == "ropebert":
        rope_theta = pretrain_cfg.rope_theta
        bert_config = get_ropebert_config(
            tokenizer.vocab_size, model_size, tokenizer.max_seq_len,
            rope_theta=rope_theta,
        )
    elif model_type == "modernbert":
        bert_config = get_modernbert_config(
            tokenizer.vocab_size, model_size, tokenizer.max_seq_len,
            global_attn_every_n_layers=pretrain_cfg.global_attn_every_n_layers,
            local_attention_window=pretrain_cfg.local_attention_window,
        )
    else:
        bert_config = get_bert_config(tokenizer.vocab_size, model_size, tokenizer.max_seq_len)

    # Try fine-tuned checkpoints first (in ft_dir), then pretrained (in pretrain_dir)
    ft_best = os.path.join(ft_dir, f"finetune_{finetune_mode}_{model_size}_best.pt")
    ft_last = os.path.join(ft_dir, f"finetune_{finetune_mode}_{model_size}.pt")
    pt_last = os.path.join(pretrain_dir, f"pretrain_{model_size}.pt")

    checkpoint_path = None
    is_finetuned = False
    for path in [ft_best, ft_last]:
        if os.path.exists(path):
            checkpoint_path = path
            is_finetuned = True
            break

    if checkpoint_path is None:
        if os.path.exists(pt_last):
            checkpoint_path = pt_last

    if checkpoint_path is None:
        raise FileNotFoundError(
            f"No checkpoint found in {ft_dir} or {pretrain_dir}"
        )

    log(f"Loading {model_type} checkpoint from {checkpoint_path}")

    if is_finetuned:
        # Fine-tuned: load full state dict into the fine-tune wrapper
        sd = torch.load(checkpoint_path, weights_only=True, map_location="cpu")

        # Extract backbone weights (strip 'fm.' prefix)
        backbone_sd = {}
        for k, v in sd.items():
            if k.startswith("fm."):
                backbone_sd[k[len("fm."):]] = v

        if model_type == "ropebert":
            if finetune_mode in ("cls", "cls_attack"):
                model = ProvenanceRoPEBERTFineTuneCLS(
                    bert_config, backbone_sd or None, device=device, freeze_backbone=freeze_backbone,
                )
            else:
                model = ProvenanceRoPEBERTFineTuneLP(
                    bert_config, backbone_sd or None, device=device, freeze_backbone=freeze_backbone,
                )
        elif model_type == "modernbert":
            if finetune_mode in ("cls", "cls_attack"):
                model = ProvenanceModernBERTFineTuneCLS(
                    bert_config, backbone_sd or None, device=device, freeze_backbone=freeze_backbone,
                )
            else:
                model = ProvenanceModernBERTFineTuneLP(
                    bert_config, backbone_sd or None, device=device, freeze_backbone=freeze_backbone,
                )
        else:
            if finetune_mode in ("cls", "cls_attack"):
                model = ProvenanceBERTFineTuneCLS(
                    bert_config, backbone_sd or None, device=device, freeze_backbone=freeze_backbone,
                )
            else:
                model = ProvenanceBERTFineTuneLP(
                    bert_config, backbone_sd or None, device=device, freeze_backbone=freeze_backbone,
                )

        # Load the full state dict (including classifier/head weights)
        model.load_state_dict(sd)
    else:
        # Pretrained only: load backbone into fine-tune wrapper
        sd = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
        if model_type == "ropebert":
            if finetune_mode in ("cls", "cls_attack"):
                model = ProvenanceRoPEBERTFineTuneCLS(
                    bert_config, sd, device=device, freeze_backbone=freeze_backbone,
                )
            else:
                model = ProvenanceRoPEBERTFineTuneLP(
                    bert_config, sd, device=device, freeze_backbone=freeze_backbone,
                )
        elif model_type == "modernbert":
            if finetune_mode in ("cls", "cls_attack"):
                model = ProvenanceModernBERTFineTuneCLS(
                    bert_config, sd, device=device, freeze_backbone=freeze_backbone,
                )
            else:
                model = ProvenanceModernBERTFineTuneLP(
                    bert_config, sd, device=device, freeze_backbone=freeze_backbone,
                )
        else:
            if finetune_mode in ("cls", "cls_attack"):
                model = ProvenanceBERTFineTuneCLS(
                    bert_config, sd, device=device, freeze_backbone=freeze_backbone,
                )
            else:
                model = ProvenanceBERTFineTuneLP(
                    bert_config, sd, device=device, freeze_backbone=freeze_backbone,
                )

    model.eval()
    return model


def _load_pretrained_model(
    pretrain_dir: str,
    tokenizer: ProvenanceTokenizer,
    model_size: str,
    device: str,
    model_type: str = "bert",
    pretrain_cfg=None,
):
    """Load the pretrained model wrapped in LP for scoring (no fine-tuning)."""
    if model_type == "ropebert":
        rope_theta = pretrain_cfg.rope_theta
        bert_config = get_ropebert_config(
            tokenizer.vocab_size, model_size, tokenizer.max_seq_len,
            rope_theta=rope_theta,
        )
    elif model_type == "modernbert":
        bert_config = get_modernbert_config(
            tokenizer.vocab_size, model_size, tokenizer.max_seq_len,
            global_attn_every_n_layers=pretrain_cfg.global_attn_every_n_layers,
            local_attention_window=pretrain_cfg.local_attention_window,
        )
    else:
        bert_config = get_bert_config(tokenizer.vocab_size, model_size, tokenizer.max_seq_len)

    pt_last = os.path.join(pretrain_dir, f"pretrain_{model_size}.pt")

    if not os.path.exists(pt_last):
        raise FileNotFoundError(f"No pretrained checkpoint found in {pretrain_dir}")
    checkpoint_path = pt_last

    log(f"Loading {model_type} pretrained checkpoint from {checkpoint_path}")
    sd = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
    if model_type == "ropebert":
        model = ProvenanceRoPEBERTFineTuneLP(bert_config, sd, device=device)
    elif model_type == "modernbert":
        model = ProvenanceModernBERTFineTuneLP(bert_config, sd, device=device)
    else:
        model = ProvenanceBERTFineTuneLP(bert_config, sd, device=device)
    model.eval()
    return model



def _prepare_score_batch_cls(
    edges: List[Tuple[str, str, str, float]],
    sampler: ProvenanceWalkSampler,
    tokenizer: ProvenanceTokenizer,
    indexid2msg: dict,
    walk_length: int,
):
    """Prepare a batch for CLS scoring (no negative sampling).

    Returns:
        input_ids: [B, L]
        attention_mask: [B, L]
        target_mask: [B, L]
    """
    batch_input_ids = []
    batch_attn = []
    batch_target = []

    for src, dst, etype, ts in edges:
        walk_nodes, walk_edge_types = sampler.sample_bidirectional_walk(
            src, max_ts=ts, walk_length=walk_length,
        )
        dst_walk_nodes, dst_walk_edge_types = sampler.sample_context_walk(
            dst, max_ts=ts, walk_length=walk_length,
        )
        tok_ids, attn, tgt_mask = tokenizer.tokenize_for_finetune(
            walk_nodes, walk_edge_types, dst, etype, indexid2msg, mode="cls",
            dst_walk_nodes=dst_walk_nodes, dst_walk_edge_types=dst_walk_edge_types,
        )
        batch_input_ids.append(tok_ids)
        batch_attn.append(attn)
        batch_target.append(tgt_mask)

    # Pad to max length
    max_len = max(t.size(0) for t in batch_input_ids)
    pad_id = tokenizer.pad_id
    B = len(batch_input_ids)

    input_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros(B, max_len, dtype=torch.bool)
    target_mask = torch.zeros(B, max_len, dtype=torch.bool)

    for i in range(B):
        L = batch_input_ids[i].size(0)
        input_ids[i, :L] = batch_input_ids[i]
        attention_mask[i, :L] = batch_attn[i]
        target_mask[i, :L] = batch_target[i]

    return input_ids, attention_mask, target_mask


def _prepare_score_batch_lp(
    edges: List[Tuple[str, str, str, float]],
    sampler: ProvenanceWalkSampler,
    tokenizer: ProvenanceTokenizer,
    indexid2msg: dict,
    walk_length: int,
):
    """Prepare a batch for LP scoring (masked destination prediction).

    Returns:
        input_ids: [B, L]
        attention_mask: [B, L]
        target_mask: [B, L]
        target_token_ids: [B, L]
    """
    batch_input_ids = []
    batch_attn = []
    batch_target_mask = []
    batch_target_ids = []

    for src, dst, etype, ts in edges:
        walk_nodes, walk_edge_types = sampler.sample_bidirectional_walk(
            src, max_ts=ts, walk_length=walk_length,
        )

        # Get masked version
        tok_ids_masked, attn_masked, tgt_mask = tokenizer.tokenize_for_finetune(
            walk_nodes, walk_edge_types, dst, etype, indexid2msg, mode="lp",
        )

        # Build target: true token IDs where masked, -100 elsewhere
        target_ids = torch.full_like(tok_ids_masked, -100)
        mask_positions = tgt_mask.nonzero(as_tuple=True)[0]

        if dst in indexid2msg:
            dst_type, dst_label = indexid2msg[dst]
            dst_tokens = tokenizer.tokenize_node(dst_type, dst_label)
            for j, pos in enumerate(mask_positions):
                if j < len(dst_tokens):
                    target_ids[pos] = dst_tokens[j]

        batch_input_ids.append(tok_ids_masked)
        batch_attn.append(attn_masked)
        batch_target_mask.append(tgt_mask)
        batch_target_ids.append(target_ids)

    # Pad
    max_len = max(t.size(0) for t in batch_input_ids)
    pad_id = tokenizer.pad_id
    B = len(batch_input_ids)

    input_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros(B, max_len, dtype=torch.bool)
    target_mask = torch.zeros(B, max_len, dtype=torch.bool)
    target_token_ids = torch.full((B, max_len), -100, dtype=torch.long)

    for i in range(B):
        L = batch_input_ids[i].size(0)
        input_ids[i, :L] = batch_input_ids[i]
        attention_mask[i, :L] = batch_attn[i]
        target_mask[i, :L] = batch_target_mask[i]
        target_token_ids[i, :L] = batch_target_ids[i]

    return input_ids, attention_mask, target_mask, target_token_ids


# ── Parallel walk sampling ──────────────────────────────────────────────────

# Module-level globals inherited by forked worker processes (read-only, COW).
_mp_sampler: Optional[ProvenanceWalkSampler] = None
_mp_walk_length: int = 10


def _walk_worker(edge_chunk: List[Tuple[str, str, str, float]]) -> list:
    """Sample src + dst walks for a chunk of edges (runs in forked child)."""
    results = []
    for src, dst, etype, ts in edge_chunk:
        src_wn, src_we = _mp_sampler.sample_bidirectional_walk(
            src, max_ts=ts, walk_length=_mp_walk_length,
        )
        dst_wn, dst_we = _mp_sampler.sample_context_walk(
            dst, max_ts=ts, walk_length=_mp_walk_length,
        )
        results.append((src_wn, src_we, dst_wn, dst_we))
    return results


def _parallel_sample_walks(
    edges: List[Tuple[str, str, str, float]],
    sampler: ProvenanceWalkSampler,
    walk_length: int,
    num_workers: Optional[int] = None,
) -> list:
    """Pre-sample all walks using multiprocessing (fork, read-only COW)."""
    global _mp_sampler, _mp_walk_length
    _mp_sampler = sampler
    _mp_walk_length = walk_length

    if num_workers is None:
        num_workers = min(mp.cpu_count(), 8)

    # Fall back to sequential for small edge counts or single worker
    if num_workers <= 1 or len(edges) < 2000:
        return _walk_worker(edges)

    chunk_size = max(1, (len(edges) + num_workers - 1) // num_workers)
    chunks = [edges[i:i + chunk_size] for i in range(0, len(edges), chunk_size)]

    ctx = mp.get_context("fork")
    with ctx.Pool(num_workers) as pool:
        result_chunks = pool.map(_walk_worker, chunks)

    return [w for chunk in result_chunks for w in chunk]


@torch.no_grad()
def _score_edges_cls(
    edges: List[Tuple[str, str, str, float]],
    sampler: ProvenanceWalkSampler,
    model: ProvenanceBERTFineTuneCLS,
    tokenizer: ProvenanceTokenizer,
    indexid2msg: dict,
    walk_length: int,
    batch_size: int,
    device: str,
) -> List[float]:
    """Score edges using CLS mode: raw logit as anomaly score.

    Walks are pre-sampled in parallel across CPU cores (fork-based
    multiprocessing), then tokenized and scored in batched GPU passes.
    """
    # Phase 1: parallel walk sampling on CPU
    all_walks = _parallel_sample_walks(edges, sampler, walk_length)

    # Phase 2: tokenize + GPU inference in batches
    scores = []
    for batch_start in range(0, len(edges), batch_size):
        batch_edges = edges[batch_start:batch_start + batch_size]
        batch_walks = all_walks[batch_start:batch_start + batch_size]

        batch_input_ids = []
        batch_attn = []
        batch_target = []
        for (src_wn, src_we, dst_wn, dst_we), (src, dst, etype, ts) in zip(batch_walks, batch_edges):
            tok_ids, attn, tgt_mask = tokenizer.tokenize_for_finetune(
                src_wn, src_we, dst, etype, indexid2msg, mode="cls",
                dst_walk_nodes=dst_wn, dst_walk_edge_types=dst_we,
            )
            batch_input_ids.append(tok_ids)
            batch_attn.append(attn)
            batch_target.append(tgt_mask)

        max_len = max(t.size(0) for t in batch_input_ids)
        pad_id = tokenizer.pad_id
        B = len(batch_input_ids)
        input_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros(B, max_len, dtype=torch.bool)
        target_mask = torch.zeros(B, max_len, dtype=torch.bool)
        for i in range(B):
            L = batch_input_ids[i].size(0)
            input_ids[i, :L] = batch_input_ids[i]
            attention_mask[i, :L] = batch_attn[i]
            target_mask[i, :L] = batch_target[i]

        logits = model.predict(
            input_ids.to(device),
            attention_mask.to(device),
            target_mask.to(device),
        )
        scores.extend(logits.squeeze(-1).cpu().numpy().tolist())

    return scores


@torch.no_grad()
def _score_edges_lp(
    edges: List[Tuple[str, str, str, float]],
    sampler: ProvenanceWalkSampler,
    model: ProvenanceBERTFineTuneLP,
    tokenizer: ProvenanceTokenizer,
    indexid2msg: dict,
    walk_length: int,
    batch_size: int,
    device: str,
    edge_score_weight: float = 0.0,
) -> List[float]:
    """Score edges: node CE + optional lambda * edge-type CE."""
    scores = []
    for batch_start in range(0, len(edges), batch_size):
        batch_edges = edges[batch_start:batch_start + batch_size]
        input_ids, attn_mask, tgt_mask, tgt_ids = _prepare_score_batch_lp(
            batch_edges, sampler, tokenizer, indexid2msg, walk_length,
        )

        node_scores = model.anomaly_score(
            input_ids.to(device),
            attn_mask.to(device),
            tgt_mask.to(device),
            tgt_ids.to(device),
        )  # [B]

        # Optional edge-type scoring
        if edge_score_weight > 0:
            e_input_ids, e_attn_mask, e_tgt_mask, e_tgt_ids = _prepare_score_batch_edge(
                batch_edges, sampler, tokenizer, indexid2msg, walk_length,
            )
            e_scores = model.anomaly_score(
                e_input_ids.to(device),
                e_attn_mask.to(device),
                e_tgt_mask.to(device),
                e_tgt_ids.to(device),
            )
            node_scores = node_scores + edge_score_weight * e_scores

        scores.extend(node_scores.cpu().numpy().tolist())

    return scores


def _prepare_score_batch_edge(
    edges: List[Tuple[str, str, str, float]],
    sampler: ProvenanceWalkSampler,
    tokenizer: ProvenanceTokenizer,
    indexid2msg: dict,
    walk_length: int,
):
    """Prepare batch with edge type masked, destination visible.

    Returns:
        input_ids: [B, L]
        attention_mask: [B, L]
        target_mask: [B, L]
        target_token_ids: [B, L]
    """
    batch_input_ids = []
    batch_attn = []
    batch_tgt_mask = []
    batch_tgt_ids = []

    for src, dst, etype, ts in edges:
        walk_nodes, walk_edge_types = sampler.sample_bidirectional_walk(
            src, max_ts=ts, walk_length=walk_length,
        )
        tok_ids, attn, edge_mask, edge_tgt = tokenizer.tokenize_for_edge_scoring(
            walk_nodes, walk_edge_types, dst, etype, indexid2msg,
        )
        batch_input_ids.append(tok_ids)
        batch_attn.append(attn)
        batch_tgt_mask.append(edge_mask)
        batch_tgt_ids.append(edge_tgt)

    # Pad
    max_len = max(t.size(0) for t in batch_input_ids)
    pad_id = tokenizer.pad_id
    B = len(batch_input_ids)

    input_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros(B, max_len, dtype=torch.bool)
    target_mask = torch.zeros(B, max_len, dtype=torch.bool)
    target_token_ids = torch.full((B, max_len), -100, dtype=torch.long)

    for i in range(B):
        L = batch_input_ids[i].size(0)
        input_ids[i, :L] = batch_input_ids[i]
        attention_mask[i, :L] = batch_attn[i]
        target_mask[i, :L] = batch_tgt_mask[i]
        target_token_ids[i, :L] = batch_tgt_ids[i]

    return input_ids, attention_mask, target_mask, target_token_ids




def _score_graph(
    graph,
    model,
    tokenizer: ProvenanceTokenizer,
    indexid2msg: dict,
    rel2id: dict,
    finetune_mode: str,
    walk_length: int,
    batch_size: int,
    time_weight: str,
    half_life: float,
    device: str,
    num_walks: int = 1,
    edge_score_weight: float = 2.0,
    context_graph: Optional[nx.MultiDiGraph] = None,
    prebuilt_sampler: Optional[ProvenanceWalkSampler] = None,
) -> pd.DataFrame:
    """Score all edges in a single graph and return a DataFrame.

    prebuilt_sampler: if provided, reuse this sampler instead of constructing a
    new one. Use this in day/all context modes to avoid rebuilding adjacency
    lists (O(E)) for every snapshot graph.
    context_graph: used only when prebuilt_sampler is None; builds a fresh
    sampler from context_graph (or graph itself if context_graph is None).
    """
    edges = _collect_edges(graph)
    if not edges:
        return pd.DataFrame(columns=["loss", "srcnode", "dstnode", "time", "edge_type"])

    if prebuilt_sampler is not None:
        sampler = prebuilt_sampler
    else:
        sampler = ProvenanceWalkSampler(
            context_graph if context_graph is not None else graph,
            walk_length=walk_length, num_walks=1,
            time_weight=time_weight, half_life=half_life,
        )

    score_fn = {
        "cls": _score_edges_cls,
        "cls_attack": _score_edges_cls,
        "lp": _score_edges_lp,
        "mlm": _score_edges_lp,
    }[finetune_mode]

    common_args = (edges, sampler, model, tokenizer, indexid2msg,
                   walk_length, batch_size, device)
    extra_args = (edge_score_weight,) if finetune_mode in ("lp", "mlm") else ()

    if num_walks <= 1:
        scores = score_fn(*common_args, *extra_args)
    else:
        # Average scores over multiple walks
        all_scores = np.zeros(len(edges))
        for _ in range(num_walks):
            walk_scores = score_fn(*common_args, *extra_args)
            all_scores += np.array(walk_scores)
        scores = (all_scores / num_walks).tolist()

    # Build DataFrame matching inference_loop.py format
    rows = []
    for (src, dst, etype, ts), score in zip(edges, scores):
        edge_type_id = rel2id.get(etype, 0)
        rows.append({
            "loss": float(score),
            "srcnode": int(src),
            "dstnode": int(dst),
            "time": int(ts),
            "edge_type": int(edge_type_id),
        })

    return pd.DataFrame(rows)


# ── Edge classification detection ──────────────────────────────────────────────

def _detect_edge_cls(cfg):
    """Edge classification detection: score edges with static or event embeddings."""
    log_start(__file__)

    from .models.tgn import EdgeTypeClassifier

    pretrain_cfg = cfg.featurization.feat_training.spider
    detect_cfg = cfg.detection.gnn_training.spider
    model_size = pretrain_cfg.model_size
    batch_size = detect_cfg.inference_batch_size
    device = "cuda" if torch.cuda.is_available() and not cfg._use_cpu else "cpu"

    ft_dir = cfg.detection.gnn_training._trained_models_dir
    edge_losses_dir = cfg.detection.gnn_training._edge_losses_dir

    # ── Run training first ────────────────────────────────────────
    from .finetune import main as finetune_main
    finetune_main(cfg)

    # ── Load checkpoint ───────────────────────────────────────────
    ckpt_path = os.path.join(ft_dir, f"finetune_edge_cls_{model_size}_best.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(ft_dir, f"finetune_edge_cls_{model_size}.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No edge_cls checkpoint found in {ft_dir}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    ecfg = ckpt["edge_cls_config"]
    node_to_idx = ckpt["node_to_idx"]
    edge_type_to_id = ckpt["edge_type_to_id"]

    model = EdgeTypeClassifier(
        num_nodes=ckpt["num_nodes"],
        num_edge_types=ckpt["num_edge_types"],
        entity_emb_dim=ecfg["entity_emb_dim"],
        hidden_dim=ecfg["hidden_dim"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    log(f"EdgeTypeClassifier loaded ({sum(p.numel() for p in model.parameters()):,} params)")

    # ── Load data ─────────────────────────────────────────────────
    rel2id = get_rel2id(cfg)
    indexid2msg = get_indexid2msg(cfg)
    base_dir = cfg.preprocessing.transformation._graphs_dir
    val_paths = get_all_files_from_folders(base_dir, cfg.dataset.val_files)
    test_paths = get_all_files_from_folders(base_dir, cfg.dataset.test_files)

    splits = {"val": val_paths, "test": test_paths}
    model_epoch_dir = "model_epoch_0"

    for split_name, graph_paths in splits.items():
        log(f"Edge-cls scoring {split_name} ({len(graph_paths)} graphs)...")

        for graph_path in graph_paths:
            graph = torch.load(graph_path, weights_only=False)
            edges = _collect_edges(graph)
            if not edges:
                continue

            rows = []
            for batch_start in range(0, len(edges), batch_size):
                batch = edges[batch_start:batch_start + batch_size]

                src_idx = torch.tensor(
                    [node_to_idx.get(e[0], 0) for e in batch],
                    dtype=torch.long, device=device)
                dst_idx = torch.tensor(
                    [node_to_idx.get(e[1], 0) for e in batch],
                    dtype=torch.long, device=device)
                etypes = torch.tensor(
                    [edge_type_to_id.get(e[2], 0) for e in batch],
                    dtype=torch.long, device=device)

                with torch.no_grad():
                    logits = model.predict(src_idx, dst_idx)
                    scores = F.cross_entropy(
                        logits, etypes, reduction="none")

                scores_cpu = scores.cpu().numpy().tolist()
                for (src, dst, etype, ts), score in zip(batch, scores_cpu):
                    rows.append({
                        "loss": float(score),
                        "srcnode": int(src),
                        "dstnode": int(dst),
                        "time": int(ts),
                        "edge_type": int(rel2id.get(etype, 0)),
                    })

            edge_df = pd.DataFrame(rows)
            start_time = edge_df["time"].min()
            end_time = edge_df["time"].max()
            time_interval = (
                ns_time_to_datetime_US(start_time)
                + "~"
                + ns_time_to_datetime_US(end_time)
            )

            csv_dir = os.path.join(edge_losses_dir, split_name, model_epoch_dir)
            os.makedirs(csv_dir, exist_ok=True)
            csv_file = os.path.join(csv_dir, time_interval + ".csv")
            edge_df.to_csv(csv_file, sep=",", header=True, index=False, encoding="utf-8")

        log(f"  {split_name} CSVs written to {os.path.join(edge_losses_dir, split_name)}")

    log("Edge classification detection complete.")


# ── TGN detection ──────────────────────────────────────────────────────────────

@torch.no_grad()
def _score_edges_tgn(
    all_edges_by_graph: dict,
    model,
    node_to_idx: dict,
    edge_type_to_id: dict,
    rel2id: dict,
    batch_size: int,
    device: str,
    objective: str = "contrastive",
    event_src_embs: torch.Tensor = None,
    event_dst_embs: torch.Tensor = None,
    edge_to_pair_idx: torch.Tensor = None,
) -> dict:
    """Score all edges across multiple graphs in temporal order using TGN.

    Unlike CLS/LP modes that score per-graph independently, TGN processes
    ALL edges in temporal order because memory state is cumulative.

    For contrastive objective: score = raw binary logit (higher = more anomalous).
    For edge_pred objective: score = cross-entropy loss of true edge type.
    For multi_edge_pred objective: score = binary CE of the true edge type
        under the multi-label model (higher = more surprising).
    For dist_edge_pred objective: score = -log P(true_type) under the predicted
        distribution (higher = more surprising).
    For edge_pred_emb objective: score = CE of true type + MSE between predicted
        and actual edge embedding (higher = more anomalous).

    Returns:
        dict mapping graph_path -> pd.DataFrame with scores.
    """
    # Flatten all edges with their graph origin, original index, and global index
    flat_edges = []
    global_idx = 0
    for path, edges in all_edges_by_graph.items():
        for i, edge in enumerate(edges):
            flat_edges.append((edge, path, i, global_idx))
            global_idx += 1

    # Sort globally by timestamp
    flat_edges.sort(key=lambda x: x[0][3])

    # Pre-allocate score storage
    scores_by_graph = {path: [None] * len(edges) for path, edges in all_edges_by_graph.items()}

    for batch_start in range(0, len(flat_edges), batch_size):
        batch = flat_edges[batch_start:batch_start + batch_size]

        edges_batch = [x[0] for x in batch]
        paths_batch = [x[1] for x in batch]
        indices_batch = [x[2] for x in batch]
        global_indices = [x[3] for x in batch]

        src_idx = torch.tensor(
            [node_to_idx.get(e[0], 0) for e in edges_batch], dtype=torch.long, device=device)
        dst_idx = torch.tensor(
            [node_to_idx.get(e[1], 0) for e in edges_batch], dtype=torch.long, device=device)
        edge_types = torch.tensor(
            [edge_type_to_id.get(e[2], 0) for e in edges_batch], dtype=torch.long, device=device)
        timestamps = torch.tensor(
            [e[3] for e in edges_batch], dtype=torch.long, device=device)

        # Look up per-edge contextual embeddings for this batch (if available)
        batch_event_src = None
        batch_event_dst = None
        if event_src_embs is not None and edge_to_pair_idx is not None:
            gi = torch.tensor(global_indices, dtype=torch.long)
            pair_idx = edge_to_pair_idx[gi]
            batch_event_src = event_src_embs[pair_idx].to(device)
            batch_event_dst = event_dst_embs[pair_idx].to(device)

        if objective == "edge_pred":
            logits = model.predict_edge_types(src_idx, dst_idx, timestamps,
                                              event_src_emb=batch_event_src, event_dst_emb=batch_event_dst)
            # Per-edge CE loss as anomaly score (higher = more surprising)
            per_edge_scores = torch.nn.functional.cross_entropy(
                logits, edge_types, reduction="none",
            )
            scores_cpu = per_edge_scores.cpu().numpy().tolist()

        elif objective == "multi_edge_pred":
            logits = model.predict_edge_types(src_idx, dst_idx, timestamps,
                                              event_src_emb=batch_event_src, event_dst_emb=batch_event_dst)
            # Per-edge binary CE: how surprising is this edge type under the
            # multi-label model?  Higher = more anomalous.
            per_edge_logit = logits[
                torch.arange(len(edges_batch), device=device), edge_types]
            per_edge_scores = F.binary_cross_entropy_with_logits(
                per_edge_logit, torch.ones_like(per_edge_logit),
                reduction="none",
            )
            scores_cpu = per_edge_scores.cpu().numpy().tolist()

        elif objective == "dist_edge_pred":
            logits = model.predict_edge_types(src_idx, dst_idx, timestamps,
                                              event_src_emb=batch_event_src, event_dst_emb=batch_event_dst)
            # Per-edge NLL under predicted distribution: -log P(true_type)
            log_probs = F.log_softmax(logits, dim=-1)
            per_edge_scores = -log_probs[
                torch.arange(len(edges_batch), device=device), edge_types]
            scores_cpu = per_edge_scores.cpu().numpy().tolist()

        elif objective == "edge_pred_emb":
            logits = model.predict_edge_types(src_idx, dst_idx, timestamps,
                                              event_src_emb=batch_event_src, event_dst_emb=batch_event_dst)
            ce_scores = F.cross_entropy(
                logits, edge_types, reduction="none")
            pred_emb = model.predict_edge_embedding(src_idx, dst_idx, timestamps,
                                                    event_src_emb=batch_event_src, event_dst_emb=batch_event_dst)
            target_emb = model.edge_emb(edge_types)
            mse_scores = (pred_emb - target_emb).pow(2).mean(dim=-1)
            per_edge_scores = ce_scores + mse_scores
            scores_cpu = per_edge_scores.cpu().numpy().tolist()

        elif objective == "node_pred":
            logits = model.predict_node_types(src_idx, dst_idx, edge_types, timestamps,
                                              event_src_emb=batch_event_src, event_dst_emb=batch_event_dst)
            dst_types = model.node_types[dst_idx]
            # Per-edge CE loss as anomaly score (higher = more surprising)
            per_edge_scores = F.cross_entropy(
                logits, dst_types, reduction="none")
            scores_cpu = per_edge_scores.cpu().numpy().tolist()

        else:
            logits = model.classify_edges(src_idx, dst_idx, edge_types, timestamps,
                                          event_src_emb=batch_event_src, event_dst_emb=batch_event_dst)
            per_edge_scores = logits.squeeze(-1)
            scores_cpu = per_edge_scores.cpu().numpy().tolist()

        for score, path, idx in zip(scores_cpu, paths_batch, indices_batch):
            scores_by_graph[path][idx] = score

        # Update memories after scoring (with anomaly scores for score-gated memory)
        model.update_memory(src_idx, dst_idx, edge_types, timestamps,
                            edge_scores=per_edge_scores)

    # Build DataFrames
    result = {}
    for path, edges in all_edges_by_graph.items():
        scores = scores_by_graph[path]
        if not edges:
            result[path] = pd.DataFrame(columns=["loss", "srcnode", "dstnode", "time", "edge_type"])
            continue
        rows = []
        for (src, dst, etype, ts), score in zip(edges, scores):
            rows.append({
                "loss": float(score) if score is not None else 0.0,
                "srcnode": int(src),
                "dstnode": int(dst),
                "time": int(ts),
                "edge_type": int(rel2id.get(etype, 0)),
            })
        result[path] = pd.DataFrame(rows)

    return result


def _detect_tgn(cfg):
    """TGN detection: score val/test edges with temporal entity memory."""
    log_start(__file__)

    from .models.tgn import ProvenanceTGN

    pretrain_cfg = cfg.featurization.feat_training.spider
    detect_cfg = cfg.detection.gnn_training.spider
    model_size = pretrain_cfg.model_size
    batch_size = detect_cfg.inference_batch_size
    device = "cuda" if torch.cuda.is_available() and not cfg._use_cpu else "cpu"

    ft_dir = cfg.detection.gnn_training._trained_models_dir
    edge_losses_dir = cfg.detection.gnn_training._edge_losses_dir

    # ── Run training first (same pattern as existing detect.py) ────
    from .finetune import main as finetune_main
    finetune_main(cfg)

    # ── Load checkpoint ────────────────────────────────────────────
    ckpt_path = os.path.join(ft_dir, f"finetune_tgn_{model_size}_best.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(ft_dir, f"finetune_tgn_{model_size}.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No TGN checkpoint found in {ft_dir}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    tgn_config = ckpt["tgn_config"]
    node_to_idx = ckpt["node_to_idx"]
    edge_type_to_id = ckpt["edge_type_to_id"]
    objective = tgn_config.get("objective", "contrastive")

    use_event_bert_emb = tgn_config.get("use_event_bert_emb", False)
    event_bert_emb_dim = tgn_config.get("event_bert_emb_dim", 0)

    model = ProvenanceTGN(
        num_nodes=ckpt["num_nodes"],
        num_edge_types=ckpt["num_edge_types"],
        entity_emb_dim=tgn_config["entity_emb_dim"],
        memory_dim=tgn_config["memory_dim"],
        edge_emb_dim=tgn_config["edge_emb_dim"],
        time_dim=tgn_config["time_dim"],
        num_heads=tgn_config["num_heads"],
        use_node_type_emb=tgn_config.get("use_node_type_emb", False),
        use_time_emb=tgn_config.get("use_time_emb", True),
        use_memory=tgn_config.get("use_memory", True),
        use_entity_emb=tgn_config.get("use_entity_emb", True),
        use_event_bert_emb=use_event_bert_emb,
        event_bert_emb_dim=event_bert_emb_dim,
        score_gated_memory=tgn_config.get("score_gated_memory", False),
        temporal_decay=tgn_config.get("temporal_decay", False),
        anomaly_accumulator=tgn_config.get("anomaly_accumulator", False),
        device=device,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    log(f"TGN model loaded ({sum(p.numel() for p in model.parameters()):,} params)")

    # ── Load data ──────────────────────────────────────────────────
    rel2id = get_rel2id(cfg)
    base_dir = cfg.preprocessing.transformation._graphs_dir
    val_paths = get_all_files_from_folders(base_dir, cfg.dataset.val_files)
    test_paths = get_all_files_from_folders(base_dir, cfg.dataset.test_files)

    splits = {"val": val_paths, "test": test_paths}
    model_epoch_dir = "model_epoch_0"

    reset_memory = detect_cfg.tgn.reset_memory_on_inference

    for split_name, graph_paths in splits.items():
        log(f"TGN scoring {split_name} ({len(graph_paths)} graphs)...")

        if reset_memory:
            # Production mode: start with empty memory (no training history)
            model.reset_memory()
            log(f"  Memory reset to zero (reset_memory_on_inference=True)")
        else:
            # Research mode: start from trained memory state (fresh per split)
            trained_memory = ckpt["memory_state"]
            model.load_memory_state({k: v.to(device) for k, v in trained_memory.items()})

        # Collect all edges per graph
        all_edges_by_graph = {}
        total_edges = 0
        for gp in graph_paths:
            graph = torch.load(gp, weights_only=False)
            edges = _collect_edges(graph)
            all_edges_by_graph[gp] = edges
            total_edges += len(edges)
        log(f"  {split_name}: {total_edges:,} edges across {len(graph_paths)} graphs")

        # Score all edges in temporal order with memory updates
        result_dfs = _score_edges_tgn(
            all_edges_by_graph, model, node_to_idx, edge_type_to_id,
            rel2id, batch_size, device, objective=objective,
            event_src_embs=None,
            event_dst_embs=None,
            edge_to_pair_idx=None,
        )

        # Write CSVs (same format as existing detect.py)
        for graph_path in graph_paths:
            edge_df = result_dfs[graph_path]
            if edge_df.empty:
                continue

            start_time = edge_df["time"].min()
            end_time = edge_df["time"].max()
            time_interval = (
                ns_time_to_datetime_US(start_time)
                + "~"
                + ns_time_to_datetime_US(end_time)
            )

            csv_dir = os.path.join(edge_losses_dir, split_name, model_epoch_dir)
            os.makedirs(csv_dir, exist_ok=True)
            csv_file = os.path.join(csv_dir, time_interval + ".csv")
            edge_df.to_csv(csv_file, sep=",", header=True, index=False, encoding="utf-8")

        log(f"  {split_name} CSVs written to {os.path.join(edge_losses_dir, split_name)}")

    log("TGN detection complete.")


# ── Perplexity-based detection (decoder-only LLaMA) ────────────────────────────

@torch.no_grad()
def _score_edges_perplexity(
    edges: List[Tuple[str, str, str, float]],
    sampler: ProvenanceWalkSampler,
    model,
    tokenizer: ProvenanceTokenizer,
    indexid2msg: dict,
    walk_length: int,
    batch_size: int,
    device: str,
) -> List[float]:
    """Score edges using causal LM perplexity: mean next-token prediction loss.

    For each edge, samples a walk including the edge context, runs teacher-forced
    causal LM forward pass, and averages per-token cross-entropy as anomaly score.
    Higher score = model didn't expect this sequence = more anomalous.

    Walks are pre-sampled in parallel across CPU cores.
    """
    # Phase 1: parallel walk sampling on CPU
    all_walks = _parallel_sample_walks(edges, sampler, walk_length)

    # Phase 2: tokenize + GPU inference in batches
    scores = []
    for batch_start in range(0, len(edges), batch_size):
        batch_edges = edges[batch_start:batch_start + batch_size]
        batch_walks = all_walks[batch_start:batch_start + batch_size]

        batch_token_ids = []
        for (src_wn, src_we, dst_wn, dst_we), (src, dst, etype, ts) in zip(batch_walks, batch_edges):
            # Tokenize the walk with dst context (same as CLS mode but no masking)
            tok_ids, attn, _ = tokenizer.tokenize_for_finetune(
                src_wn, src_we, dst, etype, indexid2msg, mode="cls",
                dst_walk_nodes=dst_wn, dst_walk_edge_types=dst_we,
            )
            batch_token_ids.append(tok_ids)

        if not batch_token_ids:
            continue

        # Pad to max length
        max_len = max(t.size(0) for t in batch_token_ids)
        pad_id = tokenizer.pad_id
        B = len(batch_token_ids)
        input_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros(B, max_len, dtype=torch.bool)
        for i in range(B):
            L = batch_token_ids[i].size(0)
            input_ids[i, :L] = batch_token_ids[i]
            attention_mask[i, :L] = True

        # Labels = input_ids; model shifts internally for next-token prediction.
        # Set padding to -100 so it's ignored in loss.
        labels = input_ids.clone()
        labels[~attention_mask] = -100

        # Forward pass returns per-sample loss
        out = model.modified_fwd(
            input_ids.to(device),
            attention_mask.to(device),
            labels.to(device),
            return_loss=False,
        )
        logits = out.logits  # [B, L, V]

        # Compute per-token loss, then average per sample
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].to(device).contiguous()
        B, L, V = shift_logits.shape
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
        per_token_loss = loss_fct(
            shift_logits.view(-1, V), shift_labels.view(-1)
        ).view(B, L)

        # Average over valid (non-padding) positions
        valid = shift_labels != -100
        num_valid = valid.float().sum(dim=1).clamp(min=1)
        per_edge_scores = (per_token_loss * valid.float()).sum(dim=1) / num_valid
        scores.extend(per_edge_scores.cpu().numpy().tolist())

    return scores


def _detect_perplexity(cfg):
    """Perplexity-based detection using decoder-only LLaMA model.

    Loads the pretrained LLaMA model directly (no fine-tuning) and scores
    each edge by computing the mean next-token prediction loss over the
    walk containing that edge. Higher perplexity = more anomalous.
    """
    log_start(__file__)

    from .models.llama import ProvenanceLLaMA, get_llama_config

    pretrain_cfg = cfg.featurization.feat_training.spider
    detect_cfg = cfg.detection.gnn_training.spider
    model_size = pretrain_cfg.model_size
    walk_length = detect_cfg.finetune_walk_len
    batch_size = detect_cfg.inference_batch_size
    num_walks = detect_cfg.num_inference_walks
    time_weight = pretrain_cfg.corpus.time_weight
    half_life = pretrain_cfg.corpus.half_life
    graph_context_mode = pretrain_cfg.graph_context_mode

    device = "cuda" if torch.cuda.is_available() and not cfg._use_cpu else "cpu"

    pretrain_dir = cfg.featurization.feat_training._model_dir
    edge_losses_dir = cfg.detection.gnn_training._edge_losses_dir

    # ── Load tokenizer ──────────────────────────────────────────────────
    tokenizer = ProvenanceTokenizer(cfg)
    tokenizer.load(os.path.join(pretrain_dir, "tokenizer.pt"))
    log(f"Loaded tokenizer (vocab_size={tokenizer.vocab_size})")

    # ── Load pretrained LLaMA model ─────────────────────────────────────
    rope_theta = pretrain_cfg.rope_theta
    llama_config = get_llama_config(
        tokenizer.vocab_size, model_size, tokenizer.max_seq_len,
        rope_theta=rope_theta,
    )

    checkpoint_path = os.path.join(pretrain_dir, f"pretrain_{model_size}.pt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No pretrained checkpoint found in {pretrain_dir}")

    log(f"Loading LLaMA checkpoint from {checkpoint_path}")
    sd = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
    model = ProvenanceLLaMA(llama_config)
    model.load_state_dict(sd)
    model = model.to(device)
    model.eval()
    log(f"LLaMA model on {device} ({sum(p.numel() for p in model.parameters()):,} params)")

    # ── Load data ───────────────────────────────────────────────────────
    indexid2msg = get_indexid2msg(cfg)
    rel2id = get_rel2id(cfg)
    base_dir = cfg.preprocessing.transformation._graphs_dir

    val_paths = get_all_files_from_folders(base_dir, cfg.dataset.val_files)
    test_paths = get_all_files_from_folders(base_dir, cfg.dataset.test_files)

    splits = {"val": val_paths, "test": test_paths}
    model_epoch_dir = "model_epoch_0"

    log(f"Graph context mode: {graph_context_mode}")

    for split_name, graph_paths in splits.items():
        log(f"Perplexity scoring {split_name} ({len(graph_paths)} graphs)...")

        # Build context graph(s) — same logic as CLS/LP modes
        if graph_context_mode == "all":
            split_context = _build_context_graph(graph_paths)
            log(f"  {split_name}: merged context — {split_context.number_of_nodes():,} nodes, "
                f"{split_context.number_of_edges():,} edges")
            split_sampler = ProvenanceWalkSampler(
                split_context, walk_length=walk_length, num_walks=1,
                time_weight=time_weight, half_life=half_life,
            )
        elif graph_context_mode == "day":
            day_groups = _group_paths_by_day(graph_paths)
            day_samplers = {
                day: ProvenanceWalkSampler(
                    _build_context_graph(paths),
                    walk_length=walk_length, num_walks=1,
                    time_weight=time_weight, half_life=half_life,
                )
                for day, paths in day_groups.items()
            }

        for graph_path in log_tqdm(graph_paths, desc=f"Detect ({split_name})"):
            graph = torch.load(graph_path, weights_only=False)
            edges = _collect_edges(graph)
            if not edges:
                continue

            # Build or reuse sampler
            if graph_context_mode == "all":
                sampler = split_sampler
            elif graph_context_mode == "day":
                day_folder = os.path.basename(os.path.dirname(graph_path))
                sampler = day_samplers.get(day_folder)
                if sampler is None:
                    sampler = ProvenanceWalkSampler(
                        graph, walk_length=walk_length, num_walks=1,
                        time_weight=time_weight, half_life=half_life,
                    )
            else:  # window
                sampler = ProvenanceWalkSampler(
                    graph, walk_length=walk_length, num_walks=1,
                    time_weight=time_weight, half_life=half_life,
                )

            # Score with optional multi-walk averaging
            if num_walks <= 1:
                edge_scores = _score_edges_perplexity(
                    edges, sampler, model, tokenizer, indexid2msg,
                    walk_length, batch_size, device,
                )
            else:
                all_scores = np.zeros(len(edges))
                for _ in range(num_walks):
                    walk_scores = _score_edges_perplexity(
                        edges, sampler, model, tokenizer, indexid2msg,
                        walk_length, batch_size, device,
                    )
                    all_scores += np.array(walk_scores)
                edge_scores = (all_scores / num_walks).tolist()

            # Build DataFrame
            rows = []
            for (src, dst, etype, ts), score in zip(edges, edge_scores):
                rows.append({
                    "loss": float(score),
                    "srcnode": int(src),
                    "dstnode": int(dst),
                    "time": int(ts),
                    "edge_type": int(rel2id.get(etype, 0)),
                })
            edge_df = pd.DataFrame(rows)

            if edge_df.empty:
                continue

            start_time = edge_df["time"].min()
            end_time = edge_df["time"].max()
            time_interval = (
                ns_time_to_datetime_US(start_time)
                + "~"
                + ns_time_to_datetime_US(end_time)
            )

            csv_dir = os.path.join(edge_losses_dir, split_name, model_epoch_dir)
            os.makedirs(csv_dir, exist_ok=True)
            csv_file = os.path.join(csv_dir, time_interval + ".csv")
            edge_df.to_csv(csv_file, sep=",", header=True, index=False, encoding="utf-8")

        log(f"  {split_name} CSVs written to {os.path.join(edge_losses_dir, split_name)}")

    log("Perplexity detection complete.")


def main(cfg):
    """Score val/test edges with SPIDER and write CSV files.

    Modes:
      cls:        CLS fine-tuning + sigmoid scoring
      lp:         LP fine-tuning + mean cross-entropy scoring
      mlm:        pretrained model only (no fine-tuning) + CE scoring
      tgn:        TGN with attention-based entity memory
      edge_cls:   edge type prediction from static BERT entity embeddings
      perplexity: decoder-only LLaMA, direct perplexity scoring (no fine-tuning)
    """
    log_start(__file__)

    pretrain_cfg = cfg.featurization.feat_training.spider
    detect_cfg = cfg.detection.gnn_training.spider
    model_type = pretrain_cfg.model_type
    model_size = pretrain_cfg.model_size
    finetune_mode = detect_cfg.finetune_mode

    # TGN uses a completely different pipeline (temporal scoring with memory updates)
    if finetune_mode == "tgn":
        _detect_tgn(cfg)
        return

    if finetune_mode == "edge_cls":
        _detect_edge_cls(cfg)
        return

    if finetune_mode == "perplexity":
        _detect_perplexity(cfg)
        return

    walk_length = detect_cfg.finetune_walk_len
    freeze_backbone = detect_cfg.freeze_backbone
    batch_size = detect_cfg.inference_batch_size
    num_walks = detect_cfg.num_inference_walks
    time_weight = pretrain_cfg.corpus.time_weight
    half_life = pretrain_cfg.corpus.half_life
    mask_edge_type = pretrain_cfg.mask_edge_type
    edge_score_weight = detect_cfg.edge_score_weight if mask_edge_type else 0.0
    graph_context_mode = pretrain_cfg.graph_context_mode
    log(f"Graph context mode: {graph_context_mode}")

    device = "cuda" if torch.cuda.is_available() and not cfg._use_cpu else "cpu"

    # ── Directories ────────────────────────────────────────────────────
    pretrain_dir = cfg.featurization.feat_training._model_dir
    ft_dir = cfg.detection.gnn_training._trained_models_dir

    # ── Load tokenizer ──────────────────────────────────────────────────
    tokenizer = ProvenanceTokenizer(cfg)
    tokenizer.load(os.path.join(pretrain_dir, "tokenizer.pt"))
    log(f"Loaded tokenizer (vocab_size={tokenizer.vocab_size})")

    # ── Load model ─────────────────────────────────────────────────────
    if finetune_mode == "mlm":
        model = _load_pretrained_model(pretrain_dir, tokenizer, model_size, device, model_type, pretrain_cfg)
    else:
        from .finetune import main as finetune_main
        finetune_main(cfg)
        model = _load_finetuned_model(
            pretrain_dir, ft_dir, tokenizer, model_size, model_type, finetune_mode, freeze_backbone, device, pretrain_cfg,
        )
    log(f"Model on {device} ({sum(p.numel() for p in model.parameters()):,} params)")

    # ── Load data ───────────────────────────────────────────────────────
    indexid2msg = get_indexid2msg(cfg)
    rel2id = get_rel2id(cfg)
    base_dir = cfg.preprocessing.transformation._graphs_dir
    edge_losses_dir = cfg.detection.gnn_training._edge_losses_dir

    train_paths = get_all_files_from_folders(base_dir, cfg.dataset.train_files)
    val_paths   = get_all_files_from_folders(base_dir, cfg.dataset.val_files)
    test_paths  = get_all_files_from_folders(base_dir, cfg.dataset.test_files)

    splits = {
        "val": val_paths,
        "test": test_paths,
    }

    # ── Score edges and write CSVs ──────────────────────────────────────
    model_epoch_dir = "model_epoch_0"

    for split_name, graph_paths in splits.items():
        log(f"Scoring {split_name} edges ({len(graph_paths)} graphs, context={graph_context_mode})...")

        # Build context graph(s) for this split.
        # window: no merged context; sampler uses only the current snapshot.
        # day:    one merged context graph per calendar day; walks can cross
        #         windows within the same day but not across days.
        # all:    one merged context graph from all snapshots in this split;
        #         no training data is included, so no leakage across splits.
        if graph_context_mode == "all":
            split_context = _build_context_graph(graph_paths)
            log(f"  {split_name}: merged context — {split_context.number_of_nodes():,} nodes, "
                f"{split_context.number_of_edges():,} edges")
            split_sampler = ProvenanceWalkSampler(
                split_context, walk_length=walk_length, num_walks=1,
                time_weight=time_weight, half_life=half_life,
            )
            log(f"  {split_name}: sampler built (forward+backward adj ready)")
        elif graph_context_mode == "day":
            day_groups = _group_paths_by_day(graph_paths)
            day_contexts = {day: _build_context_graph(paths)
                           for day, paths in day_groups.items()}
            day_samplers = {
                day: ProvenanceWalkSampler(
                    ctx, walk_length=walk_length, num_walks=1,
                    time_weight=time_weight, half_life=half_life,
                )
                for day, ctx in day_contexts.items()
            }
            log(f"  {split_name}: {len(day_samplers)} day-level samplers built")

        for graph_path in log_tqdm(graph_paths, desc=f"Detect ({split_name})"):
            graph = torch.load(graph_path)

            if graph_context_mode == "window":
                prebuilt = None
                context = None
            elif graph_context_mode == "all":
                prebuilt = split_sampler
                context = None
            else:  # day
                day_folder = os.path.basename(os.path.dirname(graph_path))
                prebuilt = day_samplers.get(day_folder)
                context = None

            edge_df = _score_graph(
                graph, model, tokenizer, indexid2msg, rel2id,
                finetune_mode, walk_length, batch_size,
                time_weight, half_life, device, num_walks,
                edge_score_weight,
                context_graph=context,
                prebuilt_sampler=prebuilt,
            )

            if edge_df.empty:
                continue

            # Build time interval string for filename
            start_time = edge_df["time"].min()
            end_time = edge_df["time"].max()
            time_interval = (
                ns_time_to_datetime_US(start_time)
                + "~"
                + ns_time_to_datetime_US(end_time)
            )

            # Write CSV
            csv_dir = os.path.join(edge_losses_dir, split_name, model_epoch_dir)
            os.makedirs(csv_dir, exist_ok=True)
            csv_file = os.path.join(csv_dir, time_interval + ".csv")
            edge_df.to_csv(csv_file, sep=",", header=True, index=False, encoding="utf-8")

        log(f"  {split_name} CSVs written to {os.path.join(edge_losses_dir, split_name)}")

    log("Detection complete.")
