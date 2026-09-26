"""
SPIDER fine-tuning for anomaly detection on provenance graphs.

Supports multiple modes:
- CLS: Binary classification with [CLS] token
- CLS_ATTACK: Supervised binary classification with real attack walks
- LP: Link prediction via masked destination node likelihood
- TGN: Temporal Graph Network with attention-based entity memory (self-supervised)

Follows CyberGFM's cls_finetune.py and lp_finetune.py patterns.
"""

import os
import pickle
import random
import time
from collections import defaultdict
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch_geometric.utils import scatter

from .training_utils import (
    WarmupLinearScheduler, WarmupCosineScheduler,
    pad_token_id_lists, pad_tensor_batch,
)

from pidsmaker.utils.utils import get_all_files_from_folders, get_indexid2msg, log, log_start
from pidsmaker.utils.labelling import get_GP_of_each_attack, get_ground_truth_edges
from pidsmaker.utils.dataset_utils import get_rel2id, ntype2id

from .models.bert import MODEL_SIZES, ProvenanceBERTFineTuneCLS, ProvenanceBERTFineTuneLP, get_bert_config
from .models.modernbert import ProvenanceModernBERTFineTuneCLS, ProvenanceModernBERTFineTuneLP, get_modernbert_config
from .models.ropebert import ProvenanceRoPEBERTFineTuneCLS, ProvenanceRoPEBERTFineTuneLP, get_ropebert_config
from .models.tgn import ProvenanceTGN
from .data.sampler import ProvenanceWalkSampler
from .data.tokenizer_bpe import ProvenanceTokenizerBPE as ProvenanceTokenizer


def _collect_edges(graph) -> List[Tuple[str, str, str, float]]:
    """Collect all edges as (src, dst, edge_type, time) from a NetworkX graph."""
    edges = []
    for src, dst, _, attrs in graph.edges(data=True, keys=True):
        edges.append((
            str(src), str(dst),
            attrs.get("label", ""),
            attrs.get("time", 0.0),
        ))
    return edges


def _build_neg_index(indexid2msg: dict) -> dict:
    """Build node_type -> list of (label, node_id) pairs for negative sampling.

    Grouping by (type, label) lets us efficiently exclude candidates whose
    textual representation is identical to the positive destination.
    """
    type_to_labeled: dict = defaultdict(list)
    for node_id, (node_type, node_label) in indexid2msg.items():
        type_to_labeled[node_type].append((node_label, str(node_id)))
    return type_to_labeled


def _precompute_negatives(
    edges: List[Tuple],
    indexid2msg: dict,
    type_to_labeled: dict,
    real_edge_set: set,
    max_tries: int = 20,
) -> dict:
    """Pre-compute one negative dst per unique (src, dst, etype) triple.

    Each negative satisfies:
    - Same node type as dst, but a *different* textual label (distinct tokens).
    - (src, neg_dst, etype) does not exist in real_edge_set.

    Returns a dict mapping (src, dst, etype) -> neg_dst string.

    Speed: builds two O(V) lookup tables once, then uses rejection sampling
    (random.choices, O(max_tries)) per triple — avoids the O(|type|) list
    comprehension + shuffle that the naive approach does per unique key.
    """
    all_node_ids = [str(n) for n in indexid2msg]

    # type_to_all[node_type] -> flat list of all node_ids of that type
    type_to_all: dict = {t: [n for _, n in nodes] for t, nodes in type_to_labeled.items()}
    # same_label[(node_type, label)] -> set of node_ids sharing that label
    # Used for O(1) exclusion instead of per-triple list filtering
    same_label: dict = defaultdict(set)
    for node_type, labeled_nodes in type_to_labeled.items():
        for label, node_id in labeled_nodes:
            same_label[(node_type, label)].add(node_id)

    # Deduplicate keys upfront to skip repeated computation
    unique_keys = dict.fromkeys(
        (str(e[0]), str(e[1]), str(e[2])) for e in edges
    )
    neg_map: dict = {}
    for src, dst, etype in unique_keys:
        dst_info = indexid2msg.get(dst)
        neg_dst = None
        if dst_info:
            dst_type, dst_label = dst_info
            pool = type_to_all.get(dst_type, [])
            excl = same_label.get((dst_type, dst_label), set())
            if pool:
                # Rejection sampling: O(max_tries), no large list construction
                for c in random.choices(pool, k=min(max_tries, len(pool))):
                    if c not in excl and (src, c, etype) not in real_edge_set:
                        neg_dst = c
                        break
                if neg_dst is None:
                    # Relax real-edge constraint, keep different-label guarantee
                    for c in random.choices(pool, k=min(max_tries, len(pool))):
                        if c not in excl:
                            neg_dst = c
                            break
        if neg_dst is None:
            neg_dst = random.choice(all_node_ids)
        neg_map[(src, dst, etype)] = neg_dst
    return neg_map


# ── Attack walk extraction (for cls_attack mode) ──────────────────────────────

def _extract_attack_walks(
    cfg,
    walk_length: int,
    time_weight: str,
    half_life: float,
    num_walks: int,
) -> List[Tuple[List[str], List[str], str, str]]:
    """Extract backward context walks that terminate at malicious attack nodes
    strictly within the attack time window for each node.

    Uses get_GP_of_each_attack to obtain per-attack (node IDs, time_range) so
    that only edges whose timestamp falls inside the known attack period are
    considered. This prevents sampling walks from before or after the attack,
    which would not carry attack-specific provenance patterns.

    Returns:
        List of (walk_nodes, walk_edge_types, dst_node_id, edge_type) tuples.
        These are used as the positive (anomalous, label=1) class in cls_attack.
    """
    attack_to_nids = get_GP_of_each_attack(cfg)

    # Build: mal_node_str -> list of (start_ns, end_ns) attack windows
    mal_node_to_windows: dict = defaultdict(list)
    for attack_info in attack_to_nids.values():
        start_ns, end_ns = attack_info["time_range"]
        for nid in attack_info["nids"]:
            mal_node_to_windows[str(nid)].append((start_ns, end_ns))

    mal_nids_str = set(mal_node_to_windows.keys())

    base_dir = cfg.preprocessing.transformation._graphs_dir
    attack_graph_paths = (
        get_all_files_from_folders(base_dir, cfg.dataset.val_files)
        + get_all_files_from_folders(base_dir, cfg.dataset.test_files)
    )

    attack_walks: List[Tuple[List[str], List[str], str, str]] = []
    n_mal_edges = 0

    for graph_path in attack_graph_paths:
        graph = torch.load(graph_path, weights_only=False)
        sampler = ProvenanceWalkSampler(
            graph, walk_length=walk_length, num_walks=1,
            time_weight=time_weight, half_life=half_life,
        )

        # First pass: collect unique (src, dst, etype) triples within the attack window.
        # Keep the latest timestamp per triple so the backward walk has maximum context.
        triple_to_ts: dict = {}  # (src_str, dst_str, etype) -> latest ts
        for src, dst, _, attrs in graph.edges(data=True, keys=True):
            dst_str = str(dst)
            if dst_str not in mal_nids_str:
                continue
            ts = attrs.get("time", 0.0)
            if not any(start <= ts <= end for start, end in mal_node_to_windows[dst_str]):
                continue
            etype = attrs.get("label", "")
            key = (str(src), dst_str, etype)
            if key not in triple_to_ts or ts > triple_to_ts[key]:
                triple_to_ts[key] = ts

        n_mal_edges += len(triple_to_ts)

        # Second pass: sample num_walks unique walks per unique triple.
        for (src_str, dst_str, etype), ts in triple_to_ts.items():
            # dst backward walk is deterministic (no randomness in sample_context_walk
            # for the first hop) — sample once and reuse across all src walk variants.
            dst_walk_nodes, dst_walk_edge_types = sampler.sample_context_walk(
                dst_str, max_ts=ts, walk_length=walk_length,
            )
            seen_seqs: set = set()
            n_tries = 0
            max_tries = num_walks * 5
            while len(seen_seqs) < num_walks and n_tries < max_tries:
                n_tries += 1
                walk_nodes, walk_edge_types = sampler.sample_bidirectional_walk(
                    src_str, max_ts=ts, walk_length=walk_length,
                )
                seq_key = tuple(walk_edge_types)
                if seq_key not in seen_seqs:
                    seen_seqs.add(seq_key)
                    attack_walks.append((
                        walk_nodes, walk_edge_types,
                        dst_walk_nodes, dst_walk_edge_types,
                        dst_str, etype,
                    ))

    log(f"Extracted {len(attack_walks)} unique attack walks from {n_mal_edges} unique malicious triples "
        f"({len(mal_nids_str)} malicious nodes)")
    return attack_walks


def _save_attack_walks(walks: List[Tuple], path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(walks, f)
    log(f"Saved {len(walks)} attack walks to {path}")


def _load_attack_walks(path: str) -> List[Tuple]:
    with open(path, "rb") as f:
        walks = pickle.load(f)
    log(f"Loaded {len(walks)} attack walks from {path}")
    return walks


def _prepare_finetune_batch_cls_attack(
    benign_edges: List[Tuple[str, str, str, float]],
    attack_walks_sample: List[Tuple[List[str], List[str], List[str], List[str], str, str]],
    sampler: ProvenanceWalkSampler,
    tokenizer: ProvenanceTokenizer,
    indexid2msg: dict,
    walk_length: int,
):
    """Prepare a CLS batch from real benign edges (label=0) and real attack walks (label=1).

    Args:
        benign_edges: edges from normal training graphs; provide context and destination.
        attack_walks_sample: pre-extracted attack walks (same length as benign_edges).
            Each element is (src_walk_nodes, src_walk_edge_types,
                             dst_walk_nodes, dst_walk_edge_types, dst, etype).

    Returns:
        input_ids, attention_mask, target_mask, labels — same format as
        _prepare_finetune_batch_cls, ready to pass to model.forward().
    """
    batch_input_ids = []
    batch_attn = []
    batch_target = []
    batch_labels = []

    for (src, dst, etype, ts), (
        aw_nodes, aw_etypes, aw_dst_walk_nodes, aw_dst_walk_etypes, aw_dst, aw_etype
    ) in zip(benign_edges, attack_walks_sample):
        # Benign example (label=0): sample src bidirectional walk + dst backward walk
        walk_nodes, walk_edge_types = sampler.sample_bidirectional_walk(src, max_ts=ts, walk_length=walk_length)
        dst_walk_nodes, dst_walk_edge_types = sampler.sample_context_walk(dst, max_ts=ts, walk_length=walk_length)
        tok_ids, attn, tgt_mask = tokenizer.tokenize_for_finetune(
            walk_nodes, walk_edge_types, dst, etype, indexid2msg, mode="cls",
            dst_walk_nodes=dst_walk_nodes, dst_walk_edge_types=dst_walk_edge_types,
        )
        batch_input_ids.append(tok_ids)
        batch_attn.append(attn)
        batch_target.append(tgt_mask)
        batch_labels.append(0.0)

        # Attack example (label=1) — use pre-extracted src and dst walks
        tok_ids, attn, tgt_mask = tokenizer.tokenize_for_finetune(
            aw_nodes, aw_etypes, aw_dst, aw_etype, indexid2msg, mode="cls",
            dst_walk_nodes=aw_dst_walk_nodes, dst_walk_edge_types=aw_dst_walk_etypes,
        )
        batch_input_ids.append(tok_ids)
        batch_attn.append(attn)
        batch_target.append(tgt_mask)
        batch_labels.append(1.0)

    # Pad to max length
    input_ids = pad_tensor_batch(batch_input_ids, pad_value=tokenizer.pad_id)
    attention_mask = pad_tensor_batch(batch_attn, pad_value=0)
    target_mask = pad_tensor_batch(batch_target, pad_value=0)
    labels = torch.zeros(len(batch_input_ids), 1)
    for i in range(len(batch_labels)):
        labels[i, 0] = batch_labels[i]

    return input_ids, attention_mask, target_mask, labels


def _prepare_finetune_batch_cls(
    edges: List[Tuple[str, str, str, float]],
    sampler: ProvenanceWalkSampler,
    tokenizer: ProvenanceTokenizer,
    indexid2msg: dict,
    walk_length: int,
    neg_map: dict = None,
    neg_ratio: float = 1.0,
):
    """Prepare a batch for CLS fine-tuning: positive + negative edges.

    Args:
        neg_map: Pre-computed (src, dst, etype) -> neg_dst dict. When provided,
                 negatives are guaranteed to have a different label from dst and
                 not to be real edges. Falls back to random node otherwise.

    Returns:
        input_ids: [B, L]
        attention_mask: [B, L]
        target_mask: [B, L]
        labels: [B, 1]
    """
    all_nodes = list(sampler.nodes)
    batch_input_ids = []
    batch_attn = []
    batch_target = []
    batch_labels = []

    for src, dst, etype, ts in edges:
        # Positive edge
        walk_nodes, walk_edge_types = sampler.sample_bidirectional_walk(src, max_ts=ts, walk_length=walk_length)
        dst_walk_nodes, dst_walk_edge_types = sampler.sample_context_walk(dst, max_ts=ts, walk_length=walk_length)
        tok_ids, attn, tgt_mask = tokenizer.tokenize_for_finetune(
            walk_nodes, walk_edge_types, dst, etype, indexid2msg, mode="cls",
            dst_walk_nodes=dst_walk_nodes, dst_walk_edge_types=dst_walk_edge_types,
        )
        batch_input_ids.append(tok_ids)
        batch_attn.append(attn)
        batch_target.append(tgt_mask)
        batch_labels.append(0.0)  # Normal

        # Hard negative: pre-computed (different label, not a real edge) or random
        if neg_map is not None:
            neg_dst = neg_map.get((str(src), str(dst), str(etype)), str(random.choice(all_nodes)))
        else:
            neg_dst = str(random.choice(all_nodes))
        neg_dst_walk_nodes, neg_dst_walk_edge_types = sampler.sample_context_walk(neg_dst, max_ts=ts, walk_length=walk_length)
        tok_ids, attn, tgt_mask = tokenizer.tokenize_for_finetune(
            walk_nodes, walk_edge_types, str(neg_dst), etype, indexid2msg, mode="cls",
            dst_walk_nodes=neg_dst_walk_nodes, dst_walk_edge_types=neg_dst_walk_edge_types,
        )
        batch_input_ids.append(tok_ids)
        batch_attn.append(attn)
        batch_target.append(tgt_mask)
        batch_labels.append(1.0)  # Anomalous

    # Pad to max length
    input_ids = pad_tensor_batch(batch_input_ids, pad_value=tokenizer.pad_id)
    attention_mask = pad_tensor_batch(batch_attn, pad_value=0)
    target_mask = pad_tensor_batch(batch_target, pad_value=0)

    labels = torch.tensor(batch_labels, dtype=torch.float32).unsqueeze(-1)
    return input_ids, attention_mask, target_mask, labels


def _prepare_finetune_batch_lp(
    edges: List[Tuple[str, str, str, float]],
    sampler: ProvenanceWalkSampler,
    tokenizer: ProvenanceTokenizer,
    indexid2msg: dict,
    walk_length: int,
):
    """Prepare a batch for LP fine-tuning: mask destination, predict tokens.

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
        walk_nodes, walk_edge_types = sampler.sample_bidirectional_walk(src, max_ts=ts, walk_length=walk_length)

        # Get full tokenization (to know dst tokens)
        tok_ids_full, attn_full, _ = tokenizer.tokenize_for_finetune(
            walk_nodes, walk_edge_types, dst, etype, indexid2msg, mode="cls"
        )

        # Get masked version
        tok_ids_masked, attn_masked, tgt_mask = tokenizer.tokenize_for_finetune(
            walk_nodes, walk_edge_types, dst, etype, indexid2msg, mode="lp"
        )

        # Build target: true token IDs where masked, -100 elsewhere
        target_ids = torch.full_like(tok_ids_masked, -100)
        # The LP mask positions correspond to the destination tokens
        mask_positions = tgt_mask.nonzero(as_tuple=True)[0]

        # Get actual destination tokens for targets
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
    input_ids = pad_tensor_batch(batch_input_ids, pad_value=tokenizer.pad_id)
    attention_mask = pad_tensor_batch(batch_attn, pad_value=0)
    target_mask = pad_tensor_batch(batch_target_mask, pad_value=0)
    target_token_ids = pad_tensor_batch(batch_target_ids, pad_value=-100)

    return input_ids, attention_mask, target_mask, target_token_ids


# ── TGN helpers ────────────────────────────────────────────────────────────────

def _build_node_to_idx(indexid2msg: dict) -> dict:
    """Build str(node_id) -> contiguous index mapping. Index 0 reserved for unknown."""
    sorted_keys = sorted(indexid2msg.keys(), key=lambda x: int(x))
    return {str(k): i + 1 for i, k in enumerate(sorted_keys)}


def _precompute_cross_type_negatives(
    edges: List[Tuple],
    indexid2msg: dict,
    node_to_idx: dict,
) -> dict:
    """Pre-compute cross-type negative dst per unique (src, dst, etype) triple.

    Each negative is a node of a DIFFERENT type than dst (e.g. file -> subject,
    subject -> netflow), creating structurally implausible edges that are
    maximally separated from benign ones in feature space.  This ensures the
    classifier learns a wide decision boundary, minimizing false positives.

    Returns: {(src, dst, etype): neg_dst_str}
    """
    type_to_nodes: dict = defaultdict(list)
    for node_id, (node_type, _) in indexid2msg.items():
        nid_str = str(node_id)
        if nid_str in node_to_idx:
            type_to_nodes[node_type].append(nid_str)

    all_types = list(type_to_nodes.keys())
    all_node_ids = list(node_to_idx.keys())

    unique_keys = dict.fromkeys(
        (str(e[0]), str(e[1]), str(e[2])) for e in edges
    )

    neg_map: dict = {}
    for src, dst, etype in unique_keys:
        dst_info = indexid2msg.get(dst)
        if dst_info:
            dst_type = dst_info[0]
            other_types = [t for t in all_types if t != dst_type]
            if other_types:
                neg_type = random.choice(other_types)
                neg_map[(src, dst, etype)] = random.choice(type_to_nodes[neg_type])
                continue
        # Fallback: random node (only if dst type unknown or all nodes share one type)
        neg_map[(src, dst, etype)] = random.choice(all_node_ids)

    return neg_map


def _precompute_entity_embeddings(
    cfg,
    indexid2msg: dict,
    node_to_idx: dict,
    num_nodes: int,
    entity_emb_dim: int,
    device: str,
) -> torch.Tensor:
    """Pre-compute static BERT embeddings for all entities via mean pooling.

    Runs each entity's (type, label) tokens through frozen pretrained BERT,
    mean-pools over non-padding positions to get a single [H] vector.

    Returns: [num_nodes, entity_emb_dim] tensor.
    """
    pretrain_cfg = cfg.featurization.feat_training.spider
    model_type = pretrain_cfg.model_type
    model_size = pretrain_cfg.model_size
    pretrain_dir = cfg.featurization.feat_training._model_dir
    batch_size = pretrain_cfg.batch_size

    # Use a large batch size since node labels are very short (~5 tokens vs 128 for walks)
    emb_batch_size = batch_size * 16

    # Load tokenizer
    tokenizer = ProvenanceTokenizer(cfg)
    tokenizer.load(os.path.join(pretrain_dir, "tokenizer.pt"))

    # Load pretrained backbone
    if model_type == "ropebert":
        from .models.ropebert import ProvenanceRoPEBERT
        rope_theta = getattr(pretrain_cfg, 'rope_theta', 10000.0)
        bert_config = get_ropebert_config(
            tokenizer.vocab_size, model_size, tokenizer.max_seq_len,
            rope_theta=rope_theta,
        )
        bert = ProvenanceRoPEBERT(bert_config)
    elif model_type == "modernbert":
        from .models.modernbert import ProvenanceModernBERT
        bert_config = get_modernbert_config(
            tokenizer.vocab_size, model_size, tokenizer.max_seq_len,
            global_attn_every_n_layers=pretrain_cfg.global_attn_every_n_layers,
            local_attention_window=pretrain_cfg.local_attention_window,
        )
        bert = ProvenanceModernBERT(bert_config)
    elif model_type == "t5":
        from .models.t5 import ProvenanceT5, get_t5_config
        bert_config = get_t5_config(tokenizer.vocab_size, model_size, tokenizer.max_seq_len)
        bert = ProvenanceT5(bert_config)
    else:
        from .models.bert import ProvenanceBERT
        bert_config = get_bert_config(tokenizer.vocab_size, model_size, tokenizer.max_seq_len)
        bert = ProvenanceBERT(bert_config)

    pt_path = os.path.join(pretrain_dir, f"pretrain_{model_size}.pt")

    sd = torch.load(pt_path, weights_only=True, map_location="cpu")
    bert.load_state_dict(sd)
    bert = bert.to(device)
    bert.eval()

    entity_embs = torch.zeros(num_nodes, entity_emb_dim)

    # Build list sorted by contiguous index for deterministic batching
    items = sorted(
        ((str(nid), ntype, nlabel) for nid, (ntype, nlabel) in indexid2msg.items()),
        key=lambda x: node_to_idx.get(x[0], 0),
    )

    total_batches = (len(items) + emb_batch_size - 1) // emb_batch_size
    for batch_idx, batch_start in enumerate(range(0, len(items), emb_batch_size)):
        batch = items[batch_start:batch_start + emb_batch_size]

        if batch_idx % 20 == 0:
            log(f"  Embedding batch {batch_idx+1}/{total_batches} ({batch_start:,}/{len(items):,} nodes)")


        batch_token_ids = []
        batch_indices = []
        for nid_str, ntype, nlabel in batch:
            idx = node_to_idx.get(nid_str)
            if idx is None:
                continue
            tids = tokenizer.tokenize_node(ntype, nlabel)
            if tids:
                batch_token_ids.append(tids)
                batch_indices.append(idx)

        if not batch_token_ids:
            continue

        input_ids, attention_mask = pad_token_id_lists(batch_token_ids, tokenizer.pad_id, tokenizer.max_seq_len)

        with torch.no_grad():
            hidden = bert.modified_fwd(
                input_ids.to(device),
                attention_mask.to(device),
                labels=torch.full_like(input_ids, -100).to(device),
                skip_cls=True,
            )  # [B, max_len, H]

        # Mean pool over non-padding positions
        mask_f = attention_mask.to(device).unsqueeze(-1).float()
        pooled = (hidden * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)

        for i, idx in enumerate(batch_indices):
            entity_embs[idx] = pooled[i].cpu()

    del bert
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    log(f"Pre-computed entity embeddings: {entity_embs.shape}")
    return entity_embs



def _train_edge_cls(cfg):
    """Edge type classification from BERT embeddings.

    Path mode: static entity embeddings per node (existing behavior).
    Event mode: per-edge contextual embeddings from masked-event BERT pass.

    Predicts edge type given (src, dst) BERT embeddings.  No temporal memory
    or state — edges are shuffled each epoch.  CE loss, anomaly score = CE of
    true edge type.
    """
    log_start(__file__)

    from .models.tgn import EdgeTypeClassifier
    import torch.nn as nn

    pretrain_cfg = cfg.featurization.feat_training.spider
    detect_cfg = cfg.detection.gnn_training.spider
    model_size = pretrain_cfg.model_size
    finetune_epochs = detect_cfg.finetune_epochs
    finetune_lr = detect_cfg.finetune_lr
    scheduler_type = pretrain_cfg.scheduler

    device = "cuda" if torch.cuda.is_available() and not cfg._use_cpu else "cpu"
    ft_dir = cfg.detection.gnn_training._trained_models_dir
    os.makedirs(ft_dir, exist_ok=True)

    H = MODEL_SIZES[model_size].H
    tgn_cfg = detect_cfg.tgn
    hidden_dim = tgn_cfg.edge_emb_dim
    batch_size = tgn_cfg.batch_size

    # ── Data loading ───────────────────────────────────────────────
    indexid2msg = get_indexid2msg(cfg)
    rel2id = get_rel2id(cfg)
    base_dir = cfg.preprocessing.transformation._graphs_dir
    train_graph_paths = get_all_files_from_folders(base_dir, cfg.dataset.train_files)

    node_to_idx = _build_node_to_idx(indexid2msg)
    num_nodes = len(node_to_idx) + 1

    edge_type_to_id = {k: v for k, v in rel2id.items() if isinstance(k, str)}
    num_edge_types = max(edge_type_to_id.values()) + 1 if edge_type_to_id else 1

    train_edges = []
    for p in train_graph_paths:
        g = torch.load(p, weights_only=False)
        train_edges.extend(_collect_edges(g))
    log(f"Training edges: {len(train_edges)}")

    # ── Embeddings ─────────────────────────────────────────────────
    entity_embs_path = os.path.join(ft_dir, "entity_embs_tgn.pt")
    if os.path.exists(entity_embs_path):
        entity_embs = torch.load(entity_embs_path, map_location="cpu")
        log(f"Loaded cached entity embeddings: {entity_embs.shape}")
    else:
        log("Pre-computing entity BERT embeddings...")
        entity_embs = _precompute_entity_embeddings(
            cfg, indexid2msg, node_to_idx, num_nodes, H, device,
        )
        torch.save(entity_embs, entity_embs_path)

    model = EdgeTypeClassifier(
        num_nodes=num_nodes,
        num_edge_types=num_edge_types,
        entity_emb_dim=H,
        hidden_dim=hidden_dim,
    ).to(device)
    model.set_entity_embeddings(entity_embs)

    log(f"EdgeTypeClassifier: {sum(p.numel() for p in model.parameters()):,} params, "
        f"{num_nodes:,} entities, {num_edge_types} edge types")

    opt = AdamW(model.parameters(), lr=finetune_lr, betas=(0.9, 0.99),
                eps=1e-10, weight_decay=0.02)
    updates_per_epoch = max(1, len(train_edges) // batch_size)
    total_steps = updates_per_epoch * finetune_epochs
    warmup_steps = max(1, total_steps // 20)
    if scheduler_type == "cosine":
        scheduler = WarmupCosineScheduler(opt, warmup_steps, total_steps)
    else:
        scheduler = WarmupLinearScheduler(opt, warmup_steps, total_steps)

    log_path = os.path.join(ft_dir, f"finetune_edge_cls_{model_size}_log.txt")
    with open(log_path, "w") as f:
        f.write("epoch,update,loss,lr\n")

    best_loss = float("inf")
    updates = 0

    for epoch in range(finetune_epochs):
        model.train()
        random.shuffle(train_edges)

        epoch_loss = 0.0
        n_batches = 0
        st = time.time()

        for batch_start in range(0, len(train_edges), batch_size):
            batch_edges = train_edges[batch_start:batch_start + batch_size]

            src_idx = torch.tensor(
                [node_to_idx.get(e[0], 0) for e in batch_edges],
                dtype=torch.long, device=device)
            dst_idx = torch.tensor(
                [node_to_idx.get(e[1], 0) for e in batch_edges],
                dtype=torch.long, device=device)
            etypes = torch.tensor(
                [edge_type_to_id.get(e[2], 0) for e in batch_edges],
                dtype=torch.long, device=device)

            logits = model.predict(src_idx, dst_idx)
            loss = F.cross_entropy(logits, etypes)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1
            updates += 1

            if updates % 50 == 0:
                lr = scheduler.get_last_lr()[0]
                log(f"[e{epoch+1}|u{updates}] loss={loss.item():.4f} lr={lr:.2e}")

        avg_loss = epoch_loss / max(1, n_batches)
        elapsed = time.time() - st
        log(f"Epoch {epoch+1}/{finetune_epochs}: avg_loss={avg_loss:.4f} ({elapsed:.1f}s)")

        with open(log_path, "a") as f:
            f.write(f"{epoch+1},{updates},{avg_loss:.6f},{scheduler.get_last_lr()[0]:.2e}\n")

        ckpt = {
            "model_state_dict": model.state_dict(),
            "node_to_idx": node_to_idx,
            "edge_type_to_id": edge_type_to_id,
            "num_nodes": num_nodes,
            "num_edge_types": num_edge_types,
            "pretrain_mode": "path",
            "edge_cls_config": {
                "entity_emb_dim": H,
                "hidden_dim": hidden_dim,
            },
        }
        save_path = os.path.join(ft_dir, f"finetune_edge_cls_{model_size}.pt")
        torch.save(ckpt, save_path)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(ft_dir, f"finetune_edge_cls_{model_size}_best.pt")
            torch.save(ckpt, best_path)

    log(f"Edge classification training complete. Best loss: {best_loss:.4f}")


def _train_tgn(cfg):
    """Self-supervised TGN training.

    Five objectives are supported (tgn.objective):
      contrastive — BCE loss with two-tier negative sampling (same-type +
        cross-type corrupted dst).  Score at inference = raw binary logit.
      edge_pred — cross-entropy loss predicting the edge type from entity
        memories and embeddings (no negative sampling needed).  Score at
        inference = per-edge CE loss of the true edge type.
      multi_edge_pred — like edge_pred but groups edges by (src, dst) pair
        within each batch and uses multi-hot BCE to predict ALL co-occurring
        edge types.  Score at inference = per-edge binary CE of the true type.
      dist_edge_pred — groups edges by (src, dst) pair and predicts the
        proportion of each edge type (count distribution).  KL divergence
        loss.  Score at inference = -log P(true_type) under the predicted
        distribution.
      edge_pred_emb — predicts both the edge type (CE) and the edge embedding
        vector (MSE).  Score at inference = CE + MSE per edge.

    No attack labels are used in any mode.
    """
    log_start(__file__)

    pretrain_cfg = cfg.featurization.feat_training.spider
    detect_cfg = cfg.detection.gnn_training.spider
    model_size = pretrain_cfg.model_size
    finetune_epochs = detect_cfg.finetune_epochs
    finetune_lr = detect_cfg.finetune_lr
    scheduler_type = pretrain_cfg.scheduler

    device = "cuda" if torch.cuda.is_available() and not cfg._use_cpu else "cpu"
    ft_dir = cfg.detection.gnn_training._trained_models_dir
    os.makedirs(ft_dir, exist_ok=True)

    # ── TGN config ─────────────────────────────────────────────────
    H = MODEL_SIZES[model_size].H
    tgn_cfg = detect_cfg.tgn
    memory_dim = tgn_cfg.memory_dim
    edge_emb_dim = tgn_cfg.edge_emb_dim
    time_dim = tgn_cfg.time_dim
    num_heads = tgn_cfg.num_heads
    tgn_batch_size = tgn_cfg.batch_size
    use_node_type_emb = tgn_cfg.use_node_type_emb
    use_time_emb = tgn_cfg.use_time_emb
    use_memory = tgn_cfg.use_memory
    use_entity_emb = tgn_cfg.use_entity_emb
    score_gated_memory = tgn_cfg.score_gated_memory
    temporal_decay = tgn_cfg.temporal_decay
    anomaly_accumulator = tgn_cfg.anomaly_accumulator
    objective = tgn_cfg.objective  # "contrastive" / "edge_pred" / "multi_edge_pred"

    use_event_bert_emb = False

    # ── Data loading ───────────────────────────────────────────────
    indexid2msg = get_indexid2msg(cfg)
    rel2id = get_rel2id(cfg)
    base_dir = cfg.preprocessing.transformation._graphs_dir
    train_graph_paths = get_all_files_from_folders(base_dir, cfg.dataset.train_files)

    node_to_idx = _build_node_to_idx(indexid2msg)
    num_nodes = len(node_to_idx) + 1  # +1 for index-0 unknown sentinel

    # Build node type tensor: [num_nodes] mapping contiguous idx -> type ID
    node_types_tensor = torch.zeros(num_nodes, dtype=torch.long)
    for nid_str, (ntype, _) in indexid2msg.items():
        idx = node_to_idx.get(str(nid_str))
        if idx is not None:
            node_types_tensor[idx] = ntype2id.get(ntype, 0)

    edge_type_to_id = {k: v for k, v in rel2id.items() if isinstance(k, str)}
    num_edge_types = max(edge_type_to_id.values()) + 1 if edge_type_to_id else 1

    # Collect and sort all training edges by timestamp
    train_edges = []
    for p in train_graph_paths:
        g = torch.load(p, weights_only=False)
        train_edges.extend(_collect_edges(g))
    train_edges.sort(key=lambda e: e[3])
    log(f"Training edges: {len(train_edges)} (sorted by time)")

    # ── Negative sampling (contrastive only) ─────────────────────
    if objective == "contrastive":
        type_to_labeled = _build_neg_index(indexid2msg)
        real_edge_set: set = set()
        for p in train_graph_paths:
            g = torch.load(p, weights_only=False)
            for src, dst, _, attrs in g.edges(data=True, keys=True):
                real_edge_set.add((str(src), str(dst), attrs.get("label", "")))
        neg_map_same = _precompute_negatives(train_edges, indexid2msg, type_to_labeled, real_edge_set)
        neg_map_cross = _precompute_cross_type_negatives(train_edges, indexid2msg, node_to_idx)
        log(f"Pre-computed negatives for {len(neg_map_same)} unique triples "
            f"(same-type + cross-type, 2 per benign edge)")
    else:
        log(f"Objective: {objective} (no negative sampling needed)")

    # ── Entity embeddings ──────────────────────────────────────────
    entity_embs_path = os.path.join(ft_dir, "entity_embs_tgn.pt")
    if os.path.exists(entity_embs_path):
        entity_embs = torch.load(entity_embs_path, map_location="cpu")
        log(f"Loaded cached entity embeddings: {entity_embs.shape}")
    else:
        log("Pre-computing entity BERT embeddings...")
        entity_embs = _precompute_entity_embeddings(
            cfg, indexid2msg, node_to_idx, num_nodes, H, device,
        )
        torch.save(entity_embs, entity_embs_path)

    edge_to_pair_idx = None
    event_src_embs_gpu = None
    event_dst_embs_gpu = None
    event_bert_emb_dim = 0

    # ── Build model ────────────────────────────────────────────────
    model = ProvenanceTGN(
        num_nodes=num_nodes,
        num_edge_types=num_edge_types,
        entity_emb_dim=H,
        memory_dim=memory_dim,
        edge_emb_dim=edge_emb_dim,
        time_dim=time_dim,
        num_heads=num_heads,
        use_node_type_emb=use_node_type_emb,
        use_time_emb=use_time_emb,
        use_memory=use_memory,
        use_entity_emb=use_entity_emb,
        use_event_bert_emb=use_event_bert_emb,
        event_bert_emb_dim=event_bert_emb_dim,
        score_gated_memory=score_gated_memory,
        temporal_decay=temporal_decay,
        anomaly_accumulator=anomaly_accumulator,
        device=device,
    ).to(device)
    model.set_entity_embeddings(entity_embs)
    model.set_node_types(node_types_tensor)
    log(f"TGN model: {sum(p.numel() for p in model.parameters()):,} trainable params, "
        f"{num_nodes:,} entities, {num_edge_types} edge types, "
        f"use_memory={use_memory}, use_entity_emb={use_entity_emb}, "
        f"use_event_bert_emb={use_event_bert_emb}, "
        f"score_gated={score_gated_memory}, decay={temporal_decay}, accum={anomaly_accumulator}")

    # ── Optimizer & scheduler ──────────────────────────────────────
    opt = AdamW(model.parameters(), lr=finetune_lr, betas=(0.9, 0.99), eps=1e-10, weight_decay=0.02)
    updates_per_epoch = max(1, len(train_edges) // tgn_batch_size)
    total_steps = updates_per_epoch * finetune_epochs
    warmup_steps = max(1, total_steps // 20)
    if scheduler_type == "cosine":
        scheduler = WarmupCosineScheduler(opt, warmup_steps, total_steps)
    else:
        scheduler = WarmupLinearScheduler(opt, warmup_steps, total_steps)

    # ── Training loop ──────────────────────────────────────────────
    log_path = os.path.join(ft_dir, f"finetune_tgn_{model_size}_log.txt")
    with open(log_path, "w") as f:
        f.write("epoch,update,loss,lr\n")

    all_node_keys = list(node_to_idx.keys())
    best_loss = float("inf")
    updates = 0

    for epoch in range(finetune_epochs):
        model.train()
        model.reset_memory()

        epoch_loss = 0.0
        n_batches = 0
        st = time.time()

        # Process edges in TEMPORAL order (no shuffling)
        for batch_start in range(0, len(train_edges), tgn_batch_size):
            batch_edges = train_edges[batch_start:batch_start + tgn_batch_size]

            src_idx = torch.tensor(
                [node_to_idx.get(e[0], 0) for e in batch_edges], dtype=torch.long, device=device)
            dst_idx = torch.tensor(
                [node_to_idx.get(e[1], 0) for e in batch_edges], dtype=torch.long, device=device)
            etypes = torch.tensor(
                [edge_type_to_id.get(e[2], 0) for e in batch_edges], dtype=torch.long, device=device)
            ts = torch.tensor(
                [e[3] for e in batch_edges], dtype=torch.long, device=device)

            # Look up precomputed per-edge contextual embeddings for this batch
            if event_src_embs_gpu is not None:
                batch_pair_idx = edge_to_pair_idx[batch_start:batch_start + len(batch_edges)]
                batch_event_src = event_src_embs_gpu[batch_pair_idx]
                batch_event_dst = event_dst_embs_gpu[batch_pair_idx]
            else:
                batch_event_src = None
                batch_event_dst = None

            if objective == "edge_pred":
                # ── Edge type prediction: predict edge type from memories ──
                logits = model.predict_edge_types(src_idx, dst_idx, ts,
                                                  event_src_emb=batch_event_src, event_dst_emb=batch_event_dst)  # [B, num_edge_classes]
                loss = torch.nn.CrossEntropyLoss()(logits, etypes)
                # Per-edge anomaly scores for score-gated memory (detached)
                if score_gated_memory:
                    with torch.no_grad():
                        edge_scores = torch.nn.functional.cross_entropy(
                            logits, etypes, reduction="none")
                else:
                    edge_scores = None

            elif objective == "multi_edge_pred":
                # ── Multi-label edge type prediction ──
                # Group by (src, dst) pair; predict ALL co-occurring edge types
                pair_keys = torch.stack([src_idx, dst_idx], dim=1)  # [B, 2]
                unique_pairs, inverse = torch.unique(
                    pair_keys, dim=0, return_inverse=True)  # [P, 2], [B]
                num_pairs = unique_pairs.size(0)
                num_classes = num_edge_types + 1  # match edge_predictor output

                # Multi-hot target: which edge types occur for each pair
                multi_hot = torch.zeros(num_pairs, num_classes, device=device)
                multi_hot[inverse, etypes] = 1.0

                # Use latest timestamp per pair for prediction
                pair_ts = scatter(
                    ts, inverse, dim=0, dim_size=num_pairs, reduce="max")

                # For multi/dist_edge_pred with event embeddings, aggregate per pair
                if batch_event_src is not None:
                    pair_event_src = scatter(
                        batch_event_src, inverse, dim=0, dim_size=num_pairs, reduce="mean")
                    pair_event_dst = scatter(
                        batch_event_dst, inverse, dim=0, dim_size=num_pairs, reduce="mean")
                else:
                    pair_event_src = None
                    pair_event_dst = None

                logits = model.predict_edge_types(
                    unique_pairs[:, 0], unique_pairs[:, 1], pair_ts,
                    event_src_emb=pair_event_src, event_dst_emb=pair_event_dst,
                )  # [P, num_classes]
                loss = F.binary_cross_entropy_with_logits(logits, multi_hot)

                # Per-edge anomaly scores for score-gated memory
                if score_gated_memory:
                    with torch.no_grad():
                        edge_logits = logits[inverse]  # [B, num_classes]
                        per_edge_logit = edge_logits[
                            torch.arange(len(batch_edges), device=device), etypes]
                        edge_scores = F.binary_cross_entropy_with_logits(
                            per_edge_logit, torch.ones_like(per_edge_logit),
                            reduction="none")
                else:
                    edge_scores = None

            elif objective == "dist_edge_pred":
                # ── Edge type distribution prediction ──
                # Group by (src, dst) pair; predict the proportion of each type
                pair_keys = torch.stack([src_idx, dst_idx], dim=1)  # [B, 2]
                unique_pairs, inverse = torch.unique(
                    pair_keys, dim=0, return_inverse=True)  # [P, 2], [B]
                num_pairs = unique_pairs.size(0)
                num_classes = num_edge_types + 1  # match edge_predictor output

                # Count edge types per pair → normalize to distribution
                one_hot = torch.zeros(len(batch_edges), num_classes, device=device)
                one_hot[torch.arange(len(batch_edges), device=device), etypes] = 1.0
                counts = scatter(
                    one_hot, inverse, dim=0, dim_size=num_pairs, reduce="sum")
                target_dist = counts / counts.sum(dim=1, keepdim=True).clamp(min=1)

                # Use latest timestamp per pair for prediction
                pair_ts = scatter(
                    ts, inverse, dim=0, dim_size=num_pairs, reduce="max")

                if batch_event_src is not None:
                    pair_event_src = scatter(
                        batch_event_src, inverse, dim=0, dim_size=num_pairs, reduce="mean")
                    pair_event_dst = scatter(
                        batch_event_dst, inverse, dim=0, dim_size=num_pairs, reduce="mean")
                else:
                    pair_event_src = None
                    pair_event_dst = None

                logits = model.predict_edge_types(
                    unique_pairs[:, 0], unique_pairs[:, 1], pair_ts,
                    event_src_emb=pair_event_src, event_dst_emb=pair_event_dst,
                )  # [P, num_classes]
                log_probs = F.log_softmax(logits, dim=-1)
                loss = F.kl_div(log_probs, target_dist, reduction="batchmean")

                # Per-edge anomaly scores for score-gated memory
                if score_gated_memory:
                    with torch.no_grad():
                        edge_log_probs = log_probs[inverse]  # [B, num_classes]
                        edge_scores = -edge_log_probs[
                            torch.arange(len(batch_edges), device=device), etypes]
                else:
                    edge_scores = None

            elif objective == "edge_pred_emb":
                # ── Edge type + embedding prediction ──
                # CE loss on edge type + MSE loss on edge embedding vector
                logits = model.predict_edge_types(src_idx, dst_idx, ts,
                                                  event_src_emb=batch_event_src, event_dst_emb=batch_event_dst)
                ce_loss = F.cross_entropy(logits, etypes)

                pred_emb = model.predict_edge_embedding(src_idx, dst_idx, ts,
                                                        event_src_emb=batch_event_src, event_dst_emb=batch_event_dst)
                target_emb = model.edge_emb(etypes).detach()
                mse_loss = F.mse_loss(pred_emb, target_emb)

                loss = ce_loss + mse_loss

                # Per-edge anomaly scores for score-gated memory
                if score_gated_memory:
                    with torch.no_grad():
                        edge_scores = F.cross_entropy(
                            logits, etypes, reduction="none")
                else:
                    edge_scores = None

            elif objective == "node_pred":
                # ── Node type prediction: predict dst node type ──
                logits = model.predict_node_types(src_idx, dst_idx, etypes, ts,
                                                  event_src_emb=batch_event_src, event_dst_emb=batch_event_dst)  # [B, NUM_NODE_TYPES]
                dst_types = model.node_types[dst_idx]  # [B] ground truth
                loss = F.cross_entropy(logits, dst_types)

                # Per-edge anomaly scores for score-gated memory
                if score_gated_memory:
                    with torch.no_grad():
                        edge_scores = F.cross_entropy(
                            logits, dst_types, reduction="none")
                else:
                    edge_scores = None

            else:
                # ── Contrastive: benign vs negatives ──
                logits_benign = model.classify_edges(src_idx, dst_idx, etypes, ts,
                                                     event_src_emb=batch_event_src, event_dst_emb=batch_event_dst)
                labels_benign = torch.zeros(len(batch_edges), 1, device=device)

                # Tier 1: same-type negative (label=1)
                neg_same_strs = [
                    neg_map_same.get((e[0], e[1], e[2]), random.choice(all_node_keys))
                    for e in batch_edges
                ]
                neg_same_idx = torch.tensor(
                    [node_to_idx.get(n, 0) for n in neg_same_strs], dtype=torch.long, device=device)
                logits_neg_same = model.classify_edges(src_idx, neg_same_idx, etypes, ts,
                                                       event_src_emb=batch_event_src, event_dst_emb=batch_event_dst)

                # Tier 2: cross-type negative (label=1)
                neg_cross_strs = [
                    neg_map_cross.get((e[0], e[1], e[2]), random.choice(all_node_keys))
                    for e in batch_edges
                ]
                neg_cross_idx = torch.tensor(
                    [node_to_idx.get(n, 0) for n in neg_cross_strs], dtype=torch.long, device=device)
                logits_neg_cross = model.classify_edges(src_idx, neg_cross_idx, etypes, ts,
                                                        event_src_emb=batch_event_src, event_dst_emb=batch_event_dst)

                labels_neg = torch.ones(len(batch_edges) * 2, 1, device=device)

                loss = torch.nn.BCEWithLogitsLoss()(
                    torch.cat([logits_benign, logits_neg_same, logits_neg_cross], dim=0),
                    torch.cat([labels_benign, labels_neg], dim=0),
                )
                # Per-edge anomaly scores for score-gated memory (detached)
                if score_gated_memory:
                    with torch.no_grad():
                        edge_scores = logits_benign.squeeze(-1).clone()
                else:
                    edge_scores = None

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            scheduler.step()

            # ── Sync differentiable memory to buffer, then update ──
            # detach_memory() breaks the graph from the previous batch (truncated
            # BPTT depth 1), then update_memory() creates a fresh _diff_memory
            # overlay that the NEXT batch's predict/classify will read from.
            model.detach_memory()
            model.update_memory(src_idx, dst_idx, etypes, ts, edge_scores=edge_scores)

            epoch_loss += loss.item()
            n_batches += 1
            updates += 1

            if updates % 50 == 0:
                lr = scheduler.get_last_lr()[0]
                log(f"[e{epoch+1}|u{updates}] loss={loss.item():.4f} lr={lr:.2e}")

        avg_loss = epoch_loss / max(1, n_batches)
        elapsed = time.time() - st
        log(f"Epoch {epoch+1}/{finetune_epochs}: avg_loss={avg_loss:.4f} ({elapsed:.1f}s)")

        with open(log_path, "a") as f:
            f.write(f"{epoch+1},{updates},{avg_loss:.6f},{scheduler.get_last_lr()[0]:.2e}\n")

        # ── Save checkpoint ────────────────────────────────────────
        ckpt = {
            "model_state_dict": model.state_dict(),
            "memory_state": model.get_memory_state(),
            "node_to_idx": node_to_idx,
            "edge_type_to_id": edge_type_to_id,
            "num_nodes": num_nodes,
            "num_edge_types": num_edge_types,
            "tgn_config": {
                "memory_dim": memory_dim,
                "edge_emb_dim": edge_emb_dim,
                "time_dim": time_dim,
                "entity_emb_dim": H,
                "num_heads": num_heads,
                "use_node_type_emb": use_node_type_emb,
                "use_time_emb": use_time_emb,
                "use_memory": use_memory,
                "use_entity_emb": use_entity_emb,
                "score_gated_memory": score_gated_memory,
                "temporal_decay": temporal_decay,
                "anomaly_accumulator": anomaly_accumulator,
                "objective": objective,
                "use_event_bert_emb": use_event_bert_emb,
                "event_bert_emb_dim": event_bert_emb_dim,
            },
        }
        save_path = os.path.join(ft_dir, f"finetune_tgn_{model_size}.pt")
        torch.save(ckpt, save_path)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(ft_dir, f"finetune_tgn_{model_size}_best.pt")
            torch.save(ckpt, best_path)

    log(f"TGN fine-tuning complete. Best loss: {best_loss:.4f}")


def main(cfg):
    log_start(__file__)

    pretrain_cfg = cfg.featurization.feat_training.spider
    detect_cfg = cfg.detection.gnn_training.spider
    model_type = pretrain_cfg.model_type
    model_size = pretrain_cfg.model_size
    finetune_mode = detect_cfg.finetune_mode

    # TGN uses a completely different pipeline (no walks, no BERT sequence classification)
    if finetune_mode == "tgn":
        _train_tgn(cfg)
        return

    if finetune_mode == "edge_cls":
        _train_edge_cls(cfg)
        return

    if finetune_mode == "perplexity":
        log("Perplexity mode: no fine-tuning needed (pretrained model used directly)")
        return

    finetune_epochs = detect_cfg.finetune_epochs
    finetune_walk_len = detect_cfg.finetune_walk_len
    finetune_lr = detect_cfg.finetune_lr
    finetune_margin = detect_cfg.finetune_margin
    freeze_backbone = detect_cfg.freeze_backbone
    if finetune_mode == "lp" and freeze_backbone:
        freeze_backbone = False  # LP has no extra parameters; must train the backbone
    batch_size = pretrain_cfg.batch_size
    num_walks = detect_cfg.num_inference_walks
    time_weight = pretrain_cfg.corpus.time_weight
    half_life = pretrain_cfg.corpus.half_life
    scheduler_type = pretrain_cfg.scheduler

    device = "cuda" if torch.cuda.is_available() and not cfg._use_cpu else "cpu"

    # ── Directories ────────────────────────────────────────────────────
    pretrain_dir = cfg.featurization.feat_training._model_dir
    ft_dir = cfg.detection.gnn_training._trained_models_dir
    os.makedirs(ft_dir, exist_ok=True)

    # ── Load tokenizer ──────────────────────────────────────────────────
    tokenizer = ProvenanceTokenizer(cfg)
    tokenizer.load(os.path.join(pretrain_dir, "tokenizer.pt"))
    log(f"Loaded tokenizer (vocab_size={tokenizer.vocab_size})")

    # ── Load pretrained model ───────────────────────────────────────────
    if model_type == "ropebert":
        rope_theta = getattr(pretrain_cfg, 'rope_theta', 10000.0)
        bert_config = get_ropebert_config(
            tokenizer.vocab_size, model_size, tokenizer.max_seq_len,
            rope_theta=rope_theta,
        )
        log(f"Using RoPEBERT architecture (theta={rope_theta})")
    elif model_type == "modernbert":
        bert_config = get_modernbert_config(
            tokenizer.vocab_size, model_size, tokenizer.max_seq_len,
            global_attn_every_n_layers=pretrain_cfg.global_attn_every_n_layers,
            local_attention_window=pretrain_cfg.local_attention_window,
        )
        log(f"Using ModernBERT architecture (global every {pretrain_cfg.global_attn_every_n_layers}, "
            f"window {pretrain_cfg.local_attention_window})")
    else:
        bert_config = get_bert_config(tokenizer.vocab_size, model_size, tokenizer.max_seq_len)
        log(f"Using traditional BERT architecture")

    pretrained_path = os.path.join(pretrain_dir, f"pretrain_{model_size}.pt")

    sd = torch.load(pretrained_path, weights_only=True, map_location="cpu")
    log(f"Loaded pretrained weights from {pretrained_path}")

    # cls_attack uses BCE (margin=0); cls uses the configured ranking margin
    cls_margin = 0 if finetune_mode == "cls_attack" else finetune_margin

    if model_type == "ropebert":
        if finetune_mode in ("cls", "cls_attack"):
            model = ProvenanceRoPEBERTFineTuneCLS(bert_config, sd, device=device, freeze_backbone=freeze_backbone, margin=cls_margin)
        elif finetune_mode == "lp":
            model = ProvenanceRoPEBERTFineTuneLP(bert_config, sd, device=device, freeze_backbone=freeze_backbone)
        else:
            raise ValueError(f"Unknown finetune mode: {finetune_mode}")
    elif model_type == "modernbert":
        if finetune_mode in ("cls", "cls_attack"):
            model = ProvenanceModernBERTFineTuneCLS(bert_config, sd, device=device, freeze_backbone=freeze_backbone, margin=cls_margin)
        elif finetune_mode == "lp":
            model = ProvenanceModernBERTFineTuneLP(bert_config, sd, device=device, freeze_backbone=freeze_backbone)
        else:
            raise ValueError(f"Unknown finetune mode: {finetune_mode}")
    else:
        if finetune_mode in ("cls", "cls_attack"):
            model = ProvenanceBERTFineTuneCLS(bert_config, sd, device=device, freeze_backbone=freeze_backbone, margin=cls_margin)
        elif finetune_mode == "lp":
            model = ProvenanceBERTFineTuneLP(bert_config, sd, device=device, freeze_backbone=freeze_backbone)
        else:
            raise ValueError(f"Unknown finetune mode: {finetune_mode}")

    log(f"Fine-tuning {model_type} in {finetune_mode} mode (freeze={freeze_backbone})")

    # ── Load data ───────────────────────────────────────────────────────
    indexid2msg = get_indexid2msg(cfg)
    base_dir = cfg.preprocessing.transformation._graphs_dir
    train_graph_paths = get_all_files_from_folders(base_dir, cfg.dataset.train_files)

    log(f"Loading training graphs...")
    train_graphs = [torch.load(p) for p in train_graph_paths]
    train_samplers = [
        ProvenanceWalkSampler(g, walk_length=finetune_walk_len, num_walks=num_walks, time_weight=time_weight, half_life=half_life)
        for g in train_graphs
    ]

    # Collect all training edges
    train_edges = []
    for g in train_graphs:
        train_edges.extend(_collect_edges(g))
    log(f"Training edges: {len(train_edges)}")

    # ── Negative sampling pre-computation (cls mode only) ───────────────
    # Build once before training. Groups nodes by (type, label) so each
    # negative has the same node type as dst but a different textual label
    # (different tokens). Also checks against the real edge set so we never
    # use an edge that already exists in the training graphs.
    train_neg_map = None
    if finetune_mode == "cls":
        type_to_labeled = _build_neg_index(indexid2msg)
        real_edge_set: set = set()
        for g in train_graphs:
            for src, dst, _, attrs in g.edges(data=True, keys=True):
                real_edge_set.add((str(src), str(dst), attrs.get("label", "")))
        train_neg_map = _precompute_negatives(
            train_edges, indexid2msg, type_to_labeled, real_edge_set
        )
        log(f"Pre-computed negatives for {len(train_neg_map)} unique train edge triples")

    # ── Attack walk corpus (cls_attack mode only) ────────────────────────
    attack_walks = None
    if finetune_mode == "cls_attack":
        num_attack_walks = detect_cfg.num_attack_walks
        # Curated file (hand-validated, node-filtered) takes priority when present
        curated_path = os.path.join(ft_dir, "attack_walks_curated.pkl")
        attack_walks_path = os.path.join(ft_dir, "attack_walks.pkl")
        if os.path.exists(curated_path):
            attack_walks = _load_attack_walks(curated_path)
            log("Using curated attack walks (hand-filtered)")
        elif os.path.exists(attack_walks_path):
            attack_walks = _load_attack_walks(attack_walks_path)
            # Re-extract if cached file has old 4-tuple format (need 6-tuple with dst walks)
            if attack_walks and len(attack_walks[0]) != 6:
                log("Stale cache (4-tuple format), re-extracting with dst walks...")
                attack_walks = None
        if attack_walks is None and not os.path.exists(curated_path):
            log("Extracting attack walks from val/test graphs...")
            attack_walks = _extract_attack_walks(
                cfg, finetune_walk_len, time_weight, half_life, num_attack_walks,
            )
            _save_attack_walks(attack_walks, attack_walks_path)
        log(f"Attack walk corpus: {len(attack_walks)} sequences")
        if not attack_walks:
            raise ValueError(
                "No attack walks found — check ground_truth_relative_path and "
                "attack_to_time_window in the dataset config"
            )

    # ── Optimizer ───────────────────────────────────────────────────────
    opt = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=finetune_lr,
        betas=(0.9, 0.99),
        eps=1e-10,
        weight_decay=0.02,
    )
    updates_per_epoch = len(train_edges) // batch_size
    total_steps = updates_per_epoch * finetune_epochs
    warmup_steps = max(1, total_steps // 20)  # 5% warmup
    if scheduler_type == "cosine":
        scheduler = WarmupCosineScheduler(opt, warmup_steps, total_steps)
    else:
        scheduler = WarmupLinearScheduler(opt, warmup_steps, total_steps)

    # ── Training loop ───────────────────────────────────────────────────
    log_path = os.path.join(ft_dir, f"finetune_{finetune_mode}_{model_size}_log.txt")
    with open(log_path, "w") as f:
        f.write("epoch,update,loss,lr\n")

    best_val_loss = float("inf")
    updates = 0

    for epoch in range(finetune_epochs):
        random.shuffle(train_edges)
        epoch_loss = 0
        n_batches = 0
        st = time.time()

        for batch_start in range(0, len(train_edges), batch_size):
            batch_edges = train_edges[batch_start : batch_start + batch_size]

            # Use sampler from first graph (or pick randomly)
            sampler = random.choice(train_samplers)

            model.train()
            if finetune_mode == "cls":
                input_ids, attn_mask, tgt_mask, labels = _prepare_finetune_batch_cls(
                    batch_edges, sampler, tokenizer, indexid2msg, finetune_walk_len,
                    neg_map=train_neg_map,
                )
                loss = model(
                    input_ids.to(device),
                    attn_mask.to(device),
                    tgt_mask.to(device),
                    labels.to(device),
                )
            elif finetune_mode == "cls_attack":
                attack_batch = random.choices(attack_walks, k=len(batch_edges))
                input_ids, attn_mask, tgt_mask, labels = _prepare_finetune_batch_cls_attack(
                    batch_edges, attack_batch, sampler, tokenizer, indexid2msg, finetune_walk_len,
                )
                loss = model(
                    input_ids.to(device),
                    attn_mask.to(device),
                    tgt_mask.to(device),
                    labels.to(device),
                )
            elif finetune_mode == "lp":
                input_ids, attn_mask, tgt_mask, tgt_ids = _prepare_finetune_batch_lp(
                    batch_edges, sampler, tokenizer, indexid2msg, finetune_walk_len
                )
                loss = model(
                    input_ids.to(device),
                    attn_mask.to(device),
                    tgt_mask.to(device),
                    tgt_ids.to(device),
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1
            updates += 1

            if updates % 50 == 0:
                lr = scheduler.get_last_lr()[0]
                log(f"[e{epoch+1}|u{updates}] loss={loss.item():.4f} lr={lr:.2e}")

        avg_loss = epoch_loss / max(1, n_batches)
        elapsed = time.time() - st
        log(f"Epoch {epoch+1}/{finetune_epochs}: avg_loss={avg_loss:.4f} ({elapsed:.1f}s)")

        with open(log_path, "a") as f:
            f.write(f"{epoch+1},{updates},{avg_loss:.6f},{scheduler.get_last_lr()[0]:.2e}\n")

        # Save checkpoint
        save_path = os.path.join(ft_dir, f"finetune_{finetune_mode}_{model_size}.pt")
        torch.save(model.state_dict(), save_path)

        if avg_loss < best_val_loss:
            best_val_loss = avg_loss
            best_path = os.path.join(ft_dir, f"finetune_{finetune_mode}_{model_size}_best.pt")
            torch.save(model.state_dict(), best_path)

    log(f"Fine-tuning complete. Best loss: {best_val_loss:.4f}")
