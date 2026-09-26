"""Compute static node embeddings with a pretrained SPIDER model.

For each node, tokenizes its (type, label) pair, runs through the frozen pretrained
encoder, and mean-pools over non-padding positions to produce a single embedding
vector.  This allows SPIDER to be used as a drop-in replacement for
word2vec or other featurization methods in the standard GNN pipeline.
"""

import os

import numpy as np
import torch

from pidsmaker.utils.utils import get_indexid2msg, log, log_start


class GraphBasedEncoder:
    """Wraps a pretrained GNN encoder (GraphMAE/GAE/DGI) for per-graph inference.

    Instead of pre-computing a single averaged embedding per node across all
    graphs (which causes data snooping on test sets), this encoder is called
    once per graph to produce embeddings using only that graph's structure.
    """

    def __init__(self, model, t5_encoder, tokenizer, indexid2msg, emb_dim, device):
        self.model = model
        self.t5_encoder = t5_encoder
        self.tokenizer = tokenizer
        self.indexid2msg = indexid2msg
        self.emb_dim = emb_dim
        self.device = device
        self.label_cache = {}  # Persists across graphs (labels are deterministic)

    def encode_graph(self, graph_path):
        """Encode nodes using only the given graph's structure.

        Returns:
            dict: {node_id (int): embedding (np.ndarray)} for all nodes in
                  this graph.  Nodes not present in this graph are NOT included.
        """
        import torch as _torch

        g = _torch.load(graph_path)
        graph_nodes = list(g.nodes())
        if not graph_nodes:
            return {}

        # Collect node labels for T5 encoding
        node_labels = []
        valid_nodes = []
        for n in graph_nodes:
            if n in self.indexid2msg:
                lbl = self.indexid2msg[n]
                lbl = tuple(lbl) if isinstance(lbl, list) else lbl
            else:
                lbl = ("subject", "unknown")
            node_labels.append(lbl)
            valid_nodes.append(n)

        # Encode uncached labels via T5
        uncached = [lbl for lbl in set(node_labels) if lbl not in self.label_cache]
        if uncached:
            with _torch.no_grad():
                new_embs = self.t5_encoder.encode_labels(uncached, self.tokenizer, self.device)
            for i, lbl in enumerate(uncached):
                self.label_cache[lbl] = new_embs[i]

        x = _torch.stack([self.label_cache[lbl] for lbl in node_labels])

        # Build bidirectional edge_index from graph
        node2local = {n: i for i, n in enumerate(valid_nodes)}
        edge_src, edge_dst = [], []
        for src, dst, _, attrs in g.edges(data=True, keys=True):
            if src in node2local and dst in node2local:
                edge_src.append(node2local[src])
                edge_dst.append(node2local[dst])
                edge_src.append(node2local[dst])
                edge_dst.append(node2local[src])

        if not edge_src:
            # No edges: return T5-only embeddings (no GNN context)
            indexid2vec = {}
            x_np = x.cpu().numpy()
            for i, n in enumerate(valid_nodes):
                emb = x_np[i].astype(np.float32)
                norm = np.linalg.norm(emb)
                if norm > 1e-12:
                    emb = emb / norm
                indexid2vec[int(n)] = emb
            return indexid2vec

        edge_index = _torch.tensor([edge_src, edge_dst], dtype=_torch.long, device=self.device)

        with _torch.no_grad():
            encoded = self.model.encode(x, edge_index)

        encoded_np = encoded.cpu().numpy()
        indexid2vec = {}
        for i, n in enumerate(valid_nodes):
            emb = encoded_np[i].astype(np.float32)
            norm = np.linalg.norm(emb)
            if norm > 1e-12:
                emb = emb / norm
            indexid2vec[int(n)] = emb

        return indexid2vec

    def cleanup(self):
        """Release GPU memory."""
        del self.model, self.t5_encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _embed_labels(bert, tokenizer, keys, hidden_size, device, batch_size=512):
    """Embed a list of (ntype, label) keys using frozen BERT. Returns {key: np.ndarray}."""
    unique_keys = list(dict.fromkeys(keys))  # deduplicate, preserve order
    result = {}

    for batch_start in range(0, len(unique_keys), batch_size):
        batch_keys = unique_keys[batch_start:batch_start + batch_size]
        batch_token_ids, batch_keys_valid = [], []
        for key in batch_keys:
            ntype, nlabel = key
            tids = tokenizer.tokenize_node(ntype, nlabel)
            if tids:
                batch_token_ids.append(tids)
                batch_keys_valid.append(key)
            else:
                result[key] = np.zeros(hidden_size, dtype=np.float32)

        if not batch_token_ids:
            continue

        max_len = min(max(len(t) for t in batch_token_ids), tokenizer.max_seq_len)
        B = len(batch_token_ids)
        input_ids = torch.full((B, max_len), tokenizer.pad_id, dtype=torch.long)
        attention_mask = torch.zeros(B, max_len, dtype=torch.bool)
        for i, tids in enumerate(batch_token_ids):
            L = min(len(tids), max_len)
            input_ids[i, :L] = torch.tensor(tids[:L])
            attention_mask[i, :L] = True

        with torch.no_grad():
            hidden = bert.modified_fwd(
                input_ids.to(device),
                attention_mask.to(device),
                labels=torch.full_like(input_ids, -100).to(device),
                skip_cls=True,
            )

        mask_f = attention_mask.to(device).unsqueeze(-1).float()
        pooled = (hidden * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
        pooled_np = pooled.cpu().numpy()

        for i, key in enumerate(batch_keys_valid):
            emb = pooled_np[i]
            norm = np.linalg.norm(emb)
            if norm > 1e-12:
                emb = emb / norm
            result[key] = emb.astype(np.float32)

    return result


def _maybe_embed_synthetic_nodes(cfg, bert, tokenizer, hidden_size, device):
    """Compute and save T5 embeddings for synthetic attack node labels if configured."""
    import yaml
    from pidsmaker.config.pipeline import ROOT_PROJECT_PATH

    try:
        obj_cfg = cfg.detection.gnn_training.decoder.predict_edge_supervised
        mode = getattr(obj_cfg, "mode", "").strip()
        attack_edges_path = getattr(obj_cfg, "attack_edges_path", None)
    except AttributeError:
        return

    if mode != "synthetic" or not attack_edges_path or attack_edges_path == "none":
        return

    if not os.path.isabs(attack_edges_path):
        attack_edges_path = os.path.join(ROOT_PROJECT_PATH, attack_edges_path)

    if not os.path.exists(attack_edges_path):
        raise FileNotFoundError(
            f"predict_edge_supervised (synthetic mode): attack_edges_path not found: {attack_edges_path}"
        )

    with open(attack_edges_path) as f:
        data = yaml.safe_load(f)
    attack_edges = data.get("attack_edges", [])

    keys = []
    for edge in attack_edges:
        keys.append((edge["src_type"], edge["src_label"]))
        keys.append((edge["dst_type"], edge["dst_label"]))

    log(f"Embedding {len(set(keys))} unique synthetic attack node labels ...")
    synth_embs = _embed_labels(bert, tokenizer, keys, hidden_size, device)

    out_path = os.path.join(cfg.featurization.feat_training._model_dir, "synthetic_attack_node_embeddings.pt")
    torch.save(synth_embs, out_path)
    log(f"Saved synthetic attack node embeddings -> {out_path}")


def _continue_pretrain_gnn_distill(cfg, model, tokenizer, pretrain_cfg, inference_cfg, device):
    """Continue pretraining the GNN distillation model on the target dataset.

    Builds a sampler from the target dataset's graphs, then runs additional
    GNN distillation training (same loss as pretraining) to adapt the model
    to the specific dataset before inference.
    """
    import random
    import time
    from collections import defaultdict
    from copy import deepcopy

    from torch.optim import AdamW

    from pidsmaker.spider.models.gnn_distill import (
        NeighborhoodGNNTeacher, build_edge_type_encoder,
        build_gnn_distill_batch,
    )
    from pidsmaker.spider.models.graphmae import sce_loss
    from pidsmaker.spider.data.sampler import ProvenanceWalkSampler
    from pidsmaker.utils.utils import get_all_files_from_folders, get_indexid2msg

    distill_cfg = pretrain_cfg.gnn_distill
    continue_epochs = inference_cfg.continue_pretrain_epochs
    lr_factor = inference_cfg.continue_pretrain_lr_factor
    gnn_edge_emb_dim = distill_cfg.emb_dim
    gnn_hidden_dim = distill_cfg.hidden_dim
    gnn_num_layers = distill_cfg.num_layers
    gnn_num_heads = distill_cfg.num_heads
    gnn_n_neighbors_min = distill_cfg.n_neighbors_min
    gnn_n_neighbors_max = distill_cfg.n_neighbors_max
    gnn_diverse = distill_cfg.diverse_neighbors
    distill_lambda = distill_cfg.loss_weight
    gnn_loss_lambda = distill_cfg.gnn_loss_weight
    gnn_lr = distill_cfg.gnn_lr
    ema_momentum = distill_cfg.ema_momentum
    batch_size = pretrain_cfg.batch_size

    log("Continue-pretraining (gnn_distill) on target dataset ...")

    # Expand IPs for domain adaptation: keep [PRIVATE_IP]/[PUBLIC_IP] + add raw IPs
    if tokenizer.normalize_netflow_ips:
        tokenizer.expand_netflow_ips = True
        log("  Enabled IP expansion for continue-pretraining ([CATEGORY] + raw IP)")

    # Load target dataset graphs
    base_dir = cfg.preprocessing.transformation._graphs_dir
    train_files = get_all_files_from_folders(base_dir, cfg.dataset.train_files)
    if not train_files:
        log("  No graph files found for continue-pretraining, skipping")
        return model

    import torch as _torch
    from pidsmaker.spider.pretrain.pretrain_common import _load_and_merge_graphs
    graph_context_mode = getattr(pretrain_cfg, "graph_context_mode", "single")
    graphs = _load_and_merge_graphs(train_files, graph_context_mode)
    log(f"  Loaded {len(graphs)} graph(s) from {len(train_files)} files")

    ds_indexid2msg = get_indexid2msg(cfg)
    # Build samplers
    sampler_pairs = []
    for g in graphs:
        sampler = ProvenanceWalkSampler(g, walk_length=30, num_walks=10)
        sampler_pairs.append((sampler, ds_indexid2msg))

    # Build edge type encoder
    edge_type2idx = build_edge_type_encoder(sampler_pairs)

    # Build GNN teacher
    t5_hidden_dim = model.config.d_model
    gnn_teacher = NeighborhoodGNNTeacher(
        node_dim=t5_hidden_dim,
        num_edge_types=max(len(edge_type2idx), 1),
        edge_emb_dim=gnn_edge_emb_dim,
        hidden_dim=gnn_hidden_dim,
        num_layers=gnn_num_layers,
        num_heads=gnn_num_heads,
        edge_projection=distill_cfg.edge_projection,
    ).to(device)

    # EMA teacher
    ema_teacher = deepcopy(model).to(device)
    ema_teacher.requires_grad_(False)
    ema_teacher.eval()

    # Collect entities grouped by label
    combined_indexid2msg = {}
    label_to_entities = defaultdict(list)
    for sidx, (sampler, im) in enumerate(sampler_pairs):
        for node in sampler.nodes:
            if node in im:
                ntype, nlabel = im[node]
                entity_ids = tokenizer.tokenize_node(ntype, nlabel)
                if entity_ids:
                    label_key = (ntype, nlabel)
                    label_to_entities[label_key].append((node, sidx, entity_ids))
                    combined_indexid2msg[node] = (ntype, nlabel)

    n_labels = len(label_to_entities)
    log(f"  {n_labels:,} unique labels for continue-pretraining")

    if n_labels == 0:
        log("  No trainable entities, skipping")
        return model

    # Optimizer — reduced LR with short linear warmup
    model.train()
    base_lr = pretrain_cfg.lr * lr_factor
    gnn_continue_lr = gnn_lr * lr_factor
    opt = AdamW([
        {"params": model.parameters(), "lr": base_lr},
        {"params": gnn_teacher.parameters(), "lr": gnn_continue_lr},
    ], betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01)
    # Store initial LRs for warmup
    for pg in opt.param_groups:
        pg["initial_lr"] = pg["lr"]

    # Linear warmup over 10% of first epoch
    warmup_steps = max(1, n_labels // batch_size // 10)
    total_steps = (n_labels // batch_size + 1) * continue_epochs

    processed_tokens = 0
    updates = 0
    epoch = 0
    st = time.time()

    while epoch < continue_epochs:
        epoch += 1
        epoch_entities = []
        for label_key, entities in label_to_entities.items():
            pick = entities[(epoch - 1) % len(entities)]
            epoch_entities.append(pick)
        random.shuffle(epoch_entities)

        for batch_start in range(0, len(epoch_entities), batch_size):
            batch_ents = epoch_entities[batch_start:batch_start + batch_size]
            B = len(batch_ents)
            if B == 0:
                continue

            enc_ids_list = [ent[2] for ent in batch_ents]
            max_enc = min(max(len(e) for e in enc_ids_list), tokenizer.max_seq_len)
            input_ids = _torch.full((B, max_enc), tokenizer.pad_id, dtype=_torch.long)
            attention_mask = _torch.zeros(B, max_enc, dtype=_torch.bool)
            for i, enc in enumerate(enc_ids_list):
                seq_len = min(len(enc), max_enc)
                input_ids[i, :seq_len] = _torch.tensor(enc[:seq_len])
                attention_mask[i, :seq_len] = True

            entity_nodes = [ent[0] for ent in batch_ents]
            sampler_idxs = [ent[1] for ent in batch_ents]

            gnn_batch = build_gnn_distill_batch(
                entity_nodes, sampler_idxs, sampler_pairs,
                combined_indexid2msg, edge_type2idx,
                n_neighbors_min=gnn_n_neighbors_min, n_neighbors_max=gnn_n_neighbors_max,
                diverse=gnn_diverse, device=device,
            )
            if gnn_batch is None:
                continue

            node_labels, edge_index, edge_types, batch_vec, center_idx, valid_mask = gnn_batch
            n_valid = valid_mask.sum().item()
            if n_valid == 0:
                continue

            # EMA encode neighbors
            unique_labels = list(set(node_labels))
            label_to_uidx = {lbl: i for i, lbl in enumerate(unique_labels)}
            ema_tids = []
            for ntype, nlabel in unique_labels:
                tids = tokenizer.tokenize_node(ntype, nlabel)
                ema_tids.append(tids if tids else [tokenizer.pad_id])
            max_ema_len = min(max(len(t) for t in ema_tids), tokenizer.max_seq_len)
            N_unique = len(ema_tids)
            ema_input_ids = _torch.full((N_unique, max_ema_len), tokenizer.pad_id, dtype=_torch.long)
            ema_attn_mask = _torch.zeros(N_unique, max_ema_len, dtype=_torch.bool)
            for i, tids in enumerate(ema_tids):
                L = min(len(tids), max_ema_len)
                ema_input_ids[i, :L] = _torch.tensor(tids[:L])
                ema_attn_mask[i, :L] = True

            with _torch.no_grad():
                ema_embeddings = ema_teacher.mean_pool(ema_input_ids, ema_attn_mask)
            node_emb_idx = [label_to_uidx[lbl] for lbl in node_labels]
            node_embeddings = ema_embeddings[_torch.tensor(node_emb_idx, dtype=_torch.long)]

            # Forward
            model.train()
            gnn_teacher.train()
            out = model.modified_fwd(input_ids.to(device), attention_mask.to(device))
            t5_projected = out.projected

            gnn_recon_loss, gnn_encoded = gnn_teacher(
                node_embeddings, edge_index, edge_types, batch_vec, center_idx,
            )
            gnn_targets = gnn_encoded[center_idx].detach()
            distill_loss = sce_loss(t5_projected[valid_mask], gnn_targets)
            total_loss = distill_lambda * distill_loss + gnn_loss_lambda * gnn_recon_loss

            total_loss.backward()
            _torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            _torch.nn.utils.clip_grad_norm_(gnn_teacher.parameters(), 5.0)
            opt.step()
            opt.zero_grad()

            # Linear warmup
            updates += 1
            if updates <= warmup_steps:
                warmup_scale = updates / warmup_steps
                for pg in opt.param_groups:
                    pg["lr"] = pg["initial_lr"] * warmup_scale if "initial_lr" in pg else pg["lr"]

            # EMA update
            with _torch.no_grad():
                for ema_p, student_p in zip(ema_teacher.parameters(), model.parameters()):
                    ema_p.data.mul_(ema_momentum).add_(student_p.data, alpha=1.0 - ema_momentum)

            n_tokens = int(attention_mask.sum().item())
            processed_tokens += n_tokens

            if updates % 50 == 0:
                log(f"  [continue] e{epoch}/{continue_epochs} step={updates} "
                    f"d_loss={distill_loss.item():.4f} g_loss={gnn_recon_loss.item():.4f} "
                    f"({time.time()-st:.1f}s)")

    model.eval()
    elapsed = time.time() - st
    log(f"  Continue-pretraining done: {updates} steps, {epoch} epochs, "
        f"{processed_tokens:,} tokens ({elapsed:.1f}s)")

    # Restore tokenizer for inference
    tokenizer.expand_netflow_ips = False

    # Clean up
    del gnn_teacher, ema_teacher, opt
    if _torch.cuda.is_available():
        _torch.cuda.empty_cache()

    return model


def _load_target_graphs_and_samplers(cfg, pretrain_cfg):
    """Load target dataset graphs and build walk samplers. Shared by all continue-pretrain functions."""
    from pidsmaker.spider.data.sampler import ProvenanceWalkSampler
    from pidsmaker.utils.utils import get_all_files_from_folders, get_indexid2msg

    base_dir = cfg.preprocessing.transformation._graphs_dir
    train_files = get_all_files_from_folders(base_dir, cfg.dataset.train_files)
    if not train_files:
        return None, None, None

    from pidsmaker.spider.pretrain.pretrain_common import _load_and_merge_graphs
    graph_context_mode = getattr(pretrain_cfg, "graph_context_mode", "single")
    graphs = _load_and_merge_graphs(train_files, graph_context_mode)
    log(f"  Loaded {len(graphs)} graph(s) from {len(train_files)} files")

    ds_indexid2msg = get_indexid2msg(cfg)
    corpus_cfg = getattr(pretrain_cfg, "corpus", pretrain_cfg)
    walk_length = corpus_cfg.walk_length
    num_walks = corpus_cfg.num_walks

    sampler_pairs = []
    for g in graphs:
        sampler = ProvenanceWalkSampler(g, walk_length=walk_length, num_walks=num_walks)
        sampler_pairs.append((sampler, ds_indexid2msg))

    return sampler_pairs, ds_indexid2msg, graphs


def _continue_pretrain_spider(cfg, model, tokenizer, pretrain_cfg, inference_cfg, device):
    """Continue pretraining a spider model on the target dataset.

    Rebuilds behavior signatures and entity classes from the target graph,
    then fine-tunes using the same SupCon + distillation loss as pretraining
    at a reduced LR.
    """
    import random
    import time
    from collections import defaultdict
    from copy import deepcopy

    import torch as _torch
    from torch.optim import AdamW

    from pidsmaker.spider.models.spider import (
        GNNClusterTeacher,
        build_gnn_distill_batch, build_edge_type_encoder,
        supervised_contrastive_loss, sce_loss,
        StudentSignatureHead, StudentClassHead, TeacherClassHead,
    )
    from pidsmaker.spider.data.behavior_signatures import (
        build_behavior_dataset,
        BehaviorLabelVocab,
    )
    from pidsmaker.spider.data.entity_classes import classify_entity
    from pidsmaker.spider.training_utils import build_pk_batches

    gc_cfg = pretrain_cfg.spider
    continue_epochs = inference_cfg.continue_pretrain_epochs
    lr_factor = inference_cfg.continue_pretrain_lr_factor
    batch_size = pretrain_cfg.batch_size

    gc_edge_emb_dim = gc_cfg.emb_dim
    gc_hidden_dim = gc_cfg.hidden_dim
    gc_proj_dim = gc_cfg.proj_dim
    gc_num_heads = gc_cfg.num_heads
    gc_n_neighbors_min = gc_cfg.n_neighbors_min
    gc_n_neighbors_max = gc_cfg.n_neighbors_max
    gc_diverse = gc_cfg.diverse_neighbors
    gc_filter_noisy = gc_cfg.filter_noisy_edges
    gc_ema_momentum = gc_cfg.ema_momentum
    gc_supcon_weight = gc_cfg.supcon_weight
    gc_distill_weight = gc_cfg.distill_weight
    gc_temperature = gc_cfg.temperature
    gc_min_sig_size = gc_cfg.min_signature_size
    gc_min_per_class = gc_cfg.min_entities_per_class
    gc_max_per_class = gc_cfg.max_entities_per_class
    gc_samples_per_class = gc_cfg.samples_per_class
    gc_distill_loss = getattr(gc_cfg, 'distill_loss', 'sce')
    gc_teacher_loss = getattr(gc_cfg, 'teacher_loss', 'contrastive')
    gc_teacher_data = getattr(gc_cfg, 'teacher_data', 'signature,gnn_emb')
    gc_student_only_mode = getattr(gc_cfg, 'student_only_mode', 'none')
    use_teacher = gc_student_only_mode == "none"

    log("Continue-pretraining (spider) on target dataset ...")

    # Expand IPs for domain adaptation
    if tokenizer.normalize_netflow_ips:
        tokenizer.expand_netflow_ips = True
        log("  Enabled IP expansion for continue-pretraining ([CATEGORY] + raw IP)")

    result = _load_target_graphs_and_samplers(cfg, pretrain_cfg)
    if result[0] is None:
        log("  No graph files found, skipping")
        return model
    sampler_pairs, ds_indexid2msg, _ = result

    combined_indexid2msg = {}
    for _, im in sampler_pairs:
        combined_indexid2msg.update(im)

    # Resolve pretrained model directory (needed for edge_type2idx + vocab)
    pretrain_dir = cfg.featurization.feat_training._model_dir
    _pmp = getattr(pretrain_cfg, 'spider_path', None)
    if _pmp:
        pretrain_dir = _pmp

    # Load edge type encoder from pretrained checkpoint to keep indices consistent
    import json as _json
    edge_type2idx_path = os.path.join(pretrain_dir, "edge_type2idx.json")
    if os.path.exists(edge_type2idx_path):
        with open(edge_type2idx_path) as f:
            edge_type2idx = _json.load(f)
        # Add any new edge types from the target dataset
        local_edge_type2idx = build_edge_type_encoder(sampler_pairs)
        for et in local_edge_type2idx:
            if et not in edge_type2idx:
                edge_type2idx[et] = len(edge_type2idx)
        log(f"  Loaded {len(edge_type2idx)} edge types from pretrained checkpoint "
            f"({len(local_edge_type2idx)} in target dataset)")
    else:
        edge_type2idx = build_edge_type_encoder(sampler_pairs)

    # Build behavior signatures from target graph
    label_to_signature, label_to_entities_raw, vocab = build_behavior_dataset(
        sampler_pairs, combined_indexid2msg,
        filter_noisy=gc_filter_noisy,
        min_signature_size=gc_min_sig_size,
    )
    log(f"  {len(label_to_signature):,} unique entity labels, "
        f"{vocab.size:,} behavior labels")

    if not label_to_signature:
        log("  No signatures, skipping")
        return model

    # Load pretrained vocab for target vectors (must match model's num_labels)
    pretrain_vocab = BehaviorLabelVocab()
    bvocab_path = os.path.join(pretrain_dir, "behavior_vocab.txt")
    if os.path.exists(bvocab_path):
        pretrain_vocab.load(bvocab_path)
    else:
        log("  No pretrained behavior_vocab.txt found, skipping")
        return model

    # Tokenize entities
    label_to_entities = defaultdict(list)
    for label_key, raw_entities in label_to_entities_raw.items():
        ntype, nlabel = label_key
        entity_ids = tokenizer.tokenize_node(ntype, nlabel)
        if entity_ids:
            for node_id, sidx in raw_entities:
                label_to_entities[label_key].append((node_id, sidx, entity_ids))

    # Build signature target vectors using pretrained vocab
    label_to_target = {}
    for label_key, sig in label_to_signature.items():
        if label_key in label_to_entities:
            label_to_target[label_key] = pretrain_vocab.signature_to_vector(sig)

    # Entity class IDs (coarse functional classes)
    entity_class_to_id = {}
    label_to_entity_class_id = {}
    for label_key in label_to_target:
        ntype, nlabel = label_key
        ecls = classify_entity(ntype, nlabel)
        if ecls not in entity_class_to_id:
            entity_class_to_id[ecls] = len(entity_class_to_id)
        label_to_entity_class_id[label_key] = entity_class_to_id[ecls]

    n_labels = len(label_to_target)
    log(f"  {n_labels:,} trainable labels, {len(entity_class_to_id):,} entity classes")

    if n_labels == 0:
        log("  No trainable entities, skipping")
        return model

    # P×K batching
    K = gc_samples_per_class
    P = batch_size // K

    # Build models
    t5_hidden_dim = model.config.d_model
    n_entity_classes = len(entity_class_to_id)

    # Student-only ablation heads
    student_sig_head = None
    student_cls_head = None
    if gc_student_only_mode == "student_signature":
        student_sig_head = StudentSignatureHead(gc_hidden_dim, pretrain_vocab.size).to(device)
    elif gc_student_only_mode == "student_class":
        student_cls_head = StudentClassHead(gc_hidden_dim, n_entity_classes).to(device)

    # Teacher components
    gnn_teacher = None
    ema_teacher = None
    teacher_cls_head = None
    model_size = pretrain_cfg.model_size

    if use_teacher:
        gnn_teacher = GNNClusterTeacher(
            node_dim=t5_hidden_dim,
            num_edge_types=max(len(edge_type2idx), 1),
            edge_emb_dim=gc_edge_emb_dim,
            hidden_dim=gc_hidden_dim,
            num_labels=pretrain_vocab.size,
            proj_dim=gc_proj_dim,
            num_heads=gc_num_heads,
            teacher_data=gc_teacher_data,
        ).to(device)

        # Load pretrained GNN teacher weights
        teacher_pt_path = os.path.join(pretrain_dir, f"gnn_teacher_{model_size}_best.pt")
        teacher_sd = torch.load(teacher_pt_path, weights_only=True, map_location="cpu")
        gnn_teacher.load_state_dict(teacher_sd, strict=False)
        log(f"  Loaded pretrained GNN teacher from {teacher_pt_path}")

        if gc_teacher_loss == "bce":
            teacher_cls_head = TeacherClassHead(gc_proj_dim, n_entity_classes).to(device)

        # EMA teacher
        ema_teacher = deepcopy(model).to(device)
        ema_teacher.requires_grad_(False)
        ema_teacher.eval()

    # Optimizer — reduced LR with short linear warmup
    model.train()
    base_lr = pretrain_cfg.lr * lr_factor
    all_params = list(model.parameters())
    if gnn_teacher is not None:
        all_params += list(gnn_teacher.parameters())
    if teacher_cls_head is not None:
        all_params += list(teacher_cls_head.parameters())
    if student_sig_head is not None:
        all_params += list(student_sig_head.parameters())
    if student_cls_head is not None:
        all_params += list(student_cls_head.parameters())
    opt = AdamW(
        all_params,
        lr=base_lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01,
    )
    for pg in opt.param_groups:
        pg["initial_lr"] = pg["lr"]

    warmup_steps = max(1, n_labels // batch_size // 10)

    processed_tokens = 0
    updates = 0
    epoch = 0
    st = time.time()

    # EMA embedding cache
    ema_cache = {}
    ema_cache_refresh_interval = 50
    ema_cache_last_refresh = -ema_cache_refresh_interval

    while epoch < continue_epochs:
        epoch += 1

        # P×K batch construction by entity class
        class_to_labels = defaultdict(list)
        for label_key in label_to_target:
            class_to_labels[label_to_entity_class_id[label_key]].append(label_key)

        epoch_entities_by_class = defaultdict(list)
        for class_id, class_label_keys in class_to_labels.items():
            n_class = len(class_label_keys)
            if gc_max_per_class > 0 and n_class > gc_max_per_class:
                offset = (epoch - 1) * gc_max_per_class % n_class
                indices = [(offset + i) % n_class for i in range(gc_max_per_class)]
                class_label_keys = [class_label_keys[i] for i in indices]
            elif gc_min_per_class > 0 and n_class < gc_min_per_class:
                repeats = (gc_min_per_class + n_class - 1) // n_class
                class_label_keys = (class_label_keys * repeats)[:gc_min_per_class]
            for label_key in class_label_keys:
                entities = label_to_entities[label_key]
                pick = entities[(epoch - 1) % len(entities)]
                epoch_entities_by_class[class_id].append((label_key, pick, class_id))

        for items in epoch_entities_by_class.values():
            random.shuffle(items)

        epoch_batches_list = build_pk_batches(epoch_entities_by_class, P, K)

        epoch_supcon = 0.0
        epoch_distill = 0.0
        epoch_batches = 0

        for batch_items in epoch_batches_list:
            B = len(batch_items)
            if B == 0:
                continue

            enc_ids_list = [item[1][2] for item in batch_items]
            max_enc = min(max(len(e) for e in enc_ids_list), tokenizer.max_seq_len)
            input_ids = _torch.full((B, max_enc), tokenizer.pad_id, dtype=_torch.long)
            attention_mask = _torch.zeros(B, max_enc, dtype=_torch.bool)
            for i, enc in enumerate(enc_ids_list):
                seq_len = min(len(enc), max_enc)
                input_ids[i, :seq_len] = _torch.tensor(enc[:seq_len])
                attention_mask[i, :seq_len] = True

            entity_class_ids = _torch.tensor(
                [label_to_entity_class_id[item[0]] for item in batch_items], dtype=_torch.long,
            )

            # Build GNN neighborhood batch
            entity_nodes = [item[1][0] for item in batch_items]
            sampler_idxs = [item[1][1] for item in batch_items]

            gnn_batch = build_gnn_distill_batch(
                entity_nodes, sampler_idxs, sampler_pairs,
                combined_indexid2msg, edge_type2idx,
                n_neighbors_min=gc_n_neighbors_min, n_neighbors_max=gc_n_neighbors_max,
                diverse=gc_diverse, device=device,
                filter_noisy=gc_filter_noisy,
            )
            if gnn_batch is None:
                continue

            node_labels, edge_index, edge_types, batch_vec, center_idx, valid_mask = gnn_batch
            n_valid = valid_mask.sum().item()
            if n_valid == 0:
                continue

            # Encode neighbor nodes with EMA teacher (cached)
            if use_teacher:
                need_refresh = (updates - ema_cache_last_refresh) >= ema_cache_refresh_interval
                uncached_labels = [lbl for lbl in set(node_labels) if lbl not in ema_cache]

                if need_refresh:
                    all_labels_to_encode = list(set(node_labels))
                    ema_cache_last_refresh = updates
                elif uncached_labels:
                    all_labels_to_encode = uncached_labels
                else:
                    all_labels_to_encode = []

                if all_labels_to_encode:
                    ema_token_ids = []
                    for ntype, nlabel in all_labels_to_encode:
                        tids = tokenizer.tokenize_node(ntype, nlabel)
                        ema_token_ids.append(tids if tids else [tokenizer.pad_id])

                    max_ema_len = min(max(len(t) for t in ema_token_ids), tokenizer.max_seq_len)
                    N_unique = len(ema_token_ids)
                    ema_input_ids = _torch.full((N_unique, max_ema_len), tokenizer.pad_id, dtype=_torch.long)
                    ema_attn_mask = _torch.zeros(N_unique, max_ema_len, dtype=_torch.bool)
                    for i, tids in enumerate(ema_token_ids):
                        L = min(len(tids), max_ema_len)
                        ema_input_ids[i, :L] = _torch.tensor(tids[:L])
                        ema_attn_mask[i, :L] = True

                    ema_teacher.eval()
                    with _torch.no_grad():
                        ema_embeddings = ema_teacher.mean_pool(ema_input_ids, ema_attn_mask)
                    for i, lbl in enumerate(all_labels_to_encode):
                        ema_cache[lbl] = ema_embeddings[i]

                node_embeddings = _torch.stack([ema_cache[lbl] for lbl in node_labels])

            # Forward passes
            model.train()

            out = model.modified_fwd(input_ids.to(device), attention_mask.to(device))
            t5_projected = out.projected

            valid_mask_cpu = valid_mask.cpu()
            valid_class_ids = entity_class_ids[valid_mask_cpu].to(device)

            if use_teacher:
                gnn_teacher.train()
                if teacher_cls_head is not None:
                    teacher_cls_head.train()

                valid_label_keys = [batch_items[i][0] for i in range(B) if valid_mask_cpu[i]]
                sig_vecs = _torch.stack([label_to_target[lk] for lk in valid_label_keys]).to(device)

                teacher_proj = gnn_teacher(
                    node_embeddings, edge_index, edge_types, center_idx,
                    signature_vectors=sig_vecs,
                )

                # L_teacher: SupCon or BCE
                if gc_teacher_loss == "bce":
                    import torch.nn.functional as _F
                    cls_logits = teacher_cls_head(teacher_proj)
                    supcon_loss = _F.cross_entropy(cls_logits, valid_class_ids)
                else:
                    supcon_loss = supervised_contrastive_loss(
                        teacher_proj, valid_class_ids, temperature=gc_temperature,
                    )

                # L_distill: SCE or MSE
                if gc_distill_loss == "mse":
                    import torch.nn.functional as _F
                    distill_loss = _F.mse_loss(t5_projected[valid_mask], teacher_proj.detach())
                else:
                    distill_loss = sce_loss(t5_projected[valid_mask], teacher_proj.detach())

                total_loss = gc_supcon_weight * supcon_loss + gc_distill_weight * distill_loss
            else:
                import torch.nn.functional as _F
                valid_label_keys = [batch_items[i][0] for i in range(B) if valid_mask_cpu[i]]

                if gc_student_only_mode == "student_signature":
                    student_sig_head.train()
                    sig_logits = student_sig_head(t5_projected[valid_mask])
                    sig_targets = _torch.stack([label_to_target[lk] for lk in valid_label_keys]).to(device)
                    supcon_loss = _F.binary_cross_entropy_with_logits(sig_logits, sig_targets)
                    distill_loss = _torch.tensor(0.0, device=device)
                elif gc_student_only_mode == "student_class":
                    student_cls_head.train()
                    cls_logits = student_cls_head(t5_projected[valid_mask])
                    supcon_loss = _F.cross_entropy(cls_logits, valid_class_ids)
                    distill_loss = _torch.tensor(0.0, device=device)

                total_loss = supcon_loss

            total_loss.backward()
            _torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            if gnn_teacher is not None:
                _torch.nn.utils.clip_grad_norm_(gnn_teacher.parameters(), 5.0)
            opt.step()
            opt.zero_grad()

            # EMA update
            if ema_teacher is not None:
                with _torch.no_grad():
                    for ema_p, student_p in zip(ema_teacher.parameters(), model.parameters()):
                        ema_p.data.mul_(gc_ema_momentum).add_(student_p.data, alpha=1.0 - gc_ema_momentum)

            # Linear warmup
            updates += 1
            if updates <= warmup_steps:
                warmup_scale = updates / warmup_steps
                for pg in opt.param_groups:
                    pg["lr"] = pg["initial_lr"] * warmup_scale

            n_tokens = int(attention_mask.sum().item())
            processed_tokens += n_tokens

            sc_val = supcon_loss.item()
            dl_val = distill_loss.item()
            epoch_supcon += sc_val
            epoch_distill += dl_val
            epoch_batches += 1

            if updates % 50 == 0:
                log(f"  [continue] e{epoch}/{continue_epochs} step={updates} "
                    f"supcon={sc_val:.4f} distill={dl_val:.4f} "
                    f"valid={n_valid}/{B} ({time.time()-st:.1f}s)")

        if epoch_batches > 0:
            avg_sc = epoch_supcon / epoch_batches
            avg_dl = epoch_distill / epoch_batches
            log(f"  Epoch {epoch}/{continue_epochs} | supcon={avg_sc:.4f} distill={avg_dl:.4f} | "
                f"{time.time()-st:.1f}s")

    model.eval()
    log(f"  Continue-pretraining (spider) done: {updates} steps, {epoch} epochs, "
        f"{processed_tokens:,} tokens ({time.time()-st:.1f}s)")

    # Restore tokenizer for inference
    tokenizer.expand_netflow_ips = False

    del gnn_teacher, ema_teacher, opt, ema_cache
    if _torch.cuda.is_available():
        _torch.cuda.empty_cache()
    return model


def _continue_pretrain_behavior_cluster(cfg, model, tokenizer, pretrain_cfg, inference_cfg, device):
    """Continue pretraining a behavior_cluster model on the target dataset.

    Rebuilds behavior signatures from the target graph and fine-tunes the
    model using the same BCE + SupCon loss as pretraining, at a reduced LR.
    """
    import random
    import time
    from collections import defaultdict

    import torch as _torch
    from torch.optim import AdamW

    from pidsmaker.spider.models.behavior import behavior_combined_loss
    from pidsmaker.spider.data.behavior_signatures import (
        build_behavior_dataset,
        BehaviorLabelVocab,
    )

    bc_cfg = pretrain_cfg.behavior_cluster
    continue_epochs = inference_cfg.continue_pretrain_epochs
    lr_factor = inference_cfg.continue_pretrain_lr_factor
    batch_size = pretrain_cfg.batch_size
    bc_bce_weight = bc_cfg.bce_weight
    bc_contrastive_weight = bc_cfg.contrastive_weight
    bc_temperature = bc_cfg.temperature
    bc_min_sig_size = bc_cfg.min_signature_size
    bc_filter_noisy = bc_cfg.filter_noisy_edges
    bc_bce_mode = bc_cfg.bce_mode
    bc_contrastive_target = bc_cfg.contrastive_target

    log("Continue-pretraining (behavior_cluster) on target dataset ...")

    result = _load_target_graphs_and_samplers(cfg, pretrain_cfg)
    if result[0] is None:
        log("  No graph files found, skipping")
        return model
    sampler_pairs, ds_indexid2msg, _ = result

    # Build behavior signatures from target graph
    combined_indexid2msg = {}
    for _, im in sampler_pairs:
        combined_indexid2msg.update(im)

    label_to_signature, label_to_entities_raw, vocab = build_behavior_dataset(
        sampler_pairs, combined_indexid2msg,
        filter_noisy=bc_filter_noisy,
        min_signature_size=bc_min_sig_size,
    )
    log(f"  {len(label_to_signature):,} unique entity labels, "
        f"{vocab.size:,} behavior labels")

    if not label_to_signature:
        log("  No signatures, skipping")
        return model

    # Load pretrained vocab for target vectors (must match model's num_labels)
    pretrain_dir = cfg.featurization.feat_training._model_dir
    _pmp = getattr(pretrain_cfg, 'spider_path', None)
    if _pmp:
        pretrain_dir = _pmp
    pretrain_vocab = BehaviorLabelVocab()
    bvocab_path = os.path.join(pretrain_dir, "behavior_vocab.txt")
    if os.path.exists(bvocab_path):
        pretrain_vocab.load(bvocab_path)
    else:
        log("  No pretrained behavior_vocab.txt found, skipping")
        return model

    # Tokenize entities and build targets using pretrained vocab
    label_to_entities = defaultdict(list)
    for label_key, raw_entities in label_to_entities_raw.items():
        ntype, nlabel = label_key
        entity_ids = tokenizer.tokenize_node(ntype, nlabel)
        if entity_ids:
            for node_id, sidx in raw_entities:
                label_to_entities[label_key].append((node_id, sidx, entity_ids))

    label_to_target = {}
    for label_key, sig in label_to_signature.items():
        if label_key in label_to_entities:
            label_to_target[label_key] = pretrain_vocab.signature_to_vector(sig)

    # Assign class IDs (identical signatures → same class)
    sig_to_class_id = {}
    label_to_class_id = {}
    for label_key, sig in label_to_signature.items():
        if label_key in label_to_target:
            if sig not in sig_to_class_id:
                sig_to_class_id[sig] = len(sig_to_class_id)
            label_to_class_id[label_key] = sig_to_class_id[sig]

    # Assign entity class IDs (coarse functional classes)
    from pidsmaker.spider.data.entity_classes import classify_entity
    entity_class_to_id = {}
    label_to_entity_class_id = {}
    for label_key in label_to_target:
        ntype, nlabel = label_key
        ecls = classify_entity(ntype, nlabel)
        if ecls not in entity_class_to_id:
            entity_class_to_id[ecls] = len(entity_class_to_id)
        label_to_entity_class_id[label_key] = entity_class_to_id[ecls]

    # Contrastive grouping
    if bc_contrastive_target == "entity_class":
        label_to_contrastive_id = label_to_entity_class_id
    else:
        label_to_contrastive_id = label_to_class_id

    n_labels = len(label_to_target)
    log(f"  {n_labels:,} trainable labels, {len(sig_to_class_id):,} signature classes, "
        f"{len(entity_class_to_id):,} entity classes")

    if n_labels == 0:
        log("  No trainable entities, skipping")
        return model

    # Optimizer — reduced LR with short linear warmup
    model.train()
    base_lr = pretrain_cfg.lr * lr_factor
    opt = AdamW(model.parameters(), lr=base_lr, betas=(0.9, 0.95), weight_decay=0.01)
    for pg in opt.param_groups:
        pg["initial_lr"] = pg["lr"]

    warmup_steps = max(1, n_labels // batch_size // 10)

    processed_tokens = 0
    updates = 0
    epoch = 0
    st = time.time()

    while epoch < continue_epochs:
        epoch += 1

        # Group by contrastive class for pairing
        class_to_labels = defaultdict(list)
        for label_key in label_to_target:
            class_to_labels[label_to_contrastive_id[label_key]].append(label_key)

        epoch_entities = []
        for class_id, class_label_keys in class_to_labels.items():
            for label_key in class_label_keys:
                entities = label_to_entities[label_key]
                pick = entities[(epoch - 1) % len(entities)]
                epoch_entities.append((label_key, pick, class_id))

        # Sort by class_id so same-class entities are adjacent in batches
        epoch_entities.sort(key=lambda x: x[2])
        chunks = [epoch_entities[i:i + batch_size] for i in range(0, len(epoch_entities), batch_size)]
        random.shuffle(chunks)
        epoch_entities = [item for chunk in chunks for item in chunk]

        epoch_loss = 0.0
        epoch_batches = 0

        for batch_start in range(0, len(epoch_entities), batch_size):
            batch_items = epoch_entities[batch_start:batch_start + batch_size]
            B = len(batch_items)
            if B == 0:
                continue

            enc_ids_list = [item[1][2] for item in batch_items]
            max_enc = min(max(len(e) for e in enc_ids_list), tokenizer.max_seq_len)
            input_ids = _torch.full((B, max_enc), tokenizer.pad_id, dtype=_torch.long)
            attention_mask = _torch.zeros(B, max_enc, dtype=_torch.bool)
            for i, enc in enumerate(enc_ids_list):
                seq_len = min(len(enc), max_enc)
                input_ids[i, :seq_len] = _torch.tensor(enc[:seq_len])
                attention_mask[i, :seq_len] = True

            target_vecs = _torch.stack([label_to_target[item[0]] for item in batch_items])
            class_ids = _torch.tensor([item[2] for item in batch_items], dtype=_torch.long)
            entity_class_ids = _torch.tensor(
                [label_to_entity_class_id[item[0]] for item in batch_items], dtype=_torch.long,
            )

            out = model.modified_fwd(input_ids.to(device), attention_mask.to(device))
            total_loss, bce_val, con_val = behavior_combined_loss(
                out.logits, target_vecs.to(device),
                out.projection, class_ids.to(device),
                bce_weight=bc_bce_weight,
                contrastive_weight=bc_contrastive_weight,
                temperature=bc_temperature,
                bce_mode=bc_bce_mode,
                entity_class_labels=entity_class_ids.to(device),
                contrastive_target=bc_contrastive_target,
            )

            total_loss.backward()
            _torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            opt.zero_grad()

            epoch_loss += total_loss.item()
            epoch_batches += 1

            # Linear warmup
            updates += 1
            if updates <= warmup_steps:
                warmup_scale = updates / warmup_steps
                for pg in opt.param_groups:
                    pg["lr"] = pg["initial_lr"] * warmup_scale

            processed_tokens += int(attention_mask.sum().item())

        avg_loss = epoch_loss / max(epoch_batches, 1)
        elapsed = time.time() - st
        log(f"  Epoch {epoch}/{continue_epochs} | loss={avg_loss:.4f} | "
            f"lr={opt.param_groups[0]['lr']:.2e} | {elapsed:.1f}s")

    model.eval()
    log(f"  Continue-pretraining (behavior_cluster) done: {updates} steps, {epoch} epochs, "
        f"{processed_tokens:,} tokens ({time.time()-st:.1f}s)")

    del opt
    if _torch.cuda.is_available():
        _torch.cuda.empty_cache()
    return model


def _continue_pretrain_t5(cfg, model, tokenizer, pretrain_cfg, inference_cfg, device):
    """Continue pretraining a T5 model on the target dataset using walk-based next-token prediction."""
    import random
    import time

    import torch as _torch
    from torch.optim import AdamW

    corpus_cfg = getattr(pretrain_cfg, "corpus", pretrain_cfg)
    continue_epochs = inference_cfg.continue_pretrain_epochs
    lr_factor = inference_cfg.continue_pretrain_lr_factor
    batch_size = pretrain_cfg.batch_size
    log("Continue-pretraining (T5) on target dataset ...")

    # Expand IPs for domain adaptation: keep [PRIVATE_IP]/[PUBLIC_IP] + add raw IPs
    if tokenizer.normalize_netflow_ips:
        tokenizer.expand_netflow_ips = True
        log("  Enabled IP expansion for continue-pretraining ([CATEGORY] + raw IP)")

    result = _load_target_graphs_and_samplers(cfg, pretrain_cfg)
    if result[0] is None:
        log("  No graph files found, skipping")
        return model
    sampler_pairs, ds_indexid2msg, _ = result

    # Sample walks and pre-tokenize as encoder-decoder pairs
    pretokenized = []
    seen = set()
    for sampler, im in sampler_pairs:
        for node in sampler.nodes:
            if node not in im:
                continue
            ntype, nlabel = im[node]
            entity_ids = tokenizer.tokenize_node(ntype, nlabel)
            if not entity_ids:
                continue
            for _ in range(corpus_cfg.num_walks):
                walk_nodes, walk_edge_types, _, entity_pos = sampler._single_walk(node)
                if len(walk_nodes) < 2:
                    continue
                # Forward context: nodes/edges after entity position
                if entity_pos < len(walk_nodes) - 1:
                    fwd_edges = walk_edge_types[entity_pos:]
                    fwd_nodes = walk_nodes[entity_pos + 1:]
                    dir_id = tokenizer.forward_id
                    # canonical=False: during continue_pretrain, keep all tokens
                    # with category tokens prepended (full detail for dataset-specific fine-tuning)
                    decoder_ids = [dir_id] + tokenizer.tokenize_context_segment(
                        fwd_edges, fwd_nodes, im, canonical=False,
                    )
                    if len(decoder_ids) > 1:
                        sig = (tuple(entity_ids), tuple(decoder_ids))
                        if sig not in seen:
                            seen.add(sig)
                            pretokenized.append((entity_ids, decoder_ids))
                # Backward context: nodes/edges before entity position
                if entity_pos > 0:
                    bwd_edges = list(reversed(walk_edge_types[:entity_pos]))
                    bwd_nodes = list(reversed(walk_nodes[:entity_pos]))
                    dir_id = tokenizer.backward_id
                    decoder_ids = [dir_id] + tokenizer.tokenize_context_segment(
                        bwd_edges, bwd_nodes, im, canonical=False,
                    )
                    if len(decoder_ids) > 1:
                        sig = (tuple(entity_ids), tuple(decoder_ids))
                        if sig not in seen:
                            seen.add(sig)
                            pretokenized.append((entity_ids, decoder_ids))

    log(f"  Pre-tokenized {len(pretokenized):,} walks for continue-pretraining")
    if not pretokenized:
        log("  No walks, skipping")
        return model

    # Training loop — reduced LR with short linear warmup
    model.train()
    base_lr = pretrain_cfg.lr * lr_factor
    opt = AdamW(model.parameters(), lr=base_lr, betas=(0.9, 0.95), weight_decay=0.01)
    for pg in opt.param_groups:
        pg["initial_lr"] = pg["lr"]

    warmup_steps = max(1, len(pretokenized) // batch_size // 10)

    processed_tokens = 0
    updates = 0
    epoch = 0
    st = time.time()
    pad_id = tokenizer.pad_id

    while epoch < continue_epochs:
        epoch += 1
        random.shuffle(pretokenized)
        epoch_loss = 0.0
        epoch_batches = 0
        epoch_st = time.time()

        for batch_start in range(0, len(pretokenized), batch_size):
            batch = pretokenized[batch_start:batch_start + batch_size]
            B = len(batch)
            if B == 0:
                continue

            enc_list = [b[0] for b in batch]
            dec_list = [b[1] for b in batch]
            max_enc = min(max(len(e) for e in enc_list), tokenizer.max_seq_len)
            max_dec = min(max(len(d) for d in dec_list), tokenizer.max_seq_len)

            input_ids = _torch.full((B, max_enc), pad_id, dtype=_torch.long)
            attention_mask = _torch.zeros(B, max_enc, dtype=_torch.bool)
            labels = _torch.full((B, max_dec), -100, dtype=_torch.long)

            for i in range(B):
                el = min(len(enc_list[i]), max_enc)
                input_ids[i, :el] = _torch.tensor(enc_list[i][:el])
                attention_mask[i, :el] = True
                dl = min(len(dec_list[i]), max_dec)
                labels[i, :dl] = _torch.tensor(dec_list[i][:dl])

            loss = model.modified_fwd(
                input_ids.to(device), attention_mask.to(device), labels.to(device),
            )

            loss.backward()
            _torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            opt.zero_grad()

            loss_val = loss.item()
            epoch_loss += loss_val
            epoch_batches += 1

            # Linear warmup
            updates += 1
            if updates <= warmup_steps:
                warmup_scale = updates / warmup_steps
                for pg in opt.param_groups:
                    pg["lr"] = pg["initial_lr"] * warmup_scale

            n_tokens = int(attention_mask.sum().item())
            processed_tokens += n_tokens

        avg_loss = epoch_loss / max(epoch_batches, 1)
        elapsed = time.time() - epoch_st
        log(f"  Epoch {epoch}/{continue_epochs} | loss={avg_loss:.4f} | "
            f"lr={opt.param_groups[0]['lr']:.2e} | {elapsed:.1f}s")

    model.eval()
    log(f"  Continue-pretraining (T5) done: {updates} steps, {epoch} epochs, "
        f"{processed_tokens:,} tokens ({time.time()-st:.1f}s)")

    # Restore tokenizer for inference
    tokenizer.expand_netflow_ips = False

    del opt
    if _torch.cuda.is_available():
        _torch.cuda.empty_cache()
    return model


def main(cfg):
    log_start(__file__)

    pretrain_cfg = cfg.featurization.feat_training.spider
    model_type = pretrain_cfg.model_type
    model_size = pretrain_cfg.model_size
    pretrain_dir = cfg.featurization.feat_training._model_dir
    # spider_path overrides pretrain_dir for loading artifacts
    _spider_path = getattr(pretrain_cfg, 'spider_path', None)
    if _spider_path:
        pretrain_dir = _spider_path
    emb_dim = cfg.featurization.feat_training.emb_dim

    # ── GraphMAE: return per-graph encoder (avoids cross-graph data snooping) ──
    if model_type == "graphmae":
        from pidsmaker.spider.models.graphmae import load_graphmae, T5NodeEncoder
        from pidsmaker.spider.data.tokenizer_bpe import ProvenanceTokenizerBPE as ProvenanceTokenizer

        gm_cfg = pretrain_cfg.graphmae
        gm_num_layers = gm_cfg.num_layers
        gm_num_heads = gm_cfg.num_heads
        gm_decoder_layers = gm_cfg.decoder_num_layers
        gm_mask_rate = gm_cfg.mask_rate
        gm_replace_rate = gm_cfg.replace_rate

        device = "cuda" if torch.cuda.is_available() and not getattr(cfg, '_use_cpu', False) else "cpu"

        tokenizer = ProvenanceTokenizer(cfg)
        tokenizer.load(os.path.join(pretrain_dir, "tokenizer.pt"))

        from pidsmaker.spider.models.gnn_distill import get_gnn_distill_encoder_config
        t5_config = get_gnn_distill_encoder_config(tokenizer.vocab_size, model_size, tokenizer.max_seq_len)
        t5_hidden_dim = t5_config.d_model

        gm_model, t5_encoder = load_graphmae(
            pretrain_dir, in_dim=t5_hidden_dim, hidden_dim=emb_dim,
            num_encoder_layers=gm_num_layers, num_decoder_layers=gm_decoder_layers,
            num_heads=gm_num_heads, mask_rate=gm_mask_rate, replace_rate=gm_replace_rate,
            vocab_size=tokenizer.vocab_size, model_size=model_size,
            max_seq_len=tokenizer.max_seq_len,
        )
        gm_model = gm_model.to(device).eval()
        t5_encoder = t5_encoder.to(device).eval()
        indexid2msg = get_indexid2msg(cfg)

        log(f"GraphMAE encoder ready (dim={emb_dim}) — embeddings will be computed per-graph")
        return GraphBasedEncoder(gm_model, t5_encoder, tokenizer, indexid2msg, emb_dim, device)

    # ── GAE: return per-graph encoder (avoids cross-graph data snooping) ──
    if model_type == "gae":
        from pidsmaker.spider.models.gae import load_gae
        from pidsmaker.spider.models.graphmae import T5NodeEncoder
        from pidsmaker.spider.models.gnn_distill import get_gnn_distill_encoder_config
        from pidsmaker.spider.data.tokenizer_bpe import ProvenanceTokenizerBPE as ProvenanceTokenizer

        gae_cfg = pretrain_cfg.gae
        gae_num_layers = gae_cfg.num_layers
        gae_num_heads = gae_cfg.num_heads

        device = "cuda" if torch.cuda.is_available() and not getattr(cfg, '_use_cpu', False) else "cpu"

        tokenizer = ProvenanceTokenizer(cfg)
        tokenizer.load(os.path.join(pretrain_dir, "tokenizer.pt"))

        t5_config = get_gnn_distill_encoder_config(tokenizer.vocab_size, model_size, tokenizer.max_seq_len)
        t5_hidden_dim = t5_config.d_model

        gae_model, t5_encoder = load_gae(
            pretrain_dir, in_dim=t5_hidden_dim, hidden_dim=emb_dim,
            num_encoder_layers=gae_num_layers, num_heads=gae_num_heads,
            vocab_size=tokenizer.vocab_size, model_size=model_size,
            max_seq_len=tokenizer.max_seq_len,
        )
        gae_model = gae_model.to(device).eval()
        t5_encoder = t5_encoder.to(device).eval()
        indexid2msg = get_indexid2msg(cfg)

        log(f"GAE encoder ready (dim={emb_dim}) — embeddings will be computed per-graph")
        return GraphBasedEncoder(gae_model, t5_encoder, tokenizer, indexid2msg, emb_dim, device)

    # ── DGI: return per-graph encoder (avoids cross-graph data snooping) ──
    if model_type == "dgi":
        from pidsmaker.spider.models.dgi import load_dgi
        from pidsmaker.spider.models.graphmae import T5NodeEncoder
        from pidsmaker.spider.models.gnn_distill import get_gnn_distill_encoder_config
        from pidsmaker.spider.data.tokenizer_bpe import ProvenanceTokenizerBPE as ProvenanceTokenizer

        dgi_cfg = pretrain_cfg.dgi
        dgi_num_layers = dgi_cfg.num_layers
        dgi_num_heads = dgi_cfg.num_heads

        device = "cuda" if torch.cuda.is_available() and not getattr(cfg, '_use_cpu', False) else "cpu"

        tokenizer = ProvenanceTokenizer(cfg)
        tokenizer.load(os.path.join(pretrain_dir, "tokenizer.pt"))

        t5_config = get_gnn_distill_encoder_config(tokenizer.vocab_size, model_size, tokenizer.max_seq_len)
        t5_hidden_dim = t5_config.d_model

        dgi_model, t5_encoder = load_dgi(
            pretrain_dir, in_dim=t5_hidden_dim, hidden_dim=emb_dim,
            num_encoder_layers=dgi_num_layers, num_heads=dgi_num_heads,
            vocab_size=tokenizer.vocab_size, model_size=model_size,
            max_seq_len=tokenizer.max_seq_len,
        )
        dgi_model = dgi_model.to(device).eval()
        t5_encoder = t5_encoder.to(device).eval()
        indexid2msg = get_indexid2msg(cfg)

        log(f"DGI encoder ready (dim={emb_dim}) — embeddings will be computed per-graph")
        return GraphBasedEncoder(dgi_model, t5_encoder, tokenizer, indexid2msg, emb_dim, device)

    # ── DeepWalk: load Word2Vec model and map labels directly ──────────
    if model_type in ("deepwalk", "node2vec"):
        from pidsmaker.spider.models.deepwalk import load_deepwalk, deepwalk_embeddings
        dw_model = load_deepwalk(pretrain_dir)
        if emb_dim != dw_model.wv.vector_size:
            raise ValueError(
                f"featurization.feat_training.emb_dim ({emb_dim}) does not match "
                f"DeepWalk vector_size ({dw_model.wv.vector_size}). "
                f"Set emb_dim: {dw_model.wv.vector_size} in your config."
            )
        indexid2msg = get_indexid2msg(cfg)
        indexid2vec = deepwalk_embeddings(dw_model, indexid2msg)
        log(f"Computed {len(indexid2vec):,} DeepWalk node embeddings (dim={emb_dim})")
        return indexid2vec

    # ── HF pretrained models: load fine-tuned model + HF tokenizer ───────
    from pidsmaker.spider.models.hf_pretrained import HF_MODEL_TYPES
    if model_type in HF_MODEL_TYPES:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from pidsmaker.spider.models.hf_pretrained import (
            embed_nodes_hf, get_hf_hidden_size,
        )

        hidden_size = get_hf_hidden_size(model_type, model_size)
        if emb_dim != hidden_size:
            log(f"Overriding emb_dim {emb_dim} → {hidden_size} to match "
                f"{model_type} hidden size")
            emb_dim = hidden_size

        hf_model_dir = os.path.join(pretrain_dir, "hf_model")
        hf_tok_dir = os.path.join(pretrain_dir, "hf_tokenizer")
        if not os.path.isdir(hf_model_dir):
            raise FileNotFoundError(
                f"No fine-tuned HF model found at {hf_model_dir}. "
                f"Run feat_training first."
            )

        device = "cuda" if torch.cuda.is_available() and not getattr(cfg, '_use_cpu', False) else "cpu"
        hf_tokenizer = AutoTokenizer.from_pretrained(hf_tok_dir)
        model = AutoModelForCausalLM.from_pretrained(
            hf_model_dir, torch_dtype=torch.float32,
        ).to(device)

        if hf_tokenizer.pad_token is None:
            hf_tokenizer.pad_token = hf_tokenizer.eos_token

        model.eval()
        indexid2msg = get_indexid2msg(cfg)
        hf_max_seq_len = pretrain_cfg[model_type].max_seq_len
        indexid2vec = embed_nodes_hf(
            model, hf_tokenizer, indexid2msg, emb_dim, device,
            batch_size=64, max_seq_len=hf_max_seq_len,
        )
        log(f"Computed {len(indexid2vec):,} {model_type} node embeddings (dim={emb_dim})")
        return indexid2vec

    # Node labels are very short (~5 tokens), so use a large batch size
    batch_size = min(pretrain_cfg.batch_size * 16, 8192)

    # ── Load tokenizer ────────────────────────────────────────────────
    from pidsmaker.spider.data.tokenizer_bpe import ProvenanceTokenizerBPE as ProvenanceTokenizer

    tokenizer = ProvenanceTokenizer(cfg)
    tokenizer.load(os.path.join(pretrain_dir, "tokenizer.pt"))

    # ── Load pretrained BERT backbone ─────────────────────────────────
    if model_type == "ropebert":
        from pidsmaker.spider.models.ropebert import ProvenanceRoPEBERT, get_ropebert_config
        rope_theta = getattr(pretrain_cfg, 'rope_theta', 10000.0)
        bert_config = get_ropebert_config(
            tokenizer.vocab_size, model_size, tokenizer.max_seq_len,
            rope_theta=rope_theta,
        )
        bert = ProvenanceRoPEBERT(bert_config)
    elif model_type == "t5":
        from pidsmaker.spider.models.t5 import ProvenanceT5, get_t5_config
        bert_config = get_t5_config(tokenizer.vocab_size, model_size, tokenizer.max_seq_len)
        bert = ProvenanceT5(bert_config)
    elif model_type == "gnn_distill":
        from pidsmaker.spider.models.gnn_distill import ProvenanceGNNDistill, get_gnn_distill_encoder_config
        bert_config = get_gnn_distill_encoder_config(tokenizer.vocab_size, model_size, tokenizer.max_seq_len)
        gnn_hidden_dim = pretrain_cfg.gnn_distill.hidden_dim
        bert = ProvenanceGNNDistill(bert_config, gnn_hidden_dim)
    elif model_type == "spider":
        from pidsmaker.spider.models.spider import ProvenanceGNNCluster, get_spider_encoder_config
        bert_config = get_spider_encoder_config(tokenizer.vocab_size, model_size, tokenizer.max_seq_len)
        gnn_hidden_dim = pretrain_cfg.spider.hidden_dim
        bert = ProvenanceGNNCluster(bert_config, gnn_hidden_dim)
    elif model_type == "behavior_cluster":
        from pidsmaker.spider.models.behavior import ProvenanceBehaviorModel, get_behavior_encoder_config
        from pidsmaker.spider.data.behavior_signatures import BehaviorLabelVocab
        bert_config = get_behavior_encoder_config(tokenizer.vocab_size, model_size, tokenizer.max_seq_len)
        # Load behavior vocab to get num_labels
        bvocab_path = os.path.join(pretrain_dir, "behavior_vocab.txt")
        bvocab = BehaviorLabelVocab()
        if os.path.exists(bvocab_path):
            bvocab.load(bvocab_path)
        bc_cfg = pretrain_cfg.behavior_cluster
        bc_proj_dim = bc_cfg.proj_dim
        bc_bce_mode = bc_cfg.bce_mode
        # For bce_mode="class", we need num_classes; infer from checkpoint keys
        bc_num_classes = 0
        if bc_bce_mode == "class":
            sd_peek = torch.load(
                os.path.join(pretrain_dir, f"pretrain_{model_size}.pt"),
                weights_only=True, map_location="cpu",
            )
            for k, v in sd_peek.items():
                if k == "classification_head.classifier.3.weight":
                    bc_num_classes = v.shape[0]
                    break
            del sd_peek
        bert = ProvenanceBehaviorModel(
            bert_config, num_labels=max(bvocab.size, 1), proj_dim=bc_proj_dim,
            bce_mode=bc_bce_mode, num_classes=bc_num_classes,
        )
    elif model_type == "modernbert":
        from pidsmaker.spider.models.modernbert import ProvenanceModernBERT, get_modernbert_config
        bert_config = get_modernbert_config(
            tokenizer.vocab_size, model_size, tokenizer.max_seq_len,
            global_attn_every_n_layers=getattr(pretrain_cfg, 'global_attn_every_n_layers', 3),
            local_attention_window=getattr(pretrain_cfg, 'local_attention_window', 128),
        )
        bert = ProvenanceModernBERT(bert_config)
    elif model_type == "llama":
        from pidsmaker.spider.models.llama import ProvenanceLLaMA, get_llama_config
        rope_theta = getattr(pretrain_cfg, 'llama', None)
        rope_theta = rope_theta.rope_theta if rope_theta else 10000.0
        bert_config = get_llama_config(
            tokenizer.vocab_size, model_size, tokenizer.max_seq_len,
            rope_theta=rope_theta,
        )
        bert = ProvenanceLLaMA(bert_config)
    elif model_type == "roberta":
        from pidsmaker.spider.models.roberta import ProvenanceRoBERTa, get_roberta_config
        bert_config = get_roberta_config(tokenizer.vocab_size, model_size, tokenizer.max_seq_len)
        bert = ProvenanceRoBERTa(bert_config)
    else:
        from pidsmaker.spider.models.bert import ProvenanceBERT, get_bert_config
        bert_config = get_bert_config(tokenizer.vocab_size, model_size, tokenizer.max_seq_len)
        bert = ProvenanceBERT(bert_config)

    # Validate emb_dim matches model hidden size
    hidden_size = bert_config.d_model if model_type in ("t5", "gnn_distill", "spider", "behavior_cluster") else bert_config.hidden_size
    if emb_dim != hidden_size:
        raise ValueError(
            f"featurization.feat_training.emb_dim ({emb_dim}) does not match "
            f"SPIDER encoder hidden size ({hidden_size}) for model_size='{model_size}'. "
            f"Set emb_dim: {hidden_size} in your config."
        )

    # Load pretrained weights (prefer best checkpoint)
    pt_path = os.path.join(pretrain_dir, f"pretrain_{model_size}_best.pt")
    if not os.path.exists(pt_path):
        pt_path = os.path.join(pretrain_dir, f"pretrain_{model_size}.pt")
    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"No pretrained SPIDER checkpoint found in {pretrain_dir}")

    device = "cuda" if torch.cuda.is_available() and not getattr(cfg, '_use_cpu', False) else "cpu"

    sd = torch.load(pt_path, weights_only=True, map_location="cpu")
    bert.load_state_dict(sd)
    bert = bert.to(device)
    bert.eval()
    log(f"Loaded {model_type} pretrained BERT ({model_size}, H={hidden_size}) from {pt_path}")

    # ── Continue pretraining on target dataset (optional) ─────────────
    inference_cfg = cfg.featurization.feat_inference
    if inference_cfg.continue_pretrain:
        if model_type == "gnn_distill":
            bert = _continue_pretrain_gnn_distill(cfg, bert, tokenizer, pretrain_cfg, inference_cfg, device)
        elif model_type == "spider":
            bert = _continue_pretrain_spider(cfg, bert, tokenizer, pretrain_cfg, inference_cfg, device)
        elif model_type == "t5":
            bert = _continue_pretrain_t5(cfg, bert, tokenizer, pretrain_cfg, inference_cfg, device)
        elif model_type == "behavior_cluster":
            bert = _continue_pretrain_behavior_cluster(cfg, bert, tokenizer, pretrain_cfg, inference_cfg, device)
        else:
            log(f"  continue_pretrain not supported for model_type={model_type}, skipping")

    # ── Determine embedding mode for behavior_cluster ──────────────────
    use_contrastive_head = (
        model_type == "behavior_cluster"
        and getattr(pretrain_cfg.behavior_cluster, "use_contrastive_head", False)
    )
    if use_contrastive_head:
        output_dim = pretrain_cfg.behavior_cluster.proj_dim
        log(f"  behavior_cluster: using contrastive head for inference (dim={output_dim})")
    else:
        output_dim = hidden_size

    # ── Compute embeddings for all nodes ──────────────────────────────
    indexid2msg = get_indexid2msg(cfg)
    sorted_items = sorted(indexid2msg.items(), key=lambda x: int(x[0]))

    # Deduplicate: group node IDs by their unique (ntype, nlabel) text key
    text_key_to_node_ids = {}
    for node_id, (ntype, nlabel) in sorted_items:
        key = (ntype, nlabel)
        text_key_to_node_ids.setdefault(key, []).append(node_id)

    unique_items = list(text_key_to_node_ids.keys())
    n_nodes = len(sorted_items)
    n_unique = len(unique_items)

    indexid2vec = {}
    total_batches = (n_unique + batch_size - 1) // batch_size

    for batch_idx in range(0, n_unique, batch_size):
        batch_keys = unique_items[batch_idx:batch_idx + batch_size]
        batch_num = batch_idx // batch_size

        if batch_num % 20 == 0:
            log(f"  Embedding batch {batch_num+1}/{total_batches} "
                f"({batch_idx:,}/{n_unique:,} unique labels)")

        # Tokenize all unique labels in batch
        batch_token_ids = []
        batch_keys_valid = []
        for key in batch_keys:
            ntype, nlabel = key
            tids = tokenizer.tokenize_node(ntype, nlabel)
            if tids:
                batch_token_ids.append(tids)
                batch_keys_valid.append(key)
            else:
                zero_emb = np.zeros(output_dim, dtype=np.float32)
                for node_id in text_key_to_node_ids[key]:
                    indexid2vec[int(node_id)] = zero_emb

        if not batch_token_ids:
            continue

        # Pad and encode
        max_len = min(max(len(t) for t in batch_token_ids), tokenizer.max_seq_len)
        B = len(batch_token_ids)
        input_ids = torch.full((B, max_len), tokenizer.pad_id, dtype=torch.long)
        attention_mask = torch.zeros(B, max_len, dtype=torch.bool)
        for i, tids in enumerate(batch_token_ids):
            L = min(len(tids), max_len)
            input_ids[i, :L] = torch.tensor(tids[:L])
            attention_mask[i, :L] = True

        with torch.no_grad():
            if use_contrastive_head:
                # Use full forward: encoder → mean pool → contrastive projection
                out = bert.modified_fwd(
                    input_ids.to(device),
                    attention_mask.to(device),
                )
                pooled_np = out.projection.cpu().numpy()  # [B, proj_dim], already L2-normalized
            else:
                # Use raw encoder hidden states → mean pool
                hidden = bert.modified_fwd(
                    input_ids.to(device),
                    attention_mask.to(device),
                    labels=torch.full_like(input_ids, -100).to(device),
                    skip_cls=True,
                )  # [B, max_len, H]
                mask_f = attention_mask.to(device).unsqueeze(-1).float()
                pooled = (hidden * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)  # [B, H]
                pooled_np = pooled.cpu().numpy()

        # Normalize and assign to all nodes sharing the same text label
        for i, key in enumerate(batch_keys_valid):
            emb = pooled_np[i]
            norm = np.linalg.norm(emb)
            if norm > 1e-12:
                emb = emb / norm
            for node_id in text_key_to_node_ids[key]:
                indexid2vec[int(node_id)] = emb

    _maybe_embed_synthetic_nodes(cfg, bert, tokenizer, hidden_size, device)

    del bert
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    log(f"Computed {len(indexid2vec):,} static BERT node embeddings (dim={output_dim})")
    return indexid2vec
