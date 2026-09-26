"""GNN Distillation pretraining branch for SPIDER."""

import os
import random
import time
from collections import defaultdict

import torch
from torch.optim import AdamW

import wandb

from pidsmaker.utils.utils import log
from ..training_utils import WarmupCosineScheduler, pad_token_id_lists, update_ema
from ..data.tokenizer_bpe import ProvenanceTokenizerBPE as ProvenanceTokenizer
from .pretrain_common import _eval_knn_on_eval_set, _dump_gnn_distill_corpus_to_txt


def pretrain_gnn_distill(cfg, spider_cfg, sampler_pairs, combined_indexid2msg, out_dir, device,
                         model_size, max_seq_len, bpe_vocab_size, tokenizer_mode,
                         normalize_netflow_ips, canonicalize_neighbors, spider_path,
                         total_tokens, warmup_tokens, batch_size, lr,
                         use_synthetic_data, synthetic_dir):
    from copy import deepcopy
    from ..models.gnn_distill import (
        ProvenanceGNNDistill, NeighborhoodGNNTeacher,
        build_gnn_distill_batch, build_edge_type_encoder,
        get_gnn_distill_encoder_config,
    )
    from ..models.graphmae import sce_loss
    from ..data.synthetic import load_synthetic_gnn_corpus

    distill_cfg = spider_cfg.gnn_distill
    gnn_edge_emb_dim = distill_cfg.emb_dim
    gnn_hidden_dim = distill_cfg.hidden_dim
    gnn_num_layers = distill_cfg.num_layers
    gnn_num_heads = distill_cfg.num_heads
    gnn_n_neighbors_min = distill_cfg.n_neighbors_min
    gnn_n_neighbors_max = distill_cfg.n_neighbors_max
    gnn_diverse_neighbors = distill_cfg.diverse_neighbors
    distill_lambda = distill_cfg.loss_weight
    gnn_loss_lambda = distill_cfg.gnn_loss_weight
    gnn_lr = distill_cfg.gnn_lr
    ema_momentum = distill_cfg.ema_momentum

    # ── Load synthetic data for GNN distillation (if requested) ──
    if use_synthetic_data:
        synth_sampler, synth_indexid2msg = load_synthetic_gnn_corpus(synthetic_dir)
        if synth_sampler is not None:
            sampler_pairs.append((synth_sampler, synth_indexid2msg))
            for k, v in synth_indexid2msg.items():
                combined_indexid2msg[k] = v
            log(f"Added synthetic sampler: {len(synth_sampler.nodes):,} entities")

    # ── Build or load tokenizer ────────────────────────────────────
    if spider_path:
        tok_path = os.path.join(spider_path, "tokenizer.pt")
        log(f"Loading tokenizer from {tok_path}")
        tokenizer = ProvenanceTokenizer(cfg)
        tokenizer.load(tok_path)
        tokenizer.max_seq_len = max_seq_len
        log(f"  Overriding max_seq_len to {max_seq_len} (from config)")
    else:
        walk_label_corpus = []
        for sampler, ds_indexid2msg in sampler_pairs:
            for node in sampler.nodes:
                if node in ds_indexid2msg:
                    walk_label_corpus.append(ds_indexid2msg[node])
        log(f"GNN distill: {len(walk_label_corpus):,} entity labels for tokenizer")

        tokenizer = ProvenanceTokenizer(cfg, max_seq_len=max_seq_len, bpe_vocab_size=bpe_vocab_size, mode=tokenizer_mode,
                                        normalize_netflow_ips=normalize_netflow_ips,
                                        canonicalize_neighbors=canonicalize_neighbors)
        tokenizer.build_vocab(combined_indexid2msg, walk_label_corpus=walk_label_corpus)
    log(f"Vocabulary size: {tokenizer.vocab_size}")
    tokenizer.save(os.path.join(out_dir, "tokenizer.pt"))

    # ── Build edge type encoder for GNN ───────────────────────────
    gnn_edge_type2idx = build_edge_type_encoder(sampler_pairs)
    log(f"GNN distill: {len(gnn_edge_type2idx)} directed edge types")

    # ── Build models ──────────────────────────────────────────────
    enc_config = get_gnn_distill_encoder_config(tokenizer.vocab_size, model_size, max_seq_len)
    t5_hidden_dim = enc_config.d_model  # H dimension of T5 encoder
    model = ProvenanceGNNDistill(enc_config, gnn_hidden_dim).to(device)
    log(f"Model: GNNDistill-{model_size} (T5 encoder H={t5_hidden_dim} → {gnn_hidden_dim}d GNN space)")
    log(f"  Student params: {sum(p.numel() for p in model.parameters()):,}")

    # EMA teacher: a slow-moving copy of the student T5 encoder
    ema_teacher = deepcopy(model).to(device)
    ema_teacher.requires_grad_(False)
    ema_teacher.eval()
    log(f"  EMA teacher: copy of student (momentum={ema_momentum})")

    gnn_filter_noisy = distill_cfg.filter_noisy_edges
    if gnn_filter_noisy:
        log("GNN distill: edge noise filtering ENABLED")
    gnn_edge_projection = distill_cfg.edge_projection
    gnn_teacher = NeighborhoodGNNTeacher(
        node_dim=t5_hidden_dim,
        num_edge_types=max(len(gnn_edge_type2idx), 1),
        edge_emb_dim=gnn_edge_emb_dim,
        hidden_dim=gnn_hidden_dim,
        num_layers=gnn_num_layers,
        num_heads=gnn_num_heads,
        edge_projection=gnn_edge_projection,
    ).to(device)
    log(f"  GNN teacher params: {sum(p.numel() for p in gnn_teacher.parameters()):,}")
    log(f"  n_neighbors=[{gnn_n_neighbors_min}, {gnn_n_neighbors_max}], "
        f"distill_weight={distill_lambda}, gnn_weight={gnn_loss_lambda}")

    # ── Optimizer ─────────────────────────────────────────────────
    opt = AdamW([
        {"params": model.parameters(), "lr": lr},
        {"params": gnn_teacher.parameters(), "lr": gnn_lr},
    ], betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)

    # ── Collect all trainable entities, grouped by label ──────────
    if gnn_filter_noisy:
        from ..data.edge_filter import _is_noisy_file, _is_noisy_process
    n_skipped_centers = 0
    label_to_entities = defaultdict(list)
    for sidx, (sampler, ds_indexid2msg) in enumerate(sampler_pairs):
        for node in sampler.nodes:
            if node in ds_indexid2msg:
                ntype, nlabel = ds_indexid2msg[node]
                # Skip noisy center entities
                if gnn_filter_noisy:
                    if ntype == "file" and _is_noisy_file(nlabel):
                        n_skipped_centers += 1
                        continue
                    if ntype == "subject" and _is_noisy_process(nlabel):
                        n_skipped_centers += 1
                        continue
                entity_ids = tokenizer.tokenize_node(ntype, nlabel)
                if entity_ids:
                    label_key = (ntype, nlabel)
                    label_to_entities[label_key].append((node, sidx, entity_ids))
    n_total = sum(len(v) for v in label_to_entities.values())
    skip_msg = f", skipped {n_skipped_centers:,} noisy centers" if n_skipped_centers else ""
    log(f"GNN distill: {n_total:,} trainable entities, "
        f"{len(label_to_entities):,} unique labels across {len(sampler_pairs)} graph(s){skip_msg}")

    # ── Dump neighborhood corpus for analysis ─────────────────
    _dump_gnn_distill_corpus_to_txt(
        sampler_pairs, combined_indexid2msg, out_dir,
        filter_noisy=gnn_filter_noisy,
    )

    # ── Split into train/val (2% of unique labels) ───────────
    total_labels = len(label_to_entities)
    n_val_target = max(1, int(total_labels * 0.02))
    # Distribute proportionally across datasets
    val_labels = set()
    all_labels_shuffled = list(label_to_entities.keys())
    random.shuffle(all_labels_shuffled)
    for lk in all_labels_shuffled[:n_val_target]:
        val_labels.add(lk)

    train_label_to_entities = defaultdict(list)
    val_label_to_entities = defaultdict(list)
    for label_key, entities in label_to_entities.items():
        if label_key in val_labels:
            val_label_to_entities[label_key] = entities
        else:
            train_label_to_entities[label_key] = entities

    n_train_labels = len(train_label_to_entities)
    n_val_labels = len(val_label_to_entities)
    log(f"GNN distill: {n_train_labels:,} train / {n_val_labels:,} val unique labels "
        f"(2% held out for validation)")

    scheduler = WarmupCosineScheduler(opt, warmup_tokens, total_tokens)

    # ── Precompute fixed val neighborhoods (for consistent eval) ──
    val_batches_fixed = []
    if val_label_to_entities:
        val_entities = []
        for label_key, entities in val_label_to_entities.items():
            val_entities.append(entities[0])  # always first node

        for vb_start in range(0, len(val_entities), batch_size):
            vb_ents = val_entities[vb_start:vb_start + batch_size]
            vB = len(vb_ents)
            if vB == 0:
                continue

            # Pad encoder inputs
            v_enc_ids = [ent[2] for ent in vb_ents]
            v_input_ids, v_attn_mask = pad_token_id_lists(v_enc_ids, tokenizer.pad_id, tokenizer.max_seq_len)

            # Build GNN neighborhood batch (fixed neighborhoods)
            v_entity_nodes = [ent[0] for ent in vb_ents]
            v_sampler_idxs = [ent[1] for ent in vb_ents]

            v_gnn_batch = build_gnn_distill_batch(
                v_entity_nodes, v_sampler_idxs, sampler_pairs,
                combined_indexid2msg, gnn_edge_type2idx,
                n_neighbors_min=gnn_n_neighbors_min, n_neighbors_max=gnn_n_neighbors_max,
                diverse=gnn_diverse_neighbors, device=device,
                filter_noisy=gnn_filter_noisy,
            )
            if v_gnn_batch is None:
                continue

            v_node_labels, v_edge_index, v_edge_types, v_batch_vec, v_center_idx, v_valid_mask = v_gnn_batch
            if v_valid_mask.sum().item() == 0:
                continue

            val_batches_fixed.append((
                v_input_ids, v_attn_mask, v_node_labels,
                v_edge_index, v_edge_types, v_batch_vec,
                v_center_idx, v_valid_mask,
            ))
        log(f"Precomputed {len(val_batches_fixed)} fixed val batches")

    # ── Training loop ─────────────────────────────────────────────
    processed_tokens = 0
    updates = 0
    epoch = 0
    best_knn_acc = -1.0

    log_path = os.path.join(out_dir, f"gnn_distill_log_{model_size}.txt")
    with open(log_path, "w") as f:
        f.write("update,tokens,distill_loss,gnn_loss,lr,time\n")

    # EMA embedding cache: refreshed every N steps since EMA changes slowly
    ema_cache = {}  # label → embedding tensor (detached, on device)
    ema_cache_refresh_interval = 50  # refresh every 50 updates
    ema_cache_last_refresh = -ema_cache_refresh_interval  # force first refresh

    while processed_tokens < total_tokens:
        epoch += 1
        epoch_st = time.time()
        epoch_d_loss = 0.0
        epoch_g_loss = 0.0
        epoch_batches = 0

        # Pick one node per label, rotating across epochs
        epoch_entities = []
        for label_key, entities in train_label_to_entities.items():
            pick = entities[(epoch - 1) % len(entities)]
            epoch_entities.append(pick)
        random.shuffle(epoch_entities)

        for batch_start in range(0, len(epoch_entities), batch_size):
            batch_ents = epoch_entities[batch_start:batch_start + batch_size]
            B = len(batch_ents)
            if B == 0:
                continue

            # ── Pad encoder inputs for student (center entity tokens) ──
            enc_ids_list = [ent[2] for ent in batch_ents]
            input_ids, attention_mask = pad_token_id_lists(enc_ids_list, tokenizer.pad_id, tokenizer.max_seq_len)

            # ── Build GNN neighborhood batch ──────────────────────
            entity_nodes = [ent[0] for ent in batch_ents]
            sampler_idxs = [ent[1] for ent in batch_ents]

            gnn_batch = build_gnn_distill_batch(
                entity_nodes, sampler_idxs, sampler_pairs,
                combined_indexid2msg, gnn_edge_type2idx,
                n_neighbors_min=gnn_n_neighbors_min, n_neighbors_max=gnn_n_neighbors_max,
                diverse=gnn_diverse_neighbors, device=device,
                filter_noisy=gnn_filter_noisy,
            )
            if gnn_batch is None:
                continue

            node_labels, edge_index, edge_types, batch_vec, center_idx, valid_mask = gnn_batch
            n_valid = valid_mask.sum().item()
            if n_valid == 0:
                continue

            # ── Encode neighbor nodes with EMA teacher (cached) ────
            # Refresh cache periodically since EMA changes slowly
            need_refresh = (updates - ema_cache_last_refresh) >= ema_cache_refresh_interval
            uncached_labels = [lbl for lbl in set(node_labels) if lbl not in ema_cache]

            if need_refresh:
                # Full refresh: re-encode all labels seen so far
                all_labels_to_encode = list(set(node_labels))
                ema_cache_last_refresh = updates
            elif uncached_labels:
                # Only encode new labels not yet in cache
                all_labels_to_encode = uncached_labels
            else:
                all_labels_to_encode = []

            if all_labels_to_encode:
                ema_token_ids = []
                for ntype, nlabel in all_labels_to_encode:
                    tids = tokenizer.tokenize_node(ntype, nlabel)
                    ema_token_ids.append(tids if tids else [tokenizer.pad_id])

                ema_input_ids, ema_attn_mask = pad_token_id_lists(ema_token_ids, tokenizer.pad_id, tokenizer.max_seq_len)

                ema_teacher.eval()
                with torch.no_grad():
                    ema_embeddings = ema_teacher.mean_pool(ema_input_ids, ema_attn_mask)

                for i, lbl in enumerate(all_labels_to_encode):
                    ema_cache[lbl] = ema_embeddings[i]

            # Assemble node embeddings from cache
            node_embeddings = torch.stack([ema_cache[lbl] for lbl in node_labels])  # [N, d_model]

            # ── Forward passes ────────────────────────────────────
            model.train()
            gnn_teacher.train()

            # Student: T5 encoder → mean-pool → project to GNN space
            out = model.modified_fwd(input_ids.to(device), attention_mask.to(device))
            t5_projected = out.projected  # [B, gnn_hidden_dim]

            # GNN teacher: mask center nodes, encode, reconstruct EMA embeddings
            gnn_recon_loss, gnn_encoded = gnn_teacher(
                node_embeddings, edge_index, edge_types, batch_vec, center_idx,
            )

            # Extract center node embeddings with STOP GRADIENT
            gnn_targets = gnn_encoded[center_idx].detach()

            # Distillation loss (SCE) on valid entities only
            distill_loss = sce_loss(t5_projected[valid_mask], gnn_targets)

            # Combined loss
            total_loss = distill_lambda * distill_loss + gnn_loss_lambda * gnn_recon_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            torch.nn.utils.clip_grad_norm_(gnn_teacher.parameters(), 5.0)
            opt.step()
            opt.zero_grad()

            # ── EMA update ────────────────────────────────────────
            update_ema(ema_teacher, model, ema_momentum)

            # ── Tracking ──────────────────────────────────────────
            n_tokens = int(attention_mask.sum().item())
            if n_tokens == 0:
                continue

            processed_tokens += n_tokens
            scheduler.last_epoch = processed_tokens
            scheduler.step()
            updates += 1

            d_loss_val = distill_loss.item()
            g_loss_val = gnn_recon_loss.item()
            epoch_d_loss += d_loss_val
            epoch_g_loss += g_loss_val
            epoch_batches += 1
            current_lr = scheduler.get_last_lr()[0]

            if updates % 100 == 0:
                elapsed = time.time() - epoch_st
                log(
                    f"[{updates}|e{epoch}] d_loss={d_loss_val:.4f} g_loss={g_loss_val:.4f} "
                    f"valid={n_valid}/{B} lr={current_lr:.2e} "
                    f"tokens={processed_tokens:.2e} ({elapsed:.1f}s)"
                )
                with open(log_path, "a") as f:
                    f.write(f"{updates},{processed_tokens},{d_loss_val:.6f},"
                            f"{g_loss_val:.6f},{current_lr:.2e},{elapsed:.1f}\n")

            if processed_tokens >= total_tokens:
                break

        if processed_tokens >= total_tokens:
            break

        # Epoch-level logging
        if epoch_batches > 0:
            avg_d_loss = epoch_d_loss / epoch_batches
            avg_g_loss = epoch_g_loss / epoch_batches
            elapsed = time.time() - epoch_st
            log(f"Epoch {epoch} done | avg_d_loss={avg_d_loss:.4f} avg_g_loss={avg_g_loss:.4f} "
                f"batches={epoch_batches} ({elapsed:.1f}s)")

        # ── Validation on precomputed fixed val batches ──────────
        if val_batches_fixed:
            model.eval()
            gnn_teacher.eval()
            val_d_losses = []

            for (v_input_ids, v_attn_mask, v_node_labels,
                 v_edge_index, v_edge_types, v_batch_vec,
                 v_center_idx, v_valid_mask) in val_batches_fixed:

                # Encode neighbor nodes with EMA teacher (embeddings evolve)
                v_unique_labels = list(set(v_node_labels))
                v_label_to_uidx = {lbl: i for i, lbl in enumerate(v_unique_labels)}
                v_ema_tids = []
                for ntype, nlabel in v_unique_labels:
                    tids = tokenizer.tokenize_node(ntype, nlabel)
                    v_ema_tids.append(tids if tids else [tokenizer.pad_id])
                v_ema_ids, v_ema_mask = pad_token_id_lists(v_ema_tids, tokenizer.pad_id, tokenizer.max_seq_len)

                with torch.no_grad():
                    v_ema_emb = ema_teacher.mean_pool(v_ema_ids, v_ema_mask)
                v_node_emb_idx = [v_label_to_uidx[lbl] for lbl in v_node_labels]
                v_node_emb = v_ema_emb[torch.tensor(v_node_emb_idx, dtype=torch.long)]

                with torch.no_grad():
                    v_out = model.modified_fwd(v_input_ids.to(device), v_attn_mask.to(device))
                    v_t5_proj = v_out.projected
                    _, v_gnn_enc = gnn_teacher(
                        v_node_emb, v_edge_index, v_edge_types, v_batch_vec, v_center_idx,
                    )
                    v_gnn_tgt = v_gnn_enc[v_center_idx]
                    v_dloss = sce_loss(v_t5_proj[v_valid_mask], v_gnn_tgt)
                    val_d_losses.append(v_dloss.item())

        epoch_metrics = {"pretrain/epoch": epoch, "pretrain/tokens": processed_tokens}

        if val_batches_fixed and val_d_losses:
            val_distill_loss = sum(val_d_losses) / len(val_d_losses)
            log(f"Epoch {epoch} val_distill_loss={val_distill_loss:.4f}")
            epoch_metrics["pretrain/val_distill_loss"] = val_distill_loss

        # ── KNN eval on eval set ──────────────────────────────────
        knn_results = _eval_knn_on_eval_set(model, tokenizer, device, "gnn_distill", batch_size)
        if knn_results:
            knn_acc = knn_results['knn_accuracy_k5']
            epoch_metrics["pretrain/knn_accuracy_k5"] = knn_acc
            for cl, acc in knn_results['knn_accuracy_k5_per_cluster'].items():
                epoch_metrics[f"pretrain/knn_k5/{cl}"] = acc
            epoch_metrics["pretrain/ari"] = knn_results['ari']
            epoch_metrics["pretrain/nmi"] = knn_results['nmi']
            epoch_metrics["pretrain/silhouette"] = knn_results['silhouette']
            epoch_metrics["pretrain/mAP"] = knn_results['mAP']
            log(f"Epoch {epoch} knn_accuracy_k5={knn_acc:.4f} "
                f"ARI={knn_results['ari']:.4f} NMI={knn_results['nmi']:.4f} "
                f"silhouette={knn_results['silhouette']:.4f} mAP={knn_results['mAP']:.4f}")

            if knn_acc > best_knn_acc:
                best_knn_acc = knn_acc
                log(f"  New best KNN accuracy: {knn_acc:.4f}")
            epoch_metrics["pretrain/best_knn_accuracy_k5"] = best_knn_acc

        # Always save latest model (last epoch wins)
        torch.save(model.state_dict(), os.path.join(out_dir, f"pretrain_{model_size}.pt"))

        wandb.log(epoch_metrics)

    # Final save (tokenizer)
    tokenizer.save(os.path.join(out_dir, "tokenizer.pt"))
    log(f"GNN distill pretraining complete. {processed_tokens:,} tokens in {epoch} epochs.")
