"""Behavior Cluster pretraining branch for SPIDER."""

import os
import random
import time
from collections import defaultdict

import torch
from torch.optim import AdamW

import wandb

from pidsmaker.utils.utils import log
from ..training_utils import WarmupCosineScheduler, pad_token_id_lists, build_pk_batches
from ..data.tokenizer_bpe import ProvenanceTokenizerBPE as ProvenanceTokenizer
from .pretrain_common import _eval_knn_on_eval_set


def pretrain_behavior_cluster(cfg, spider_cfg, sampler_pairs, combined_indexid2msg, out_dir, device,
                              model_size, max_seq_len, bpe_vocab_size, tokenizer_mode,
                              normalize_netflow_ips, canonicalize_neighbors, spider_path,
                              total_tokens, warmup_tokens, batch_size, lr):
    from ..models.behavior import (
        ProvenanceBehaviorModel,
        get_behavior_encoder_config,
        behavior_combined_loss,
    )
    from ..data.behavior_signatures import (
        build_behavior_dataset,
        BehaviorLabelVocab,
        dump_entity_classifications,
        dump_behavior_signatures,
    )

    bc_cfg = spider_cfg.behavior_cluster
    bc_proj_dim = bc_cfg.proj_dim
    bc_bce_weight = bc_cfg.bce_weight
    bc_contrastive_weight = bc_cfg.contrastive_weight
    bc_temperature = bc_cfg.temperature

    bc_min_sig_size = bc_cfg.min_signature_size
    bc_filter_noisy = bc_cfg.filter_noisy_edges
    bc_max_per_class = bc_cfg.max_entities_per_class
    bc_min_per_class = bc_cfg.min_entities_per_class
    bc_samples_per_class = bc_cfg.samples_per_class
    bc_strip_entity_type = bc_cfg.strip_entity_type
    bc_bce_mode = bc_cfg.bce_mode
    bc_contrastive_target = bc_cfg.contrastive_target

    # ── Build or load tokenizer ────────────────────────────────────
    if spider_path:
        tok_path = os.path.join(spider_path, "tokenizer.pt")
        log(f"Loading tokenizer from {tok_path}")
        tokenizer = ProvenanceTokenizer(cfg, max_seq_len=max_seq_len, bpe_vocab_size=bpe_vocab_size,
                                        mode=tokenizer_mode, normalize_netflow_ips=normalize_netflow_ips,
                                        canonicalize_neighbors=canonicalize_neighbors)
        tokenizer.load(tok_path)
        tokenizer.max_seq_len = max_seq_len
    else:
        walk_label_corpus = []
        for sampler, ds_indexid2msg in sampler_pairs:
            for node in sampler.nodes:
                if node in ds_indexid2msg:
                    walk_label_corpus.append(ds_indexid2msg[node])
        log(f"Behavior cluster: {len(walk_label_corpus):,} entity labels for tokenizer")

        tokenizer = ProvenanceTokenizer(
            cfg, max_seq_len=max_seq_len, bpe_vocab_size=bpe_vocab_size,
            mode=tokenizer_mode, normalize_netflow_ips=normalize_netflow_ips,
            canonicalize_neighbors=canonicalize_neighbors,
            strip_entity_type=bc_strip_entity_type,
        )
        tokenizer.build_vocab(combined_indexid2msg, walk_label_corpus=walk_label_corpus)
    log(f"Vocabulary size: {tokenizer.vocab_size}")
    tokenizer.save(os.path.join(out_dir, "tokenizer.pt"))

    # ── Build behavior signature dataset (always recomputed — depends on entity_classes.py) ──
    log("Building behavior signature dataset (extracting full 1-hop signatures)...")
    label_to_signature, label_to_entities_raw, vocab = build_behavior_dataset(
        sampler_pairs, combined_indexid2msg,
        filter_noisy=bc_filter_noisy,
        min_signature_size=bc_min_sig_size,
    )
    log(f"Behavior signatures: {len(label_to_signature):,} unique entity labels, "
        f"{vocab.size:,} behavior labels in vocabulary")

    # Save for analysis
    vocab.save(os.path.join(out_dir, "behavior_vocab.txt"))

    # Dump reports
    classifications_path = os.path.join(out_dir, "entity_classifications.txt")
    dump_entity_classifications(
        sampler_pairs, filter_noisy=bc_filter_noisy, output_path=classifications_path,
    )
    log(f"Entity classification report written to {classifications_path}")

    signatures_path = os.path.join(out_dir, "behavior_signatures.txt")
    dump_behavior_signatures(label_to_signature, output_path=signatures_path)
    log(f"Behavior signatures report written to {signatures_path}")

    # ── Tokenize all entities and attach token IDs ─────────────────
    if bc_filter_noisy:
        from ..data.edge_filter import _is_noisy_file, _is_noisy_process

    label_to_entities = defaultdict(list)
    for label_key, raw_entities in label_to_entities_raw.items():
        ntype, nlabel = label_key
        entity_ids = tokenizer.tokenize_node(ntype, nlabel)
        if entity_ids:
            for node_id, sidx in raw_entities:
                label_to_entities[label_key].append((node_id, sidx, entity_ids))

    n_before_dedup = len(label_to_entities)

    # Track per-class counts before dedup
    from ..data.entity_classes import classify_entity
    pre_dedup_class_counts = defaultdict(int)
    for label_key in label_to_entities:
        ntype, nlabel = label_key
        pre_dedup_class_counts[classify_entity(ntype, nlabel)] += 1

    # ── Post-tokenization dedup: merge labels with identical token sequences ──
    # e.g. /proc/123/stat and /proc/456/stat both tokenize to [PROC_FS] [NUMBER] stat
    token_sig_to_label = {}  # tuple(token_ids) → first label_key seen
    merged_count = 0
    for label_key in list(label_to_entities.keys()):
        entity_ids = label_to_entities[label_key][0][2]  # all entities share same token_ids
        token_sig = tuple(entity_ids)
        if token_sig in token_sig_to_label:
            # Merge into existing: union signatures, combine entities
            canonical = token_sig_to_label[token_sig]
            label_to_signature[canonical] = label_to_signature[canonical] | label_to_signature[label_key]
            label_to_entities[canonical].extend(label_to_entities[label_key])
            del label_to_entities[label_key]
            del label_to_signature[label_key]
            merged_count += 1
        else:
            token_sig_to_label[token_sig] = label_key

    n_total = sum(len(v) for v in label_to_entities.values())
    log(f"Behavior cluster: {n_total:,} trainable entities, "
        f"{len(label_to_entities):,} unique labels "
        f"(deduped {merged_count:,} from {n_before_dedup:,})")

    # Write per-class dedup report
    post_dedup_class_counts = defaultdict(int)
    for label_key in label_to_entities:
        ntype, nlabel = label_key
        post_dedup_class_counts[classify_entity(ntype, nlabel)] += 1

    dedup_report_path = os.path.join(out_dir, "tokenization_dedup_report.txt")
    all_classes = sorted(pre_dedup_class_counts.keys(), key=lambda c: -pre_dedup_class_counts[c])
    with open(dedup_report_path, "w") as f:
        f.write(f"{'CLASS':<28} {'PRE-DEDUP':>10} {'POST-DEDUP':>12} {'REDUCTION':>10}\n")
        f.write("-" * 64 + "\n")
        for cls in all_classes:
            pre = pre_dedup_class_counts[cls]
            post = post_dedup_class_counts.get(cls, 0)
            reduction = 100 * (1 - post / pre) if pre > 0 else 0
            f.write(f"  {cls:<26} {pre:>10,} {post:>12,} {reduction:>9.1f}%\n")
        total_pre = sum(pre_dedup_class_counts.values())
        total_post = sum(post_dedup_class_counts.values())
        total_red = 100 * (1 - total_post / total_pre) if total_pre > 0 else 0
        f.write("-" * 64 + "\n")
        f.write(f"  {'TOTAL':<26} {total_pre:>10,} {total_post:>12,} {total_red:>9.1f}%\n")
    log(f"Tokenization dedup report: {dedup_report_path}")

    # ── Pre-compute target vectors and class IDs (done once, reused every epoch) ──
    label_to_target = {}
    for label_key, sig in label_to_signature.items():
        if label_key in label_to_entities:
            label_to_target[label_key] = vocab.signature_to_vector(sig)

    # Assign class IDs: identical signatures → same class
    sig_to_class_id = {}
    label_to_class_id = {}
    for label_key, sig in label_to_signature.items():
        if label_key in label_to_target:
            if sig not in sig_to_class_id:
                sig_to_class_id[sig] = len(sig_to_class_id)
            label_to_class_id[label_key] = sig_to_class_id[sig]

    # Assign entity class IDs for bce_mode="class" (coarse functional classes)
    entity_class_to_id = {}
    label_to_entity_class_id = {}
    for label_key in label_to_target:
        ntype, nlabel = label_key
        ecls = classify_entity(ntype, nlabel)
        if ecls not in entity_class_to_id:
            entity_class_to_id[ecls] = len(entity_class_to_id)
        label_to_entity_class_id[label_key] = entity_class_to_id[ecls]

    log(f"Pre-computed {len(label_to_target):,} target vectors, "
        f"{len(sig_to_class_id):,} unique signature classes, "
        f"{len(entity_class_to_id):,} entity classes")

    # Contrastive grouping: determines P×K batching and contrastive positives
    if bc_contrastive_target == "entity_class":
        label_to_contrastive_id = label_to_entity_class_id
        log(f"Contrastive target: entity_class ({len(entity_class_to_id)} classes)")
    else:
        label_to_contrastive_id = label_to_class_id
        log(f"Contrastive target: signature ({len(sig_to_class_id)} classes)")

    # ── Stratified train/val split ─────────────────────────────────
    # Sample val from classes with enough examples (>=10 labels),
    # taking 5% per class. Small classes go entirely to train.
    class_to_label_keys = defaultdict(list)
    for label_key in label_to_entities:
        class_to_label_keys[label_to_contrastive_id[label_key]].append(label_key)

    val_labels = set()
    for class_id, class_label_keys in class_to_label_keys.items():
        if len(class_label_keys) >= 10:
            random.shuffle(class_label_keys)
            n_val = max(1, int(len(class_label_keys) * 0.05))
            val_labels.update(class_label_keys[:n_val])

    train_label_to_entities = defaultdict(list)
    val_label_to_entities = defaultdict(list)
    for label_key, entities in label_to_entities.items():
        if label_key in val_labels:
            val_label_to_entities[label_key] = entities
        else:
            train_label_to_entities[label_key] = entities

    n_val_classes = len({label_to_contrastive_id[lk] for lk in val_label_to_entities})
    log(f"Behavior cluster: {len(train_label_to_entities):,} train / "
        f"{len(val_label_to_entities):,} val unique labels "
        f"(val covers {n_val_classes} classes)")

    # Log class size distribution
    train_class_sizes = defaultdict(int)
    for lk in train_label_to_entities:
        train_class_sizes[label_to_class_id[lk]] += 1
    sorted_sizes = sorted(train_class_sizes.values(), reverse=True)
    n_classes = len(sorted_sizes)
    effective_total = sum(
        max(bc_min_per_class, min(s, bc_max_per_class)) if bc_max_per_class > 0
        else max(bc_min_per_class, s)
        for s in sorted_sizes
    )
    log(f"  Class sizes: {n_classes} classes, max={sorted_sizes[0]:,}, "
        f"median={sorted_sizes[n_classes//2]:,}, min={sorted_sizes[-1]:,}")
    log(f"  Rebalancing: floor={bc_min_per_class}, cap={bc_max_per_class} "
        f"→ {effective_total:,} entities/epoch (was {sum(sorted_sizes):,})")
    log(f"  P×K batching: K={bc_samples_per_class} samples/class, "
        f"P={batch_size // bc_samples_per_class} classes/batch "
        f"(grouped by {bc_contrastive_target})")

    # ── Build model ────────────────────────────────────────────────
    enc_config = get_behavior_encoder_config(tokenizer.vocab_size, model_size, max_seq_len)
    n_entity_classes = len(entity_class_to_id)
    model = ProvenanceBehaviorModel(
        enc_config, num_labels=vocab.size, proj_dim=bc_proj_dim,
        bce_mode=bc_bce_mode, num_classes=n_entity_classes,
    ).to(device)
    bce_mode_desc = (f"{n_entity_classes} entity classes (CE)"
                     if bc_bce_mode == "class"
                     else f"{vocab.size} behavior labels (BCE)")
    log(f"Model: BehaviorCluster-{model_size} (T5 encoder H={enc_config.d_model}, "
        f"{bce_mode_desc}, proj_dim={bc_proj_dim})"
        + (", entity type stripped" if bc_strip_entity_type else ""))
    log(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── Optimizer & scheduler ──────────────────────────────────────
    opt = AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    scheduler = WarmupCosineScheduler(opt, warmup_tokens, total_tokens)

    # ── Precompute validation batches (P×K grouped) ────────────────
    val_batches_fixed = []
    K = bc_samples_per_class
    P = batch_size // K
    if val_label_to_entities:
        val_items = []
        for label_key, entities in val_label_to_entities.items():
            val_items.append((label_key, entities[0], label_to_contrastive_id[label_key]))

        # Group by class for P×K batching
        val_class_pools = defaultdict(list)
        for item in val_items:
            val_class_pools[item[2]].append(item)

        val_ordered = []
        val_class_ids_list = list(val_class_pools.keys())
        random.shuffle(val_class_ids_list)
        batch_buf = []
        for cid in val_class_ids_list:
            pool = val_class_pools[cid]
            # Pad with duplicates to reach K so every sample has positives
            padded = list(pool)
            while len(padded) < K:
                padded.append(random.choice(pool))
            batch_buf.extend(padded[:K])
            if len(batch_buf) >= batch_size:
                val_ordered.extend(batch_buf[:batch_size])
                batch_buf = batch_buf[batch_size:]
        if batch_buf:
            val_ordered.extend(batch_buf)

        for vb_start in range(0, len(val_ordered), batch_size):
            vb = val_ordered[vb_start:vb_start + batch_size]
            vB = len(vb)
            if vB == 0:
                continue

            v_enc_ids = [item[1][2] for item in vb]
            v_input_ids, v_attn_mask = pad_token_id_lists(v_enc_ids, tokenizer.pad_id, tokenizer.max_seq_len)

            v_targets = torch.stack([label_to_target[item[0]] for item in vb])
            v_class_ids = torch.tensor([item[2] for item in vb], dtype=torch.long)
            v_entity_class_ids = torch.tensor(
                [label_to_entity_class_id[item[0]] for item in vb], dtype=torch.long,
            )

            val_batches_fixed.append((v_input_ids, v_attn_mask, v_targets, v_class_ids, v_entity_class_ids))
        log(f"Precomputed {len(val_batches_fixed)} fixed val batches (P×K, K={K})")

    # ── Training loop ──────────────────────────────────────────────
    processed_tokens = 0
    updates = 0
    epoch = 0
    best_val_loss = float("inf")
    best_knn_acc = 0.0

    log_path = os.path.join(out_dir, f"behavior_cluster_log_{model_size}.txt")
    with open(log_path, "w") as f:
        f.write("update,tokens,total_loss,bce_loss,contrastive_loss,lr,time\n")

    while processed_tokens < total_tokens:
        epoch += 1
        epoch_st = time.time()
        epoch_total_loss = 0.0
        epoch_bce_loss = 0.0
        epoch_con_loss = 0.0
        epoch_batches = 0

        # Group labels by signature class, apply floor/cap rebalancing
        class_to_labels = defaultdict(list)
        for label_key in train_label_to_entities:
            class_to_labels[label_to_contrastive_id[label_key]].append(label_key)

        epoch_entities_by_class = defaultdict(list)
        for class_id, class_label_keys in class_to_labels.items():
            n_class = len(class_label_keys)
            # Cap per-class to avoid dominant classes flooding training
            if bc_max_per_class > 0 and n_class > bc_max_per_class:
                offset = (epoch - 1) * bc_max_per_class % n_class
                indices = [(offset + i) % n_class for i in range(bc_max_per_class)]
                class_label_keys = [class_label_keys[i] for i in indices]
            # Floor per-class: oversample small classes by repeating labels
            elif bc_min_per_class > 0 and n_class < bc_min_per_class:
                repeats = (bc_min_per_class + n_class - 1) // n_class
                class_label_keys = (class_label_keys * repeats)[:bc_min_per_class]
            for label_key in class_label_keys:
                entities = train_label_to_entities[label_key]
                pick = entities[(epoch - 1) % len(entities)]
                epoch_entities_by_class[class_id].append((label_key, pick, class_id))

        # ── P×K batch construction ────────────────────────────────
        # Each batch contains P classes × K samples per class,
        # guaranteeing every entity has K-1 positives for contrastive loss.
        for items in epoch_entities_by_class.values():
            random.shuffle(items)

        epoch_batches_list = build_pk_batches(epoch_entities_by_class, P, K)

        for batch_items in epoch_batches_list:
            B = len(batch_items)
            if B == 0:
                continue

            # ── Pad encoder inputs ─────────────────────────────────
            enc_ids_list = [item[1][2] for item in batch_items]
            input_ids, attention_mask = pad_token_id_lists(enc_ids_list, tokenizer.pad_id, tokenizer.max_seq_len)

            # ── Targets ────────────────────────────────────────────
            target_vecs = torch.stack([label_to_target[item[0]] for item in batch_items])
            class_ids = torch.tensor([item[2] for item in batch_items], dtype=torch.long)
            entity_class_ids = torch.tensor(
                [label_to_entity_class_id[item[0]] for item in batch_items], dtype=torch.long,
            )

            # ── Forward ────────────────────────────────────────────
            model.train()
            out = model.modified_fwd(input_ids.to(device), attention_mask.to(device))

            total_loss, bce_loss_val, con_loss_val = behavior_combined_loss(
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            opt.zero_grad()

            # ── Tracking ───────────────────────────────────────────
            n_tokens = int(attention_mask.sum().item())
            if n_tokens == 0:
                continue

            processed_tokens += n_tokens
            scheduler.last_epoch = processed_tokens
            scheduler.step()
            updates += 1

            tl = total_loss.item()
            bl = bce_loss_val.item()
            cl = con_loss_val.item()
            epoch_total_loss += tl
            epoch_bce_loss += bl
            epoch_con_loss += cl
            epoch_batches += 1
            current_lr = scheduler.get_last_lr()[0]

            if updates % 100 == 0:
                elapsed = time.time() - epoch_st
                log(
                    f"[{updates}|e{epoch}] loss={tl:.4f} (bce={bl:.4f} con={cl:.4f}) "
                    f"lr={current_lr:.2e} tokens={processed_tokens:.2e} ({elapsed:.1f}s)"
                )
                with open(log_path, "a") as f:
                    f.write(f"{updates},{processed_tokens},{tl:.6f},"
                            f"{bl:.6f},{cl:.6f},{current_lr:.2e},{elapsed:.1f}\n")

            if processed_tokens >= total_tokens:
                break

        if processed_tokens >= total_tokens:
            break

        # Epoch-level logging
        if epoch_batches > 0:
            avg_tl = epoch_total_loss / epoch_batches
            avg_bl = epoch_bce_loss / epoch_batches
            avg_cl = epoch_con_loss / epoch_batches
            elapsed = time.time() - epoch_st
            log(f"Epoch {epoch} done | loss={avg_tl:.4f} (bce={avg_bl:.4f} con={avg_cl:.4f}) "
                f"batches={epoch_batches} ({elapsed:.1f}s)")

        # ── Validation ─────────────────────────────────────────────
        epoch_metrics = {"pretrain/epoch": epoch, "pretrain/tokens": processed_tokens}

        if val_batches_fixed:
            model.eval()
            val_losses = []
            val_bce_losses = []
            val_con_losses = []

            for (v_input_ids, v_attn_mask, v_targets, v_class_ids, v_ecids) in val_batches_fixed:
                with torch.no_grad():
                    v_out = model.modified_fwd(v_input_ids.to(device), v_attn_mask.to(device))
                    v_total, v_bce, v_con = behavior_combined_loss(
                        v_out.logits, v_targets.to(device),
                        v_out.projection, v_class_ids.to(device),
                        bce_weight=bc_bce_weight,
                        contrastive_weight=bc_contrastive_weight,
                        temperature=bc_temperature,
                        bce_mode=bc_bce_mode,
                        entity_class_labels=v_ecids.to(device),
                        contrastive_target=bc_contrastive_target,
                    )
                    val_losses.append(v_total.item())
                    val_bce_losses.append(v_bce.item())
                    val_con_losses.append(v_con.item())

            avg_val = sum(val_losses) / len(val_losses)
            avg_val_bce = sum(val_bce_losses) / len(val_bce_losses)
            avg_val_con = sum(val_con_losses) / len(val_con_losses)
            log(f"Epoch {epoch} val_loss={avg_val:.4f} (bce={avg_val_bce:.4f} con={avg_val_con:.4f})")
            epoch_metrics["pretrain/val_loss"] = avg_val
            epoch_metrics["pretrain/val_bce_loss"] = avg_val_bce
            epoch_metrics["pretrain/val_contrastive_loss"] = avg_val_con

        # ── KNN eval ───────────────────────────────────────────────
        knn_results = _eval_knn_on_eval_set(model, tokenizer, device, "behavior_cluster", batch_size)
        if knn_results:
            knn_acc = knn_results['knn_accuracy_k5']
            epoch_metrics["pretrain/knn_accuracy_k5"] = knn_acc
            for cl, acc in knn_results['knn_accuracy_k5_per_cluster'].items():
                epoch_metrics[f"pretrain/knn_k5/{cl}"] = acc
            epoch_metrics["pretrain/ari"] = knn_results['ari']
            epoch_metrics["pretrain/nmi"] = knn_results['nmi']
            epoch_metrics["pretrain/silhouette"] = knn_results['silhouette']
            epoch_metrics["pretrain/mAP"] = knn_results['mAP']
            if knn_acc > best_knn_acc:
                best_knn_acc = knn_acc
                torch.save(model.state_dict(), os.path.join(out_dir, f"pretrain_{model_size}_best.pt"))
                log(f"Epoch {epoch} knn_accuracy_k5={knn_acc:.4f} "
                    f"ARI={knn_results['ari']:.4f} NMI={knn_results['nmi']:.4f} "
                    f"silhouette={knn_results['silhouette']:.4f} mAP={knn_results['mAP']:.4f} "
                    f"*** new best ***")
            else:
                log(f"Epoch {epoch} knn_accuracy_k5={knn_acc:.4f} "
                    f"ARI={knn_results['ari']:.4f} NMI={knn_results['nmi']:.4f} "
                    f"silhouette={knn_results['silhouette']:.4f} mAP={knn_results['mAP']:.4f}")
            epoch_metrics["pretrain/best_knn_acc"] = best_knn_acc

        # Always save latest model
        torch.save(model.state_dict(), os.path.join(out_dir, f"pretrain_{model_size}.pt"))
        wandb.log(epoch_metrics)

    # Final save
    torch.save(model.state_dict(), os.path.join(out_dir, f"pretrain_{model_size}.pt"))
    tokenizer.save(os.path.join(out_dir, "tokenizer.pt"))
    log(f"Behavior cluster pretraining complete. {processed_tokens:,} tokens in {epoch} epochs.")
