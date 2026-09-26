"""SPIDER pretraining branch: GNN teacher distilled into a T5 student."""

import json
import os
import random
import time
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.optim import AdamW

import wandb

from pidsmaker.utils.utils import log
from ..training_utils import WarmupCosineScheduler, pad_token_id_lists, update_ema, build_pk_batches
from ..data.tokenizer_bpe import ProvenanceTokenizerBPE as ProvenanceTokenizer
from .pretrain_common import _eval_knn_on_eval_set


def pretrain_spider(cfg, spider_cfg, sampler_pairs, combined_indexid2msg, out_dir, device,
                         model_size, max_seq_len, bpe_vocab_size, tokenizer_mode,
                         normalize_netflow_ips, canonicalize_neighbors, spider_path,
                         total_tokens, warmup_tokens, batch_size, lr):
    from copy import deepcopy
    from ..models.spider import (
        ProvenanceGNNCluster, GNNClusterTeacher,
        get_spider_encoder_config,
        build_gnn_distill_batch, build_edge_type_encoder,
        supervised_contrastive_loss, sce_loss,
        StudentSignatureHead, StudentClassHead, TeacherClassHead,
    )
    from ..data.behavior_signatures import build_behavior_dataset, BehaviorLabelVocab
    from ..data.entity_classes import classify_entity

    gc_cfg = spider_cfg.spider
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
    gc_strip_entity_type = gc_cfg.strip_entity_type
    gc_distill_loss = getattr(gc_cfg, 'distill_loss', 'sce')
    gc_teacher_loss = getattr(gc_cfg, 'teacher_loss', 'contrastive')
    gc_teacher_data = getattr(gc_cfg, 'teacher_data', 'signature,gnn_emb')
    gc_student_only_mode = getattr(gc_cfg, 'student_only_mode', 'none')

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
        log(f"GNN cluster: {len(walk_label_corpus):,} entity labels for tokenizer")
        tokenizer = ProvenanceTokenizer(
            cfg, max_seq_len=max_seq_len, bpe_vocab_size=bpe_vocab_size,
            mode=tokenizer_mode, normalize_netflow_ips=normalize_netflow_ips,
            canonicalize_neighbors=canonicalize_neighbors,
            strip_entity_type=gc_strip_entity_type,
        )
        tokenizer.build_vocab(combined_indexid2msg, walk_label_corpus=walk_label_corpus)
    log(f"Vocabulary size: {tokenizer.vocab_size}")
    tokenizer.save(os.path.join(out_dir, "tokenizer.pt"))

    # ── Build edge type encoder for GNN ───────────────────────────
    gnn_edge_type2idx = build_edge_type_encoder(sampler_pairs)
    log(f"GNN cluster: {len(gnn_edge_type2idx)} directed edge types")
    with open(os.path.join(out_dir, "edge_type2idx.json"), "w") as f:
        json.dump(gnn_edge_type2idx, f)

    # ── Build behavior dataset (for entity class assignment) ──────
    log("Building behavior signatures for entity class assignment...")
    label_to_signature, label_to_entities_raw, vocab = build_behavior_dataset(
        sampler_pairs, combined_indexid2msg,
        filter_noisy=gc_filter_noisy, min_signature_size=gc_min_sig_size,
    )
    log(f"Behavior signatures: {len(label_to_signature):,} unique entity labels")
    vocab.save(os.path.join(out_dir, "behavior_vocab.txt"))

    # ── Tokenize entities and assign entity class IDs ─────────────
    if gc_filter_noisy:
        from ..data.edge_filter import _is_noisy_file, _is_noisy_process

    label_to_entities = defaultdict(list)
    for label_key, raw_entities in label_to_entities_raw.items():
        ntype, nlabel = label_key
        entity_ids = tokenizer.tokenize_node(ntype, nlabel)
        if entity_ids:
            for node_id, sidx in raw_entities:
                label_to_entities[label_key].append((node_id, sidx, entity_ids))

    # Post-tokenization dedup (same as behavior_cluster)
    token_sig_to_label = {}
    merged_count = 0
    for label_key in list(label_to_entities.keys()):
        entity_ids = label_to_entities[label_key][0][2]
        token_sig = tuple(entity_ids)
        if token_sig in token_sig_to_label:
            canonical = token_sig_to_label[token_sig]
            label_to_signature[canonical] = label_to_signature[canonical] | label_to_signature[label_key]
            label_to_entities[canonical].extend(label_to_entities[label_key])
            del label_to_entities[label_key]
            del label_to_signature[label_key]
            merged_count += 1
        else:
            token_sig_to_label[token_sig] = label_key

    # Entity class IDs (~70 coarse functional classes)
    entity_class_to_id = {}
    label_to_entity_class_id = {}
    for label_key in label_to_entities:
        ntype, nlabel = label_key
        ecls = classify_entity(ntype, nlabel)
        if ecls not in entity_class_to_id:
            entity_class_to_id[ecls] = len(entity_class_to_id)
        label_to_entity_class_id[label_key] = entity_class_to_id[ecls]

    # Pre-compute signature target vectors (binary multilabel)
    label_to_target = {}
    for label_key, sig in label_to_signature.items():
        if label_key in label_to_entities:
            label_to_target[label_key] = vocab.signature_to_vector(sig)

    n_entity_classes = len(entity_class_to_id)
    n_total = sum(len(v) for v in label_to_entities.values())
    log(f"GNN cluster: {n_total:,} trainable entities, "
        f"{len(label_to_entities):,} unique labels "
        f"(deduped {merged_count:,}), {n_entity_classes} entity classes, "
        f"{vocab.size} behavior labels")

    # ── Stratified train/val split by entity class ────────────────
    class_to_label_keys = defaultdict(list)
    for label_key in label_to_entities:
        class_to_label_keys[label_to_entity_class_id[label_key]].append(label_key)

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

    n_val_classes = len({label_to_entity_class_id[lk] for lk in val_label_to_entities})
    log(f"GNN cluster: {len(train_label_to_entities):,} train / "
        f"{len(val_label_to_entities):,} val unique labels "
        f"(val covers {n_val_classes} classes)")

    K = gc_samples_per_class
    P = batch_size // K
    log(f"  P×K batching: K={K} samples/class, P={P} classes/batch")

    # ── Build models ──────────────────────────────────────────────
    enc_config = get_spider_encoder_config(tokenizer.vocab_size, model_size, max_seq_len)
    t5_hidden_dim = enc_config.d_model
    use_teacher = gc_student_only_mode == "none"

    model = ProvenanceGNNCluster(enc_config, gc_hidden_dim).to(device)
    log(f"Model: GNNCluster-{model_size} (T5 encoder H={t5_hidden_dim} → {gc_hidden_dim}d GNN space)")
    log(f"  Student params: {sum(p.numel() for p in model.parameters()):,}")

    # Student-only ablation heads
    student_sig_head = None
    student_cls_head = None
    if gc_student_only_mode == "student_signature":
        student_sig_head = StudentSignatureHead(gc_hidden_dim, vocab.size).to(device)
        log(f"  Student-only ablation: predict signature ({vocab.size} labels)")
    elif gc_student_only_mode == "student_class":
        student_cls_head = StudentClassHead(gc_hidden_dim, n_entity_classes).to(device)
        log(f"  Student-only ablation: predict entity class ({n_entity_classes} classes)")

    # Teacher components (only when not in student-only mode)
    ema_teacher = None
    gnn_teacher = None
    teacher_cls_head = None
    if use_teacher:
        ema_teacher = deepcopy(model).to(device)
        ema_teacher.requires_grad_(False)
        ema_teacher.eval()
        log(f"  EMA teacher: copy of student (momentum={gc_ema_momentum})")

        gnn_teacher = GNNClusterTeacher(
            node_dim=t5_hidden_dim,
            num_edge_types=max(len(gnn_edge_type2idx), 1),
            edge_emb_dim=gc_edge_emb_dim,
            hidden_dim=gc_hidden_dim,
            num_labels=vocab.size,
            proj_dim=gc_proj_dim,
            num_heads=gc_num_heads,
            teacher_data=gc_teacher_data,
        ).to(device)
        log(f"  GNN teacher params: {sum(p.numel() for p in gnn_teacher.parameters()):,}")
        log(f"  teacher_data={gc_teacher_data}, teacher_loss={gc_teacher_loss}, "
            f"distill_loss={gc_distill_loss}")
        log(f"  supcon_weight={gc_supcon_weight}, distill_weight={gc_distill_weight}, τ={gc_temperature}")

        if gc_teacher_loss == "bce":
            teacher_cls_head = TeacherClassHead(gc_proj_dim, n_entity_classes).to(device)
            log(f"  Teacher BCE head: {gc_proj_dim} → {n_entity_classes} classes")

    # ── Optimizer ─────────────────────────────────────────────────
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
        lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1,
    )
    scheduler = WarmupCosineScheduler(opt, warmup_tokens, total_tokens)

    # ── Training loop ─────────────────────────────────────────────
    processed_tokens = 0
    updates = 0
    epoch = 0
    best_knn_acc = -1.0

    log_path = os.path.join(out_dir, f"spider_log_{model_size}.txt")
    with open(log_path, "w") as f:
        f.write("update,tokens,loss1,loss2,lr,time\n")

    # EMA embedding cache
    ema_cache = {}
    ema_cache_refresh_interval = 50
    ema_cache_last_refresh = -ema_cache_refresh_interval

    while processed_tokens < total_tokens:
        epoch += 1
        epoch_st = time.time()
        epoch_supcon = 0.0
        epoch_distill = 0.0
        epoch_batches = 0

        # ── P×K batch construction by entity class ────────────────
        class_to_labels = defaultdict(list)
        for label_key in train_label_to_entities:
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
                entities = train_label_to_entities[label_key]
                pick = entities[(epoch - 1) % len(entities)]
                epoch_entities_by_class[class_id].append((label_key, pick, class_id))

        for items in epoch_entities_by_class.values():
            random.shuffle(items)

        epoch_batches_list = build_pk_batches(epoch_entities_by_class, P, K)

        for batch_items in epoch_batches_list:
            B = len(batch_items)
            if B == 0:
                continue

            # ── Pad encoder inputs ────────────────────────────────
            enc_ids_list = [item[1][2] for item in batch_items]
            input_ids, attention_mask = pad_token_id_lists(enc_ids_list, tokenizer.pad_id, tokenizer.max_seq_len)

            entity_class_ids = torch.tensor(
                [label_to_entity_class_id[item[0]] for item in batch_items], dtype=torch.long,
            )

            # ── Build GNN neighborhood batch ──────────────────────
            entity_nodes = [item[1][0] for item in batch_items]
            sampler_idxs = [item[1][1] for item in batch_items]

            gnn_batch = build_gnn_distill_batch(
                entity_nodes, sampler_idxs, sampler_pairs,
                combined_indexid2msg, gnn_edge_type2idx,
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

            # ── Encode neighbor nodes with EMA teacher (cached) ───
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

                    ema_input_ids, ema_attn_mask = pad_token_id_lists(ema_token_ids, tokenizer.pad_id, tokenizer.max_seq_len)

                    ema_teacher.eval()
                    with torch.no_grad():
                        ema_embeddings = ema_teacher.mean_pool(ema_input_ids, ema_attn_mask)
                    for i, lbl in enumerate(all_labels_to_encode):
                        ema_cache[lbl] = ema_embeddings[i]

                node_embeddings = torch.stack([ema_cache[lbl] for lbl in node_labels])

            # ── Forward passes ────────────────────────────────────
            model.train()

            # Student: T5 encoder → project to GNN space
            out = model.modified_fwd(input_ids.to(device), attention_mask.to(device))
            t5_projected = out.projected  # [B, hidden_dim]

            valid_mask_cpu = valid_mask.cpu()
            valid_class_ids = entity_class_ids[valid_mask_cpu].to(device)

            if use_teacher:
                # ── Teacher-student mode ──────────────────────────
                gnn_teacher.train()
                if teacher_cls_head is not None:
                    teacher_cls_head.train()

                # Build signature vectors for valid entities
                valid_label_keys = [batch_items[i][0] for i in range(B) if valid_mask_cpu[i]]
                sig_vecs = torch.stack([label_to_target[lk] for lk in valid_label_keys]).to(device)

                # GNN teacher: encode neighborhood + fuse
                teacher_proj = gnn_teacher(
                    node_embeddings, edge_index, edge_types, center_idx,
                    signature_vectors=sig_vecs,
                )

                # L_teacher: SupCon or BCE on entity class
                if gc_teacher_loss == "bce":
                    cls_logits = teacher_cls_head(teacher_proj)
                    supcon_loss = F.cross_entropy(cls_logits, valid_class_ids)
                else:
                    supcon_loss = supervised_contrastive_loss(
                        teacher_proj, valid_class_ids, temperature=gc_temperature,
                    )

                # L_distill: SCE or MSE
                if gc_distill_loss == "mse":
                    distill_loss = F.mse_loss(t5_projected[valid_mask], teacher_proj.detach())
                else:
                    distill_loss = sce_loss(t5_projected[valid_mask], teacher_proj.detach())

                total_loss = gc_supcon_weight * supcon_loss + gc_distill_weight * distill_loss
            else:
                # ── Student-only ablation modes ───────────────────
                valid_label_keys = [batch_items[i][0] for i in range(B) if valid_mask_cpu[i]]

                if gc_student_only_mode == "student_signature":
                    student_sig_head.train()
                    sig_logits = student_sig_head(t5_projected[valid_mask])
                    sig_targets = torch.stack([label_to_target[lk] for lk in valid_label_keys]).to(device)
                    supcon_loss = F.binary_cross_entropy_with_logits(sig_logits, sig_targets)
                    distill_loss = torch.tensor(0.0, device=device)
                elif gc_student_only_mode == "student_class":
                    student_cls_head.train()
                    cls_logits = student_cls_head(t5_projected[valid_mask])
                    supcon_loss = F.cross_entropy(cls_logits, valid_class_ids)
                    distill_loss = torch.tensor(0.0, device=device)

                total_loss = supcon_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            if gnn_teacher is not None:
                torch.nn.utils.clip_grad_norm_(gnn_teacher.parameters(), 5.0)
            opt.step()
            opt.zero_grad()

            # ── EMA update ────────────────────────────────────────
            if ema_teacher is not None:
                update_ema(ema_teacher, model, gc_ema_momentum)

            # ── Tracking ──────────────────────────────────────────
            n_tokens = int(attention_mask.sum().item())
            if n_tokens == 0:
                continue

            processed_tokens += n_tokens
            scheduler.last_epoch = processed_tokens
            scheduler.step()
            updates += 1

            sc_val = supcon_loss.item()
            dl_val = distill_loss.item()
            epoch_supcon += sc_val
            epoch_distill += dl_val
            epoch_batches += 1
            current_lr = scheduler.get_last_lr()[0]

            if updates % 100 == 0:
                elapsed = time.time() - epoch_st
                log(
                    f"[{updates}|e{epoch}] supcon={sc_val:.4f} distill={dl_val:.4f} "
                    f"valid={n_valid}/{B} lr={current_lr:.2e} "
                    f"tokens={processed_tokens:.2e} ({elapsed:.1f}s)"
                )
                with open(log_path, "a") as f:
                    f.write(f"{updates},{processed_tokens},{sc_val:.6f},"
                            f"{dl_val:.6f},{current_lr:.2e},{elapsed:.1f}\n")

            if processed_tokens >= total_tokens:
                break

        if processed_tokens >= total_tokens:
            break

        # Epoch-level logging
        if epoch_batches > 0:
            avg_sc = epoch_supcon / epoch_batches
            avg_dl = epoch_distill / epoch_batches
            elapsed = time.time() - epoch_st
            log(f"Epoch {epoch} done | supcon={avg_sc:.4f} distill={avg_dl:.4f} "
                f"batches={epoch_batches} ({elapsed:.1f}s)")

        # ── KNN eval ──────────────────────────────────────────────
        epoch_metrics = {"pretrain/epoch": epoch, "pretrain/tokens": processed_tokens}
        knn_results = _eval_knn_on_eval_set(model, tokenizer, device, "spider", batch_size)
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
                if gnn_teacher is not None:
                    torch.save(gnn_teacher.state_dict(), os.path.join(out_dir, f"gnn_teacher_{model_size}_best.pt"))
                log(f"Epoch {epoch} knn_accuracy_k5={knn_acc:.4f} "
                    f"ARI={knn_results['ari']:.4f} NMI={knn_results['nmi']:.4f} "
                    f"silhouette={knn_results['silhouette']:.4f} mAP={knn_results['mAP']:.4f} "
                    f"*** new best ***")
            else:
                log(f"Epoch {epoch} knn_accuracy_k5={knn_acc:.4f} "
                    f"ARI={knn_results['ari']:.4f} NMI={knn_results['nmi']:.4f} "
                    f"silhouette={knn_results['silhouette']:.4f} mAP={knn_results['mAP']:.4f}")
            epoch_metrics["pretrain/best_knn_acc"] = best_knn_acc

        torch.save(model.state_dict(), os.path.join(out_dir, f"pretrain_{model_size}.pt"))
        if gnn_teacher is not None:
            torch.save(gnn_teacher.state_dict(), os.path.join(out_dir, f"gnn_teacher_{model_size}.pt"))
        wandb.log(epoch_metrics)

    # Final save
    torch.save(model.state_dict(), os.path.join(out_dir, f"pretrain_{model_size}.pt"))
    if gnn_teacher is not None:
        torch.save(gnn_teacher.state_dict(), os.path.join(out_dir, f"gnn_teacher_{model_size}.pt"))
    tokenizer.save(os.path.join(out_dir, "tokenizer.pt"))
    log(f"GNN cluster pretraining complete. {processed_tokens:,} tokens in {epoch} epochs.")
