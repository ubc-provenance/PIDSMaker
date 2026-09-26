"""MLM / T5 / Causal LM pretraining branch for SPIDER.

Handles BERT, RoBERTa, ModernBERT, RoPEBERT, LLaMA, LogBERT, and T5 model types.
"""

import os
import random
import time

import torch
from torch.optim import AdamW

import wandb

from pidsmaker.utils.utils import log
from ..training_utils import (
    WarmupLinearScheduler, WarmupCosineScheduler,
    pad_token_id_lists,
)
from ..models.bert import ProvenanceBERT, get_bert_config
from ..models.roberta import ProvenanceRoBERTa, get_roberta_config
from ..models.modernbert import ProvenanceModernBERT, get_modernbert_config
from ..models.ropebert import ProvenanceRoPEBERT, get_ropebert_config
from ..models.llama import ProvenanceLLaMA, get_llama_config
from ..models.t5 import ProvenanceT5, get_t5_config
from ..data.tokenizer_bpe import ProvenanceTokenizerBPE as ProvenanceTokenizer
from ..data.sampler import ProvenanceWalkSampler
from .pretrain_common import (
    _extract_walk_label_corpus,
    _dump_corpus_to_txt,
    _dump_t5_corpus_to_txt,
    _eval_knn_on_eval_set,
)


def _build_pretokenized_val_corpus(val_data, tokenizer, walk_length, num_walks,
                                    time_weight, half_life, model_type, mask_edge_type,
                                    max_val_nodes_per_dataset=5000):
    """Pre-tokenize validation walks once to avoid repeated sampling/tokenization each epoch.

    Loads each validation graph, samples walks from all nodes, and pre-tokenizes them.
    Returns pre-tokenized examples in the same format as the training corpus.

    To keep the val set small and balanced across N datasets, each dataset contributes
    at most min(total_nodes/N, max_val_nodes_per_dataset) nodes (randomly sampled),
    so the val set stays bounded regardless of dataset count or size.
    """
    is_causal_lm = model_type == "llama"
    is_encoder_decoder = model_type == "t5"

    pretok_val = []
    seen_sigs: set = set()
    total_nodes = 0

    for graph_paths, ds_indexid2msg in val_data:
        # Distribute the per-dataset budget proportionally across files so we
        # don't need to load all graphs before subsampling.
        # Distribute the per-dataset budget proportionally across files so we
        # don't need to hold all graph samplers in memory simultaneously.
        n_files = len(graph_paths)
        k_per_file = max(1, max_val_nodes_per_dataset // n_files)
        budget = max_val_nodes_per_dataset

        for path in graph_paths:
            if budget <= 0:
                break
            graph = torch.load(path)
            sampler = ProvenanceWalkSampler(
                graph, walk_length=walk_length, num_walks=num_walks,
                time_weight=time_weight, half_life=half_life,
            )
            all_nodes = list(sampler.nodes)
            k = min(k_per_file, len(all_nodes), budget)
            sampled_nodes = random.sample(all_nodes, k)
            budget -= k
            total_nodes += k

            # Process inline while sampler is alive (graph can be GC'd after this loop)
            for node in sampled_nodes:
                walk_nodes, walk_edge_types, _, entity_pos = sampler._single_walk(node)
                if len(walk_nodes) <= 1:
                    continue

                if is_encoder_decoder:
                    entity_node = walk_nodes[entity_pos]
                    if entity_node not in ds_indexid2msg:
                        continue
                    ntype, nlabel = ds_indexid2msg[entity_node]
                    entity_ids = tokenizer.tokenize_node(ntype, nlabel)
                    if not entity_ids:
                        continue

                    for direction, edge_slice, node_slice in [
                        ("forward",  walk_edge_types[entity_pos:], walk_nodes[entity_pos + 1:]),
                        ("backward", list(reversed(walk_edge_types[:entity_pos])), list(reversed(walk_nodes[:entity_pos]))),
                    ]:
                        if not node_slice:
                            continue
                        dir_id = tokenizer.forward_id if direction == "forward" else tokenizer.backward_id
                        decoder_ids = [dir_id] + tokenizer.tokenize_context_segment(
                            edge_slice, node_slice, ds_indexid2msg
                        )
                        if len(decoder_ids) > 1:
                            sig = (tuple(entity_ids), tuple(decoder_ids))
                            if sig not in seen_sigs:
                                seen_sigs.add(sig)
                                pretok_val.append((entity_ids, decoder_ids))
                else:
                    token_ids, node_boundaries, edge_boundaries = tokenizer.tokenize_walk(
                        walk_nodes, walk_edge_types, ds_indexid2msg
                    )
                    if not token_ids:
                        continue
                    sig = tuple(token_ids)
                    if sig not in seen_sigs:
                        seen_sigs.add(sig)
                        pretok_val.append((token_ids, node_boundaries, edge_boundaries))

    log(f"Val corpus: {len(pretok_val):,} unique examples from {total_nodes:,} nodes")
    return pretok_val


@torch.no_grad()
def _evaluate(model, pretok_val, tokenizer, device, batch_size, model_type, mask_edge_type=True):
    """Evaluate prediction loss on pre-tokenized validation examples."""
    model.eval()
    total_loss = 0
    n_batches = 0
    is_causal_lm = model_type == "llama"
    is_encoder_decoder = model_type == "t5"

    for batch_start in range(0, len(pretok_val), batch_size):
        batch = pretok_val[batch_start:batch_start + batch_size]
        if not batch:
            continue

        if is_encoder_decoder:
            enc_ids_list = [entry[0] for entry in batch]
            dec_ids_list = [entry[1] for entry in batch]
            B = len(batch)
            input_ids, attn_mask = pad_token_id_lists(enc_ids_list, tokenizer.pad_id, tokenizer.max_seq_len)
            labels, _ = pad_token_id_lists(dec_ids_list, -100, tokenizer.max_seq_len)

        elif is_causal_lm:
            batch_token_ids = [tok for tok, _, _ in batch]
            B = len(batch)
            input_ids, attn_mask = pad_token_id_lists(batch_token_ids, tokenizer.pad_id, tokenizer.max_seq_len)
            labels = input_ids.clone()
            labels[~attn_mask] = -100

        else:
            # MLM: evaluate with fixed masking (no stochastic mask here — use same rate)
            batch_masked = []
            batch_targets = []
            for token_ids, node_boundaries, edge_boundaries in batch:
                if mask_edge_type and edge_boundaries:
                    masked_ids, targets, p_mask = tokenizer.mask_walk(
                        token_ids, node_boundaries, edge_boundaries
                    )
                else:
                    masked_ids, targets, p_mask = tokenizer.mask_nodes(
                        token_ids, node_boundaries
                    )
                batch_masked.append(masked_ids)
                batch_targets.append((targets, p_mask))

            input_ids, attn_mask = pad_token_id_lists(batch_masked, tokenizer.pad_id, tokenizer.max_seq_len)
            B, max_len = input_ids.shape
            labels = torch.full((B, max_len), -100, dtype=torch.long)
            predict_tensor = torch.zeros(B, max_len, dtype=torch.bool)
            for i in range(B):
                seq_len = min(len(batch_masked[i]), max_len)
                targets, p_mask = batch_targets[i]
                t_idx = 0
                for pos in range(seq_len):
                    if p_mask[pos]:
                        labels[i, pos] = targets[t_idx]
                        predict_tensor[i, pos] = True
                        t_idx += 1
            if predict_tensor.sum() == 0:
                continue

        loss = model.modified_fwd(
            input_ids.to(device),
            attn_mask.to(device),
            labels.to(device),
        )
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(1, n_batches)


def pretrain_mlm(cfg, spider_cfg, walk_corpus, sampler_pairs, combined_indexid2msg, all_val_data,
                 out_dir, device, model_type, model_size, max_seq_len, bpe_vocab_size, tokenizer_mode,
                 normalize_netflow_ips, canonicalize_neighbors, spider_path,
                 total_tokens, warmup_tokens, batch_size, lr, scheduler_type,
                 mask_rate_fixed, mask_rate_min, mask_edge_type, hvm_weight,
                 walk_dedup_enabled, use_synthetic_data, synthetic_corpus, synthetic_val_corpus):
    # ── Build or load tokenizer ─────────────────────────────────────────
    if spider_path:
        tok_path = os.path.join(spider_path, "tokenizer.pt")
        log(f"Loading tokenizer from {tok_path}")
        tokenizer = ProvenanceTokenizer(cfg)
        tokenizer.load(tok_path)
        # Override max_seq_len with the current config value (the saved tokenizer
        # may have been trained with a different max_seq_len)
        tokenizer.max_seq_len = max_seq_len
        log(f"  Overriding max_seq_len to {max_seq_len} (from config)")
        tokenizer.set_mask_rate(0, fixed_rate=mask_rate_fixed, min_rate=mask_rate_min)
    else:
        walk_label_corpus = _extract_walk_label_corpus(walk_corpus, model_type)
        log(f"Walk label corpus: {len(walk_label_corpus):,} (node_type, label) pairs "
            f"from {len(walk_corpus):,} unique walks")

        log("Building tokenizer...")
        tokenizer = ProvenanceTokenizer(cfg, max_seq_len=max_seq_len, bpe_vocab_size=bpe_vocab_size, mode=tokenizer_mode,
                                        normalize_netflow_ips=normalize_netflow_ips,
                                        canonicalize_neighbors=canonicalize_neighbors)
        tokenizer.build_vocab(combined_indexid2msg, walk_label_corpus=walk_label_corpus)
        tokenizer.set_mask_rate(0, fixed_rate=mask_rate_fixed, min_rate=mask_rate_min)
    log(f"Vocabulary size: {tokenizer.vocab_size}")
    tokenizer.save(os.path.join(out_dir, "tokenizer.pt"))
    vocab_dump_path = os.path.join(out_dir, "vocab.txt")
    tokenizer.dump_vocab(vocab_dump_path)
    log(f"Vocab dumped to {vocab_dump_path}")

    # ── Dump corpus for inspection (non-T5; T5 dumps after tokenized dedup) ──
    _CORPUS_DUMP_LIMIT = 2_000_000
    if model_type != "t5":
        if len(walk_corpus) > _CORPUS_DUMP_LIMIT:
            log(f"Skipping walk corpus dump ({len(walk_corpus):,} walks > {_CORPUS_DUMP_LIMIT:,} limit)")
        else:
            log(f"Dumping walk corpus ({len(walk_corpus):,} walks)...")
            _dump_corpus_to_txt(walk_corpus, tokenizer, out_dir)

    # ── Pre-tokenize validation corpus (once, reused every epoch) ────────
    walk_length = spider_cfg.corpus.walk_length
    num_walks = spider_cfg.corpus.num_walks
    time_weight = spider_cfg.corpus.time_weight
    half_life = spider_cfg.corpus.half_life

    pretok_val = []
    if all_val_data:
        log("Pre-tokenizing validation corpus...")
        val_st = time.time()
        pretok_val = _build_pretokenized_val_corpus(
            all_val_data, tokenizer, walk_length, num_walks,
            time_weight, half_life, model_type, mask_edge_type,
        )
        log(f"Val corpus pre-tokenized in {time.time() - val_st:.1f}s")

    # ── Add synthetic walks to the validation set ────────────────────────
    if use_synthetic_data and model_type == "t5" and synthetic_corpus:
        synth_val_before = len(pretok_val)
        for entity_id, direction, ctx_edges, ctx_nodes, s_indexid2msg in synthetic_val_corpus:
            if entity_id not in s_indexid2msg:
                continue
            ntype, nlabel = s_indexid2msg[entity_id]
            entity_ids = tokenizer.tokenize_node(ntype, nlabel)
            if not entity_ids:
                continue
            dir_id = tokenizer.forward_id if direction == "forward" else tokenizer.backward_id
            decoder_ids = [dir_id] + tokenizer.tokenize_context_segment(
                ctx_edges, ctx_nodes, s_indexid2msg
            )
            if len(decoder_ids) > 1:
                pretok_val.append((entity_ids, decoder_ids))
        log(f"Added {len(pretok_val) - synth_val_before:,} synthetic walks to validation set")

    # ── Build model ─────────────────────────────────────────────────────
    if model_type == "ropebert":
        rope_theta = spider_cfg.ropebert.rope_theta
        bert_config = get_ropebert_config(
            tokenizer.vocab_size, model_size, max_seq_len,
            rope_theta=rope_theta,
        )
        model = ProvenanceRoPEBERT(bert_config).to(device)
        log(f"Model: RoPEBERT-{model_size} (theta={rope_theta})")
    elif model_type == "llama":
        rope_theta = spider_cfg.llama.rope_theta
        llama_config = get_llama_config(
            tokenizer.vocab_size, model_size, max_seq_len,
            rope_theta=rope_theta,
        )
        model = ProvenanceLLaMA(llama_config).to(device)
        log(f"Model: LLaMA-{model_size} (theta={rope_theta})")
    elif model_type == "t5":
        t5_config = get_t5_config(tokenizer.vocab_size, model_size, max_seq_len)
        model = ProvenanceT5(t5_config).to(device)
        log(f"Model: T5-{model_size}")
    elif model_type == "modernbert":
        mb_cfg = spider_cfg.modernbert
        bert_config = get_modernbert_config(
            tokenizer.vocab_size, model_size, max_seq_len,
            global_attn_every_n_layers=mb_cfg.global_attn_every_n_layers,
            local_attention_window=mb_cfg.local_attention_window
        )
        model = ProvenanceModernBERT(bert_config).to(device)
        log(f"Model: ModernBERT-{model_size} (global every {mb_cfg.global_attn_every_n_layers}, "
            f"window {mb_cfg.local_attention_window})")
    elif model_type == "roberta":
        bert_config = get_roberta_config(tokenizer.vocab_size, model_size, max_seq_len)
        model = ProvenanceRoBERTa(bert_config).to(device)
        log(f"Model: RoBERTa-{model_size}")
    elif model_type == "logbert":
        bert_config = get_bert_config(tokenizer.vocab_size, model_size, max_seq_len)
        model = ProvenanceBERT(bert_config).to(device)
        log(f"Model: LogBERT-{model_size} (HVM weight={hvm_weight})")
    else:
        bert_config = get_bert_config(tokenizer.vocab_size, model_size, max_seq_len)
        model = ProvenanceBERT(bert_config).to(device)
        log(f"Model: BERT-{model_size}")

    # LogBERT: initialize hypersphere center for volume minimization
    hvm_center = None
    if model_type == "logbert":
        hvm_center = torch.zeros(bert_config.hidden_size, device=device)

    log(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── Optimizer and scheduler ─────────────────────────────────────────
    opt = AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.1,
    )
    if scheduler_type == "cosine":
        scheduler = WarmupCosineScheduler(opt, warmup_tokens, total_tokens)
    else:
        scheduler = WarmupLinearScheduler(opt, warmup_tokens, total_tokens)

    # ── Training loop ───────────────────────────────────────────────────
    processed_tokens = 0
    updates = 0
    epoch = 0
    best_knn_acc = -1.0

    log_path = os.path.join(out_dir, f"pretrain_log_{model_size}.txt")
    with open(log_path, "w") as f:
        f.write("update,tokens,loss,lr,mask_rate,time\n")

    # ── Pre-tokenize walk corpus ──────────────────────────────────────────
    # Tokenization is deterministic (needs indexid2msg), only masking is
    # stochastic. Pre-tokenize once to avoid per-batch repeated BPE encoding.
    is_encoder_decoder = model_type == "t5"
    pretokenized_walks = None
    pretokenized_t5 = None
    log("Pre-tokenizing walk corpus...")
    pt_st = time.time()

    if is_encoder_decoder:
        # T5: each corpus entry is already a directional walk (forward or backward)
        # Dedup on tokenized form — different raw walks can produce identical tokens
        pretokenized_t5 = []
        t5_dump_data = []  # For dump: (ntype, nlabel, direction, context_edges, context_nodes, ds_indexid2msg, decoder_ids)
        seen_tok_sigs = set()

        # Map ds_indexid2msg object identity → sampler index (for GNN distillation)
        _indexid2msg_id_to_sampler_idx = {
            id(ds_im): idx for idx, (_, ds_im) in enumerate(sampler_pairs)
        }

        for entity_node, direction, context_edges, context_nodes, ds_indexid2msg in walk_corpus:
            if entity_node not in ds_indexid2msg:
                continue
            ntype, nlabel = ds_indexid2msg[entity_node]
            entity_ids = tokenizer.tokenize_node(ntype, nlabel)
            if not entity_ids:
                continue

            _dir_map = {
                'forward': tokenizer.forward_id,
                'backward': tokenizer.backward_id,
                'forward_neigh': tokenizer.forward_neigh_id,
                'backward_neigh': tokenizer.backward_neigh_id,
            }
            dir_id = _dir_map.get(direction, tokenizer.backward_id)
            decoder_ids = [dir_id] + tokenizer.tokenize_context_segment(
                context_edges, context_nodes, ds_indexid2msg
            )
            if len(decoder_ids) > 1:  # more than just direction token
                sig = (tuple(entity_ids), tuple(decoder_ids))
                if sig not in seen_tok_sigs:
                    seen_tok_sigs.add(sig)
                    # Store entity_node + sampler_idx for GNN distillation batch construction
                    _sampler_idx = _indexid2msg_id_to_sampler_idx.get(id(ds_indexid2msg), -1)
                    pretokenized_t5.append((entity_ids, decoder_ids, entity_node, _sampler_idx))
                    t5_dump_data.append((ntype, nlabel, direction, context_edges, context_nodes, ds_indexid2msg, decoder_ids))

        log(f"Pre-tokenized {len(pretokenized_t5):,} T5 encoder-decoder pairs "
            f"(exact-deduped from {len(walk_corpus):,} walks, {time.time() - pt_st:.1f}s)")

        if walk_dedup_enabled:
            from ..data.lsh_dedup import walk_dedup
            structural_token_ids = frozenset(
                tok_id
                for tok_str, tok_id in tokenizer.token2id.items()
                if tok_str.startswith("[") and tok_str.endswith("]")
            )
            n_before = len(pretokenized_t5)
            pretokenized_t5, t5_dump_data = walk_dedup(
                pretokenized_t5, t5_dump_data, out_dir,
                structural_token_ids=structural_token_ids,
                id2token=tokenizer.id2token,
            )
            log(f"Walk dedup: {n_before:,} → {len(pretokenized_t5):,} walks "
                f"({n_before - len(pretokenized_t5):,} removed, "
                f"{time.time() - pt_st:.1f}s total)")
        else:
            log("Walk dedup: disabled (set corpus.lsh_dedup: True to enable)")

        _dump_t5_corpus_to_txt(t5_dump_data, tokenizer, out_dir)
        del t5_dump_data
    else:
        # Tokenize + deduplicate on tokenized sequence.
        # Prelabel dedup already removed most duplicates; this catches the remainder
        # where distinct labels collapse to the same tokens after normalization
        # (e.g. two walks differing only in a hash value or numeric ID).
        pretokenized_walks = []
        seen_tok_sigs: set = set()
        n_prelabel = len(walk_corpus)
        for walk_nodes, walk_edge_types, ds_indexid2msg, _epos in walk_corpus:
            token_ids, node_boundaries, edge_boundaries = tokenizer.tokenize_walk(
                walk_nodes, walk_edge_types, ds_indexid2msg
            )
            if token_ids:
                sig = tuple(token_ids)
                if sig not in seen_tok_sigs:
                    seen_tok_sigs.add(sig)
                    pretokenized_walks.append((token_ids, node_boundaries, edge_boundaries))
        log(f"Pre-tokenized {len(pretokenized_walks):,} walks "
            f"(deduped from {n_prelabel:,} prelabel-unique, {time.time() - pt_st:.1f}s)")

    corpus_len = len(pretokenized_t5 or pretokenized_walks or walk_corpus)
    log(f"Starting pretraining with {corpus_len:,} unique examples...")
    is_causal_lm = model_type == "llama"
    uses_masking = not is_causal_lm and not is_encoder_decoder

    while processed_tokens < total_tokens:
        epoch += 1
        epoch_st = time.time()

        # Shuffle corpus for this epoch (provides randomness across epochs)
        if is_encoder_decoder:
            random.shuffle(pretokenized_t5)
        elif pretokenized_walks is not None:
            random.shuffle(pretokenized_walks)
        else:
            random.shuffle(walk_corpus)

        # Process walks in batches
        for batch_start in range(0, corpus_len, batch_size):
            if is_encoder_decoder:
                # ── T5 encoder-decoder: entity → walk context ──
                batch_t5 = pretokenized_t5[batch_start:batch_start + batch_size]
                B = len(batch_t5)
                if B == 0:
                    continue

                # Pad encoder inputs (entity tokens — short, ~5-10 tokens)
                enc_ids_list = [entry[0] for entry in batch_t5]
                input_ids, attention_mask = pad_token_id_lists(enc_ids_list, tokenizer.pad_id, tokenizer.max_seq_len)

                # Pad decoder labels (walk segment tokens)
                dec_ids_list = [entry[1] for entry in batch_t5]
                labels, _ = pad_token_id_lists(dec_ids_list, -100, tokenizer.max_seq_len)

            elif is_causal_lm:
                # ── Causal LM (LLaMA): no masking, next-token prediction ──
                batch_pretok = pretokenized_walks[batch_start:batch_start + batch_size]
                B = len(batch_pretok)
                if B == 0:
                    continue

                batch_token_ids = [token_ids for token_ids, _, _ in batch_pretok]
                input_ids, attention_mask = pad_token_id_lists(batch_token_ids, tokenizer.pad_id, tokenizer.max_seq_len)

                # Labels = input_ids; the model shifts internally for next-token prediction.
                # Set padding positions to -100 so they're ignored in loss.
                labels = input_ids.clone()
                labels[~attention_mask] = -100
            else:
                # ── MLM (BERT variants): apply masking ──
                batch_pretok = pretokenized_walks[batch_start:batch_start + batch_size]
                B = len(batch_pretok)
                if B == 0:
                    continue

                batch_masked = []
                batch_targets = []
                for token_ids, node_boundaries, edge_boundaries in batch_pretok:
                    if mask_edge_type and edge_boundaries:
                        masked_ids, targets, p_mask = tokenizer.mask_walk(
                            token_ids, node_boundaries, edge_boundaries
                        )
                    else:
                        masked_ids, targets, p_mask = tokenizer.mask_nodes(
                            token_ids, node_boundaries
                        )
                    batch_masked.append(masked_ids)
                    batch_targets.append((targets, p_mask))

                input_ids, attention_mask = pad_token_id_lists(batch_masked, tokenizer.pad_id, tokenizer.max_seq_len)
                max_len = input_ids.shape[1]
                labels = torch.full((B, max_len), -100, dtype=torch.long)
                predict_tensor = torch.zeros(B, max_len, dtype=torch.bool)

                for i in range(B):
                    seq_len = min(len(batch_masked[i]), max_len)
                    targets, p_mask = batch_targets[i]
                    t_idx = 0
                    for pos in range(seq_len):
                        if p_mask[pos]:
                            labels[i, pos] = targets[t_idx]
                            predict_tensor[i, pos] = True
                            t_idx += 1

                if predict_tensor.sum() == 0:
                    continue

            # Forward + backward
            model.train()
            if updates == 0:
                log(f"  First batch shapes: input_ids={input_ids.shape}, labels={labels.shape}")
            _input_ids = input_ids.to(device)
            _attention_mask = attention_mask.to(device)
            _labels = labels.to(device)

            # LogBERT: get hidden states, compute MLM loss + HVM from single forward
            if hvm_center is not None:
                hidden = model.modified_fwd(
                    _input_ids, _attention_mask, _labels, skip_cls=True,
                )  # [B, L, H]
                # MLM loss: project hidden → vocab and compute cross-entropy
                prediction_scores = model.cls(hidden)
                mlm_loss = torch.nn.functional.cross_entropy(
                    prediction_scores.view(-1, model.config.vocab_size),
                    _labels.view(-1),
                    ignore_index=-100,
                )
                # HVM loss: mean-pool hidden states → distance to hypersphere center
                mask_f = _attention_mask.unsqueeze(-1).float()
                pooled = (hidden * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)  # [B, H]
                hvm_loss = ((pooled - hvm_center) ** 2).sum(dim=-1).mean()
                loss = mlm_loss + hvm_weight * hvm_loss
                # Update center with exponential moving average (detached)
                with torch.no_grad():
                    hvm_center = 0.99 * hvm_center + 0.01 * pooled.mean(dim=0)
            else:
                loss = model.modified_fwd(_input_ids, _attention_mask, _labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            opt.zero_grad()
            loss = loss.item()

            n_tokens = int(attention_mask.sum().item())

            # ── Track tokens + logging (shared) ──
            if n_tokens == 0:
                continue

            processed_tokens += n_tokens
            scheduler.last_epoch = processed_tokens
            scheduler.step()

            if uses_masking:
                # Update mask rate (MLM only)
                progress = min(1.0, processed_tokens / warmup_tokens)
                tokenizer.set_mask_rate(progress, fixed_rate=mask_rate_fixed, min_rate=mask_rate_min)

            updates += 1
            current_lr = scheduler.get_last_lr()[0]

            if updates % 100 == 0:
                elapsed = time.time() - epoch_st
                mask_info = f" mask={tokenizer.mask_rate:.3f}" if uses_masking else ""
                log(
                    f"[{updates}|e{epoch}] loss={loss:.4f} "
                    f"lr={current_lr:.2e}{mask_info} "
                    f"tokens={processed_tokens:.2e} ({elapsed:.1f}s)"
                )
                with open(log_path, "a") as f:
                    mask_val = tokenizer.mask_rate if uses_masking else 0.0
                    f.write(f"{updates},{processed_tokens},{loss:.6f},{current_lr:.2e},{mask_val:.4f},{elapsed:.1f}\n")

            if processed_tokens >= total_tokens:
                break

        if processed_tokens >= total_tokens:
            break

        # ── Validation ──────────────────────────────────────────────────
        epoch_metrics = {"pretrain/epoch": epoch, "pretrain/tokens": processed_tokens}
        if pretok_val:
            val_loss = _evaluate(model, pretok_val, tokenizer, device, batch_size, model_type, mask_edge_type)
            log(f"Epoch {epoch} val_loss={val_loss:.4f}")
            epoch_metrics["pretrain/val_loss"] = val_loss

        # ── KNN eval on eval set ──────────────────────────────────────
        knn_results = _eval_knn_on_eval_set(model, tokenizer, device, model_type, batch_size)
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
        if hvm_center is not None:
            torch.save(hvm_center.cpu(), os.path.join(out_dir, f"hvm_center_{model_size}.pt"))

        wandb.log(epoch_metrics)

    # Final save (ensures checkpoint is written even when token budget breaks mid-epoch)
    torch.save(model.state_dict(), os.path.join(out_dir, f"pretrain_{model_size}.pt"))
    if hvm_center is not None:
        torch.save(hvm_center.cpu(), os.path.join(out_dir, f"hvm_center_{model_size}.pt"))
    tokenizer.save(os.path.join(out_dir, "tokenizer.pt"))
    log(f"Pretraining complete. {processed_tokens} tokens processed in {epoch} epochs.")
