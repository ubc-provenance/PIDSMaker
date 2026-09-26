"""HuggingFace pretrained model fine-tuning branch for SPIDER."""

import os
import random
import time

import torch
from torch.optim import AdamW

import wandb

from pidsmaker.utils.utils import log
from ..training_utils import WarmupLinearScheduler, WarmupCosineScheduler, pad_token_id_lists


def pretrain_hf_model(cfg, model_type, model_size, indexid2msg, out_dir,
                      device, total_tokens, warmup_tokens, batch_size, lr,
                      scheduler_type, max_seq_len=None, micro_batch_size=None):
    """Fine-tune a HuggingFace pretrained causal LM on entity labels.

    Supports GPT-2 (small/medium/large/xl), Llama 3.2 (1b/3b), and OPT.
    Trains with next-token prediction on deduplicated entity labels
    (no walk sampling), then saves the model + tokenizer for later
    embedding extraction in feat_inference.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from ..models.hf_pretrained import (
        get_hf_model_name, get_hf_hidden_size, pretokenize_labels_hf,
    )

    # ── Resolve emb_dim from HF model (hidden size is fixed) ────────────
    emb_dim = cfg.featurization.feat_training.emb_dim
    hidden_size = get_hf_hidden_size(model_type, model_size)
    if emb_dim != hidden_size:
        log(f"Overriding emb_dim {emb_dim} → {hidden_size} to match "
            f"{model_type} hidden size")
        emb_dim = hidden_size

    # ── Load pretrained model + tokenizer ────────────────────────────────
    model_name = get_hf_model_name(model_type, model_size)
    log(f"Loading pretrained {model_name} from HuggingFace...")
    hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)

    if hf_tokenizer.pad_token is None:
        hf_tokenizer.pad_token = hf_tokenizer.eos_token
        model.config.pad_token_id = hf_tokenizer.eos_token_id

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"Model: {model_name} — {n_params:,} params, H={hidden_size}")

    # ── Pre-tokenize entity labels ────────────────────────────────────────
    pos_limit = model.config.max_position_embeddings
    if max_seq_len is not None:
        hf_max_seq_len = min(max_seq_len, pos_limit)
    else:
        hf_max_seq_len = min(pos_limit, 512)
    log(f"HF max_seq_len={hf_max_seq_len} (pos_limit={pos_limit})")
    log("Pre-tokenizing entity labels with HF tokenizer...")
    pretokenized = pretokenize_labels_hf(indexid2msg, hf_tokenizer, hf_max_seq_len)
    log(f"Pre-tokenized {len(pretokenized):,} unique entity labels "
        f"(from {len(indexid2msg):,} entities)")

    # ── Optimiser + scheduler ────────────────────────────────────────────
    opt = AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), eps=1e-8,
                weight_decay=0.1)
    if scheduler_type == "cosine":
        scheduler = WarmupCosineScheduler(opt, warmup_tokens, total_tokens)
    else:
        scheduler = WarmupLinearScheduler(opt, warmup_tokens, total_tokens)

    # ── Gradient accumulation for large models ──────────────────────────
    if micro_batch_size is None:
        micro_batch_size = min(batch_size, 8)
    accum_steps = max(1, batch_size // micro_batch_size)
    log(f"Gradient accumulation: micro_batch={micro_batch_size}, "
        f"accum_steps={accum_steps}, effective_batch={micro_batch_size * accum_steps}")

    # ── Training loop ────────────────────────────────────────────────────
    processed_tokens = 0
    updates = 0
    epoch = 0
    pad_id = hf_tokenizer.pad_token_id

    log_path = os.path.join(out_dir, f"pretrain_log_{model_size}.txt")
    with open(log_path, "w") as f:
        f.write("update,tokens,loss,lr,mask_rate,time\n")

    log(f"Starting fine-tuning on {len(pretokenized):,} entity labels "
        f"(target {total_tokens:.2e} tokens)...")

    while processed_tokens < total_tokens:
        epoch += 1
        epoch_st = time.time()
        random.shuffle(pretokenized)

        for batch_start in range(0, len(pretokenized), batch_size):
            batch = pretokenized[batch_start:batch_start + batch_size]
            B = len(batch)
            if B == 0:
                continue

            model.train()
            opt.zero_grad()
            accum_loss = 0.0
            accum_tokens = 0

            # Split into micro-batches for gradient accumulation
            for mb_start in range(0, B, micro_batch_size):
                mb = batch[mb_start:mb_start + micro_batch_size]
                input_ids, attention_mask = pad_token_id_lists(mb, pad_id, hf_max_seq_len)

                labels = input_ids.clone()
                labels[~attention_mask] = -100

                if updates == 0 and mb_start == 0:
                    log(f"  First micro-batch shapes: input_ids={input_ids.shape}")
                outputs = model(
                    input_ids=input_ids.to(device),
                    attention_mask=attention_mask.to(device),
                    labels=labels.to(device),
                )
                loss = outputs.loss / accum_steps
                loss.backward()
                accum_loss += outputs.loss.item()
                accum_tokens += int(attention_mask.sum().item())

            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            loss_val = accum_loss / max(1, B // micro_batch_size)

            if accum_tokens == 0:
                continue

            processed_tokens += accum_tokens
            scheduler.last_epoch = processed_tokens
            scheduler.step()
            updates += 1
            current_lr = scheduler.get_last_lr()[0]

            if updates % 100 == 0:
                elapsed = time.time() - epoch_st
                log(f"[{updates}|e{epoch}] loss={loss_val:.4f} "
                    f"lr={current_lr:.2e} tokens={processed_tokens:.2e} "
                    f"({elapsed:.1f}s)")
                with open(log_path, "a") as f:
                    f.write(f"{updates},{processed_tokens},{loss_val:.6f},"
                            f"{current_lr:.2e},0,{elapsed:.1f}\n")

            if processed_tokens >= total_tokens:
                break

        if processed_tokens >= total_tokens:
            break

        # Save checkpoint after each epoch
        model.save_pretrained(os.path.join(out_dir, "hf_model"))
        hf_tokenizer.save_pretrained(os.path.join(out_dir, "hf_tokenizer"))

        wandb.log({"pretrain/epoch": epoch, "pretrain/tokens": processed_tokens})

    # ── Final save ───────────────────────────────────────────────────────
    model.save_pretrained(os.path.join(out_dir, "hf_model"))
    hf_tokenizer.save_pretrained(os.path.join(out_dir, "hf_tokenizer"))
    log(f"Fine-tuning complete. {processed_tokens:,} tokens in {epoch} epochs.")
