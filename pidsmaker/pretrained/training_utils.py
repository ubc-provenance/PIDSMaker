"""Shared training utilities for SPIDER pretraining and fine-tuning.

Extracts common patterns: LR schedulers, batch padding, EMA updates,
P×K batch construction.
"""

import math
import random
from collections import defaultdict

import torch
from torch.optim.lr_scheduler import LRScheduler


# ── Learning rate schedulers ──────────────────────────────────────────────────

class WarmupLinearScheduler(LRScheduler):
    """Warmup + linear decay scheduler (matching CyberGFM)."""

    def __init__(self, optimizer, warmup_steps, total_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            scale = self.last_epoch / max(1, self.warmup_steps)
        else:
            scale = max(
                1e-8,
                1 - (self.last_epoch - self.warmup_steps) / (self.total_steps - self.warmup_steps),
            )
        return [group["initial_lr"] * scale for group in self.optimizer.param_groups]


class WarmupCosineScheduler(LRScheduler):
    """Warmup + cosine annealing scheduler."""

    def __init__(self, optimizer, warmup_steps, total_steps, min_lr_ratio=0.1, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            scale = self.last_epoch / max(1, self.warmup_steps)
        else:
            progress = (self.last_epoch - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            scale = self.min_lr_ratio + 0.5 * (1 - self.min_lr_ratio) * (1 + math.cos(math.pi * progress))
        return [group["initial_lr"] * scale for group in self.optimizer.param_groups]


# ── Batch padding ─────────────────────────────────────────────────────────────

def pad_token_id_lists(token_id_lists, pad_id, max_seq_len=None):
    """Pad a list of variable-length token ID lists into a batch.

    Args:
        token_id_lists: List[List[int]] — raw token ID sequences.
        pad_id: Padding token ID.
        max_seq_len: Optional cap on sequence length.

    Returns:
        input_ids: [B, L] LongTensor, padded with pad_id.
        attention_mask: [B, L] BoolTensor, True for real tokens.
    """
    if max_seq_len is not None:
        max_len = min(max(len(s) for s in token_id_lists), max_seq_len)
    else:
        max_len = max(len(s) for s in token_id_lists)
    B = len(token_id_lists)
    input_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros(B, max_len, dtype=torch.bool)
    for i, tids in enumerate(token_id_lists):
        seq_len = min(len(tids), max_len)
        input_ids[i, :seq_len] = torch.tensor(tids[:seq_len])
        attention_mask[i, :seq_len] = True
    return input_ids, attention_mask


def pad_tensor_batch(tensors, pad_value=0):
    """Pad a list of 1-D tensors to the same length.

    Args:
        tensors: List[Tensor] of shape [Li].
        pad_value: Fill value for padding positions.

    Returns:
        Tensor of shape [B, max_L] with the same dtype as the input tensors.
    """
    max_len = max(t.size(0) for t in tensors)
    B = len(tensors)
    out = torch.full((B, max_len), pad_value, dtype=tensors[0].dtype)
    for i, t in enumerate(tensors):
        L = t.size(0)
        out[i, :L] = t
    return out


# ── EMA ───────────────────────────────────────────────────────────────────────

@torch.no_grad()
def update_ema(ema_model, student_model, momentum):
    """Exponential moving average update: ema = momentum * ema + (1-momentum) * student."""
    for ema_p, student_p in zip(ema_model.parameters(), student_model.parameters()):
        ema_p.data.mul_(momentum).add_(student_p.data, alpha=1.0 - momentum)


# ── P×K batch construction ───────────────────────────────────────────────────

def build_pk_batches(epoch_entities_by_class, P, K):
    """Construct P×K batches from per-class entity pools.

    Iterates through available classes, picking P classes per batch and K samples
    per class. When a class has fewer than K remaining items, pads with random
    duplicates from that class's full pool. Exhausted classes are removed.
    Both the class iteration order and the final batch list are shuffled.

    Expects per-class items to already be shuffled by the caller.

    Args:
        epoch_entities_by_class: dict mapping class_id -> list of items.
        P: Number of classes per batch.
        K: Number of samples per class.

    Returns:
        List of batches, each batch a flat list of items.
    """
    class_ids_avail = list(epoch_entities_by_class.keys())
    random.shuffle(class_ids_avail)
    class_cursors = {cid: 0 for cid in class_ids_avail}

    epoch_batches_list = []
    while class_ids_avail:
        batch_items = []
        classes_this_batch = class_ids_avail[:P]
        for cid in classes_this_batch:
            pool = epoch_entities_by_class[cid]
            cursor = class_cursors[cid]
            remaining = len(pool) - cursor
            if remaining >= K:
                batch_items.extend(pool[cursor:cursor + K])
                class_cursors[cid] = cursor + K
            else:
                taken = list(pool[cursor:])
                while len(taken) < K:
                    taken.append(random.choice(pool))
                batch_items.extend(taken)
                class_cursors[cid] = len(pool)
        class_ids_avail = [cid for cid in class_ids_avail
                           if class_cursors[cid] < len(epoch_entities_by_class[cid])]
        random.shuffle(class_ids_avail)
        epoch_batches_list.append(batch_items)

    random.shuffle(epoch_batches_list)
    return epoch_batches_list
