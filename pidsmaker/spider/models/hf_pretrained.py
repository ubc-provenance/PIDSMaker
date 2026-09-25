"""HuggingFace pretrained model support for GPT-2, Llama 3.2, and OPT.

Loads pretrained weights from HuggingFace, fine-tunes on provenance walk
corpus with causal LM objective, and produces frozen embeddings via
mean-pooling of hidden states.
"""

import os

import numpy as np
import torch

from pidsmaker.utils.utils import log


# ── Model registries ─────────────────────────────────────────────────────

_MODEL_REGISTRY = {
    # NOTE: first entry per model type is the default when model_size is not
    # applicable (e.g. "mini" from BERT configs).
    "gpt2_pretrained": {
        "xl":     ("openai-community/gpt2-xl",      1600),  # 1.5B (default)
        "small":  ("openai-community/gpt2",          768),  # 124M
        "medium": ("openai-community/gpt2-medium",  1024),  # 355M
        "large":  ("openai-community/gpt2-large",   1280),  # 774M
    },
    "llama3_pretrained": {
        "1b": ("meta-llama/Llama-3.2-1B", 2048),  # 1.24B (default)
        "3b": ("meta-llama/Llama-3.2-3B", 3072),  # 3.21B
    },
    "opt_pretrained": {
        "1.3b": ("facebook/opt-1.3b", 2048),  # 1.3B
    },
}

HF_MODEL_TYPES = tuple(_MODEL_REGISTRY.keys())


def _resolve_hf_size(model_type, model_size):
    """Resolve model_size to a valid HF registry key.

    If model_size is already a valid key (e.g. "small", "1b"), use it directly.
    Otherwise fall back to the first (smallest) registered variant and log a
    warning so the user knows which model was selected.
    """
    registry = _MODEL_REGISTRY.get(model_type)
    if registry is None:
        raise ValueError(f"Unknown HF model type: {model_type}")
    if model_size in registry:
        return model_size
    default = next(iter(registry))
    log(f"model_size '{model_size}' not applicable for {model_type}, "
        f"using default '{default}'")
    return default


def get_hf_model_name(model_type, model_size):
    """Return the HuggingFace model ID for the given type and size."""
    size = _resolve_hf_size(model_type, model_size)
    return _MODEL_REGISTRY[model_type][size][0]


def get_hf_hidden_size(model_type, model_size):
    """Return the hidden dimension for the given type and size."""
    size = _resolve_hf_size(model_type, model_size)
    return _MODEL_REGISTRY[model_type][size][1]


# ── Text conversion ──────────────────────────────────────────────────────

def walk_to_text(walk_nodes, walk_edge_types, indexid2msg):
    """Convert a provenance walk to a plain-text string.

    Format: ``subject: bash execute file: /usr/lib/libcrypto.so read ...``
    Using natural-language-style separators so that the pretrained tokenizer
    produces meaningful subwords.
    """
    parts = []
    for i, node_id in enumerate(walk_nodes):
        if node_id not in indexid2msg:
            continue
        ntype, nlabel = indexid2msg[node_id]
        parts.append(f"{ntype}: {nlabel}")
        if i < len(walk_edge_types):
            parts.append(walk_edge_types[i])
    return " ".join(parts)


def node_to_text(ntype, nlabel):
    """Convert a single (ntype, label) pair to text."""
    return f"{ntype}: {nlabel}"


# ── Tokenization helpers ─────────────────────────────────────────────────

def pretokenize_walks_hf(walk_corpus, hf_tokenizer, max_seq_len=512):
    """Tokenize walk corpus with an HF tokenizer.

    Returns:
        list of token-ID lists (deduplicated, truncated to *max_seq_len*).
    """
    pretokenized = []
    seen = set()

    for walk_nodes, walk_edge_types, ds_indexid2msg, _epos in walk_corpus:
        text = walk_to_text(walk_nodes, walk_edge_types, ds_indexid2msg)
        if not text:
            continue

        token_ids = hf_tokenizer.encode(
            text, truncation=True, max_length=max_seq_len,
        )
        sig = tuple(token_ids)
        if sig not in seen:
            seen.add(sig)
            pretokenized.append(token_ids)

    return pretokenized


def pretokenize_labels_hf(indexid2msg, hf_tokenizer, max_seq_len=512):
    """Tokenize deduplicated entity labels with an HF tokenizer.

    Instead of walk sequences, this tokenizes the unique (ntype, nlabel) pairs
    so the model is fine-tuned directly on entity labels for provenance
    embedding quality.

    Returns:
        list of token-ID lists (deduplicated, truncated to *max_seq_len*).
    """
    pretokenized = []
    seen_labels = set()
    seen_tokens = set()

    for ntype, nlabel in indexid2msg.values():
        label_key = (ntype, nlabel)
        if label_key in seen_labels:
            continue
        seen_labels.add(label_key)

        text = node_to_text(ntype, nlabel)
        if not text:
            continue

        token_ids = hf_tokenizer.encode(
            text, truncation=True, max_length=max_seq_len,
        )
        sig = tuple(token_ids)
        if sig not in seen_tokens:
            seen_tokens.add(sig)
            pretokenized.append(token_ids)

    return pretokenized


# ── Inference ─────────────────────────────────────────────────────────────

def embed_nodes_hf(model, hf_tokenizer, indexid2msg, emb_dim, device,
                   batch_size=256, max_seq_len=512):
    """Compute frozen embeddings for every node via the fine-tuned HF model.

    For each unique (ntype, nlabel) pair: tokenize → forward → mean-pool
    last hidden layer → L2-normalise.

    Returns:
        dict mapping ``int(node_id) → np.ndarray`` of shape ``(emb_dim,)``.
    """
    model.eval()

    # Deduplicate by (ntype, nlabel) — values may be lists, so convert to tuples
    unique_keys = list({tuple(v): None for v in indexid2msg.values()}.keys())
    key2emb = {}

    for batch_start in range(0, len(unique_keys), batch_size):
        batch_keys = unique_keys[batch_start:batch_start + batch_size]
        texts = [node_to_text(ntype, nlabel) for ntype, nlabel in batch_keys]

        encoded = hf_tokenizer(
            texts, padding=True, truncation=True,
            max_length=max_seq_len, return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(
                input_ids, attention_mask=attention_mask,
                output_hidden_states=True,
            )
            hidden = outputs.hidden_states[-1]  # [B, L, H]

        # Mean-pool over non-padding positions
        mask_f = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
        pooled_np = pooled.float().cpu().numpy()

        for i, key in enumerate(batch_keys):
            emb = pooled_np[i]
            norm = np.linalg.norm(emb)
            if norm > 1e-12:
                emb = emb / norm
            key2emb[key] = emb.astype(np.float32)

    # Map node_id → embedding
    indexid2vec = {}
    zero = np.zeros(emb_dim, dtype=np.float32)
    for node_id, val in indexid2msg.items():
        key = tuple(val)
        indexid2vec[int(node_id)] = key2emb.get(key, zero).copy()

    return indexid2vec
