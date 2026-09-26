"""
GraphMAE: Masked Graph Autoencoder for provenance graph pretraining.

Encodes temporal ego-graph neighborhoods with a GAT encoder, masks a fraction
of node features, and reconstructs them via a lightweight GAT decoder.
Uses Scaled Cosine Error (SCE) loss instead of MSE for better training
stability on normalized embeddings.

Reference: Hou et al., "GraphMAE: Self-Supervised Masked Graph Autoencoders", KDD 2022.
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from transformers import T5EncoderModel

from pidsmaker.utils.utils import log
from ..training_utils import pad_token_id_lists


# ── T5-based inductive node encoder ──────────────────────────────────────────

class T5NodeEncoder(nn.Module):
    """Encodes (node_type, label) pairs via ProvenanceBPE tokenizer + T5 encoder.

    Replaces the old transductive nn.Embedding lookup with an inductive encoder
    that can handle unseen labels at inference time.
    """

    def __init__(self, config):
        super().__init__()
        self.encoder = T5EncoderModel(config)
        self.d_model = config.d_model

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, input_ids, attention_mask):
        """Encode token IDs and mean-pool over non-padding positions.

        Args:
            input_ids: [B, L] token IDs
            attention_mask: [B, L] bool mask (True = real token)

        Returns:
            [B, d_model] mean-pooled embeddings
        """
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask_f = attention_mask.unsqueeze(-1).float()
        return (hidden * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)

    @torch.no_grad()
    def encode_labels(self, node_labels, tokenizer, device="cpu", batch_size=2048):
        """Tokenize and encode a list of (ntype, nlabel) pairs.

        Args:
            node_labels: list of (ntype, nlabel) tuples
            tokenizer: ProvenanceBPE tokenizer instance
            device: target device
            batch_size: number of labels per forward pass (avoids OOM on large datasets)

        Returns:
            [N, d_model] tensor of node embeddings
        """
        token_id_lists = []
        for ntype, nlabel in node_labels:
            tids = tokenizer.tokenize_node(ntype, nlabel)
            token_id_lists.append(tids if tids else [tokenizer.pad_id])

        all_embs = []
        for i in range(0, len(token_id_lists), batch_size):
            chunk = token_id_lists[i:i + batch_size]
            input_ids, attention_mask = pad_token_id_lists(
                chunk, tokenizer.pad_id, tokenizer.max_seq_len,
            )
            all_embs.append(self.forward(input_ids.to(device), attention_mask.to(device)))
        return torch.cat(all_embs, dim=0)


# ── GraphMAE model ─────────────────────────────────────────────────────────

class GraphMAEEncoder(nn.Module):
    """Multi-layer GAT encoder."""

    def __init__(self, in_dim, hidden_dim, num_layers=2, num_heads=4, dropout=0.2):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        # First layer
        self.layers.append(GATConv(in_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout))
        self.norms.append(nn.LayerNorm(hidden_dim))

        # Subsequent layers
        for _ in range(num_layers - 1):
            self.layers.append(GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout))
            self.norms.append(nn.LayerNorm(hidden_dim))

    def forward(self, x, edge_index):
        for conv, norm in zip(self.layers, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.elu(x)
        return x


class GraphMAEDecoder(nn.Module):
    """Lightweight GAT decoder for feature reconstruction."""

    def __init__(self, hidden_dim, out_dim, num_layers=1, num_heads=4, dropout=0.2):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            is_last = (i == num_layers - 1)
            out = out_dim if is_last else hidden_dim
            heads = 1 if is_last else num_heads
            self.layers.append(GATConv(hidden_dim, out // heads, heads=heads, dropout=dropout, concat=not is_last))
            if not is_last:
                self.norms.append(nn.LayerNorm(hidden_dim))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.layers):
            x = conv(x, edge_index)
            if i < len(self.norms):
                x = self.norms[i](x)
                x = F.elu(x)
        return x


class GraphMAE(nn.Module):
    """Masked Graph Autoencoder.

    Architecture:
        1. Encode node features with a multi-layer GAT encoder
        2. Mask a fraction of node features (replace with [MASK] token or random)
        3. Decode with a lightweight GAT decoder
        4. Reconstruct masked node features using SCE loss

    The encoder produces the final node embeddings used downstream.
    """

    def __init__(
        self,
        in_dim,
        hidden_dim,
        num_encoder_layers=2,
        num_decoder_layers=1,
        num_heads=4,
        mask_rate=0.5,
        replace_rate=0.1,
        dropout=0.2,
    ):
        super().__init__()
        self.mask_rate = mask_rate
        self.replace_rate = replace_rate

        # Encoder
        self.encoder = GraphMAEEncoder(
            in_dim, hidden_dim, num_layers=num_encoder_layers,
            num_heads=num_heads, dropout=dropout,
        )

        # Decoder (reconstructs back to in_dim)
        self.decoder = GraphMAEDecoder(
            hidden_dim, in_dim, num_layers=num_decoder_layers,
            num_heads=num_heads, dropout=dropout,
        )

        # Encoder-to-decoder projection
        self.enc_to_dec = nn.Linear(hidden_dim, hidden_dim)

        # Learnable [MASK] token
        self.mask_token = nn.Parameter(torch.zeros(1, in_dim))
        nn.init.xavier_uniform_(self.mask_token)

    def _apply_mask(self, x):
        """Mask node features: some get [MASK] token, some get random replacement.

        Returns:
            masked_x: node features with masking applied
            mask_indices: indices of masked nodes (for loss computation)
        """
        num_nodes = x.size(0)
        num_mask = max(1, int(num_nodes * self.mask_rate))

        perm = torch.randperm(num_nodes, device=x.device)
        mask_indices = perm[:num_mask]

        # Split masked nodes into [MASK] token vs random replacement
        num_replace = max(0, int(num_mask * self.replace_rate))
        replace_indices = mask_indices[:num_replace]
        token_indices = mask_indices[num_replace:]

        masked_x = x.clone()
        if len(token_indices) > 0:
            masked_x[token_indices] = self.mask_token
        if len(replace_indices) > 0:
            random_indices = torch.randint(0, num_nodes, (len(replace_indices),), device=x.device)
            masked_x[replace_indices] = x[random_indices]

        return masked_x, mask_indices

    def forward(self, x, edge_index):
        """Forward pass with masking for pretraining.

        Returns:
            loss: SCE reconstruction loss on masked nodes
        """
        # Apply masking
        masked_x, mask_indices = self._apply_mask(x)

        # Encode
        encoded = self.encoder(masked_x, edge_index)

        # Project encoder output for decoder
        dec_input = self.enc_to_dec(encoded)

        # Decode
        reconstructed = self.decoder(dec_input, edge_index)

        # SCE loss on masked nodes only
        x_target = x[mask_indices]
        x_pred = reconstructed[mask_indices]
        loss = sce_loss(x_pred, x_target)

        return loss

    @torch.no_grad()
    def encode(self, x, edge_index):
        """Encode node features without masking (for inference)."""
        return self.encoder(x, edge_index)


def sce_loss(pred, target, alpha=2.0):
    """Scaled Cosine Error loss (from GraphMAE paper).

    More robust than MSE for normalized feature reconstruction.
    """
    pred = F.normalize(pred, p=2, dim=-1)
    target = F.normalize(target, p=2, dim=-1)
    cos_sim = (pred * target).sum(dim=-1)
    loss = (1 - cos_sim).pow(alpha)
    return loss.mean()


# ── Neighborhood → PyG graph conversion ───────────────────────────────────

def neighborhoods_to_pyg_batch(neighborhoods, indexid2msg, t5_encoder, tokenizer, label_cache, device="cpu"):
    """Convert a batch of temporal neighborhoods to a single batched PyG graph.

    Each neighborhood is (center_node, ref_time, [(neighbor, edge_type, time), ...]).
    Node features are produced by the T5 encoder via BPE tokenization (inductive).
    A label_cache dict is used to avoid re-encoding labels seen earlier in the epoch.

    Returns:
        x: [total_nodes, d_model] node features from T5 encoder
        edge_index: [2, total_edges] (bidirectional center↔neighbor edges)
        batch: [total_nodes] graph membership index
        node_ids: [total_nodes] original node IDs (for embedding extraction)
    """
    all_node_labels = []  # (ntype, nlabel) per node
    all_edge_src = []
    all_edge_dst = []
    all_batch = []
    all_node_ids = []
    node_offset = 0

    for graph_idx, (center_node, ref_time, neighbors) in enumerate(neighborhoods):
        # Collect unique nodes in this subgraph
        local_nodes = [center_node] + [n[0] for n in neighbors]
        # Deduplicate while preserving order (center is always index 0)
        seen = set()
        unique_nodes = []
        for n in local_nodes:
            if n not in seen:
                seen.add(n)
                unique_nodes.append(n)

        local_id = {n: i + node_offset for i, n in enumerate(unique_nodes)}
        n_nodes = len(unique_nodes)

        # Collect node labels for T5 encoding
        for n in unique_nodes:
            if n in indexid2msg:
                lbl = indexid2msg[n]
                lbl = tuple(lbl) if isinstance(lbl, list) else lbl
            else:
                lbl = ("subject", "unknown")
            all_node_labels.append(lbl)
            all_batch.append(graph_idx)
            all_node_ids.append(n)

        # Edges: center ↔ each neighbor (bidirectional for GNN message passing)
        center_local = local_id[center_node]
        for neighbor, edge_type, t in neighbors:
            if neighbor in local_id:
                nb_local = local_id[neighbor]
                # backward: neighbor → center
                all_edge_src.append(nb_local)
                all_edge_dst.append(center_local)
                # reverse direction for message passing
                all_edge_src.append(center_local)
                all_edge_dst.append(nb_local)

        node_offset += n_nodes

    if not all_node_labels:
        return None, None, None, None

    # Encode node labels via T5 — use cache for labels already encoded
    uncached = [lbl for lbl in set(all_node_labels) if lbl not in label_cache]
    if uncached:
        with torch.no_grad():
            new_embs = t5_encoder.encode_labels(uncached, tokenizer, device)
        for i, lbl in enumerate(uncached):
            label_cache[lbl] = new_embs[i]

    x = torch.stack([label_cache[lbl] for lbl in all_node_labels])  # [N, d_model]

    edge_index = torch.tensor([all_edge_src, all_edge_dst], dtype=torch.long, device=device)
    batch = torch.tensor(all_batch, dtype=torch.long, device=device)

    return x, edge_index, batch, all_node_ids


# ── Save / Load ───────────────────────────────────────────────────────────

def save_graphmae(model, t5_encoder, out_dir, name="graphmae.pt"):
    """Save GraphMAE model and T5 node encoder."""
    path = os.path.join(out_dir, name)
    torch.save({
        "model_state_dict": model.state_dict(),
        "t5_encoder_state_dict": t5_encoder.state_dict(),
    }, path)
    log(f"GraphMAE model saved -> {path}")
    return path


def load_graphmae(out_dir, in_dim, hidden_dim, num_encoder_layers, num_decoder_layers,
                  num_heads, mask_rate, replace_rate, vocab_size, model_size, max_seq_len,
                  name="graphmae.pt"):
    """Load a previously saved GraphMAE model + T5 node encoder."""
    from .gnn_distill import get_gnn_distill_encoder_config

    path = os.path.join(out_dir, name)
    checkpoint = torch.load(path, weights_only=False, map_location="cpu")

    model = GraphMAE(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        num_heads=num_heads,
        mask_rate=mask_rate,
        replace_rate=replace_rate,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    t5_config = get_gnn_distill_encoder_config(vocab_size, model_size, max_seq_len)
    t5_encoder = T5NodeEncoder(t5_config)
    t5_encoder.load_state_dict(checkpoint["t5_encoder_state_dict"])

    log(f"GraphMAE model loaded <- {path} (T5 d_model={t5_config.d_model}, hidden={hidden_dim})")
    return model, t5_encoder
