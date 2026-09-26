"""
GAE: Graph Autoencoder for provenance graph pretraining.

Encodes temporal ego-graph neighborhoods with a GAT encoder (reused from
GraphMAE) and reconstructs the adjacency matrix via inner-product decoding.
Trained with BCE loss on positive edges + negative sampling.

The encoder produces the final node embeddings used downstream.

Reference: Kipf & Welling, "Variational Graph Auto-Encoders", NeurIPS 2016 Workshop.
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from pidsmaker.utils.utils import log
from .graphmae import GraphMAEEncoder


class GAE(nn.Module):
    """Graph Autoencoder with inner-product decoder.

    Architecture:
        1. Encode node features with a multi-layer GAT encoder
        2. Decode adjacency via inner product: A_hat = sigmoid(Z @ Z^T)
        3. Train with BCE loss on positive edges + negative samples
    """

    def __init__(self, in_dim, hidden_dim, num_encoder_layers=2, num_heads=4, dropout=0.2):
        super().__init__()
        self.encoder = GraphMAEEncoder(
            in_dim, hidden_dim, num_layers=num_encoder_layers,
            num_heads=num_heads, dropout=dropout,
        )

    def forward(self, x, edge_index):
        """Encode nodes and compute adjacency reconstruction loss.

        Args:
            x: [num_nodes, in_dim] node features
            edge_index: [2, num_edges] edge indices

        Returns:
            loss: BCE reconstruction loss on edges + negative samples
        """
        z = self.encoder(x, edge_index)

        # Positive edges: inner product of connected node pairs
        src, dst = edge_index[0], edge_index[1]
        pos_score = (z[src] * z[dst]).sum(dim=-1)

        # Negative sampling: same number as positive edges
        num_nodes = z.size(0)
        num_neg = src.size(0)
        neg_src = src  # Keep same source nodes
        neg_dst = torch.randint(0, num_nodes, (num_neg,), device=z.device)
        neg_score = (z[neg_src] * z[neg_dst]).sum(dim=-1)

        # BCE loss
        pos_loss = F.binary_cross_entropy_with_logits(
            pos_score, torch.ones_like(pos_score),
        )
        neg_loss = F.binary_cross_entropy_with_logits(
            neg_score, torch.zeros_like(neg_score),
        )
        loss = (pos_loss + neg_loss) / 2

        return loss

    @torch.no_grad()
    def encode(self, x, edge_index):
        """Encode node features without loss computation (for inference)."""
        return self.encoder(x, edge_index)


# ── Save / Load ───────────────────────────────────────────────────────────

def save_gae(model, t5_encoder, out_dir, name="gae.pt"):
    """Save GAE model and T5 node encoder."""
    path = os.path.join(out_dir, name)
    torch.save({
        "model_state_dict": model.state_dict(),
        "t5_encoder_state_dict": t5_encoder.state_dict(),
    }, path)
    log(f"GAE model saved -> {path}")
    return path


def load_gae(out_dir, in_dim, hidden_dim, num_encoder_layers, num_heads,
             vocab_size, model_size, max_seq_len, name="gae.pt"):
    """Load a previously saved GAE model + T5 node encoder."""
    from .graphmae import T5NodeEncoder
    from .gnn_distill import get_gnn_distill_encoder_config

    path = os.path.join(out_dir, name)
    checkpoint = torch.load(path, weights_only=False, map_location="cpu")

    model = GAE(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        num_encoder_layers=num_encoder_layers,
        num_heads=num_heads,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    t5_config = get_gnn_distill_encoder_config(vocab_size, model_size, max_seq_len)
    t5_encoder = T5NodeEncoder(t5_config)
    t5_encoder.load_state_dict(checkpoint["t5_encoder_state_dict"])

    log(f"GAE model loaded <- {path} (T5 d_model={t5_config.d_model}, hidden={hidden_dim})")
    return model, t5_encoder
