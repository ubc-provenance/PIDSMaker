"""
DGI: Deep Graph Infomax for provenance graph pretraining.

Maximizes mutual information between node-level and graph-level representations
by contrasting real node embeddings against corrupted (shuffled) ones using a
bilinear discriminator.

The encoder produces the final node embeddings used downstream.

Reference: Velickovic et al., "Deep Graph Infomax", ICLR 2019.
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from pidsmaker.utils.utils import log
from .graphmae import GraphMAEEncoder


class DGI(nn.Module):
    """Deep Graph Infomax with bilinear discriminator.

    Architecture:
        1. Encode real node features with a multi-layer GAT encoder
        2. Corrupt the graph by shuffling node features
        3. Encode corrupted node features with the same encoder
        4. Readout: mean-pool real node embeddings into a graph-level summary
        5. Discriminator: bilinear scoring of (node, summary) pairs
        6. BCE loss: real nodes → 1, corrupted nodes → 0
    """

    def __init__(self, in_dim, hidden_dim, num_encoder_layers=2, num_heads=4, dropout=0.2):
        super().__init__()
        self.encoder = GraphMAEEncoder(
            in_dim, hidden_dim, num_layers=num_encoder_layers,
            num_heads=num_heads, dropout=dropout,
        )
        self.discriminator = nn.Bilinear(hidden_dim, hidden_dim, 1)

    def forward(self, x, edge_index):
        """Forward pass with corruption for pretraining.

        Args:
            x: [num_nodes, in_dim] node features
            edge_index: [2, num_edges] edge indices

        Returns:
            loss: BCE discrimination loss
        """
        # Positive: encode real graph
        pos_z = self.encoder(x, edge_index)

        # Graph-level summary via mean readout + sigmoid
        summary = torch.sigmoid(pos_z.mean(dim=0))

        # Negative: encode corrupted graph (shuffled node features, same structure)
        perm = torch.randperm(x.size(0), device=x.device)
        neg_z = self.encoder(x[perm], edge_index)

        # Discriminator scores
        summary_expanded = summary.unsqueeze(0).expand_as(pos_z)
        pos_score = self.discriminator(pos_z, summary_expanded).squeeze(-1)
        neg_score = self.discriminator(neg_z, summary_expanded).squeeze(-1)

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
        """Encode node features without corruption (for inference)."""
        return self.encoder(x, edge_index)


# ── Save / Load ───────────────────────────────────────────────────────────

def save_dgi(model, t5_encoder, out_dir, name="dgi.pt"):
    """Save DGI model and T5 node encoder."""
    path = os.path.join(out_dir, name)
    torch.save({
        "model_state_dict": model.state_dict(),
        "t5_encoder_state_dict": t5_encoder.state_dict(),
    }, path)
    log(f"DGI model saved -> {path}")
    return path


def load_dgi(out_dir, in_dim, hidden_dim, num_encoder_layers, num_heads,
             vocab_size, model_size, max_seq_len, name="dgi.pt"):
    """Load a previously saved DGI model + T5 node encoder."""
    from .graphmae import T5NodeEncoder
    from .gnn_distill import get_gnn_distill_encoder_config

    path = os.path.join(out_dir, name)
    checkpoint = torch.load(path, weights_only=False, map_location="cpu")

    model = DGI(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        num_encoder_layers=num_encoder_layers,
        num_heads=num_heads,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    t5_config = get_gnn_distill_encoder_config(vocab_size, model_size, max_seq_len)
    t5_encoder = T5NodeEncoder(t5_config)
    t5_encoder.load_state_dict(checkpoint["t5_encoder_state_dict"])

    log(f"DGI model loaded <- {path} (T5 d_model={t5_config.d_model}, hidden={hidden_dim})")
    return model, t5_encoder
