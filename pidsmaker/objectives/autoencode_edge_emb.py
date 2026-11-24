"""Edge embedding autoencoder of LARES (Bilot et al., ACSAC 2026).

Each edge gets an embedding computed from the embeddings of its endpoints and, through a gated
skip connection, from its own features:

    g = sigmoid(Lin([h_src, h_dst, e]))
    z = MLP([h_src, h_dst, g * e])

and an autoencoder (out -> out/2 -> out, with a sigmoid output) reconstructs z. The model is
trained to minimize the mean squared reconstruction error, and edges are scored by the sum of
their squared errors: edges whose embedding is unusual are badly reconstructed.
"""

import torch
import torch.nn as nn


class EdgeEmbAutoencoder(nn.Module):
    def __init__(self, node_dim, hid_dim, edge_dim=0, use_edge_feats=True):
        super().__init__()
        self.edge_dim = edge_dim if use_edge_feats else 0
        in_dim = 2 * node_dim + self.edge_dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(in_dim, hid_dim), nn.ReLU(), nn.Linear(hid_dim, node_dim)
        )
        self.gate = (
            nn.Sequential(nn.Linear(in_dim, self.edge_dim), nn.Sigmoid()) if self.edge_dim else None
        )
        self.autoencoder = nn.Sequential(
            nn.Linear(node_dim, max(1, node_dim // 2)),
            nn.ReLU(),
            nn.Linear(max(1, node_dim // 2), node_dim),
            nn.Sigmoid(),
        )

    def forward(self, h_src, h_dst, inference, batch=None, **kwargs):
        x = torch.cat([h_src, h_dst], dim=-1)
        if self.gate is not None:
            e = batch.edge_feats
            x = torch.cat([x, self.gate(torch.cat([x, e], dim=-1)) * e], dim=-1)
        z = self.edge_mlp(x)
        err = (self.autoencoder(z) - z) ** 2
        return {"loss": err.sum(dim=-1) if inference else err.mean()}
