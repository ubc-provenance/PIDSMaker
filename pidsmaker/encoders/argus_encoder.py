"""Encoder of Argus (Xu et al., IEEE S&P 2024): "Understanding and Bridging the Gap Between
Unsupervised Network Representation Learning and Security Analytics".

GCN layers with edge weights (`batch.edge_weight`, see the `merge_edges` transformation),
followed by an edge-conditioned convolution (NNConv) using the edge features
(`batch.edge_feats`, e.g. `edge_attrs`) when available. Argus runs a GRU over the successive
embeddings of nodes, which is added with the `snapshot_gru` encoder wrapper.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, NNConv


class ArgusEncoder(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, edge_dim, num_layers, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        dims = [in_dim] + [hid_dim] * (num_layers - 1)
        self.gcns = nn.ModuleList([GCNConv(dims[i], dims[i + 1]) for i in range(num_layers - 1)])
        if edge_dim:
            # As Argus, the weights of the convolution are computed by a small MLP of the edge features
            edge_nn = nn.Sequential(
                nn.Linear(edge_dim, 8), nn.ReLU(), nn.Linear(8, dims[-1] * out_dim)
            )
            self.last = NNConv(dims[-1], out_dim, edge_nn, aggr="mean")
        else:
            self.last = GCNConv(dims[-1], out_dim)
        self.use_edge_feats = bool(edge_dim)

    def forward(self, x, edge_index, batch=None, edge_feats=None, **kwargs):
        edge_weight = getattr(batch, "edge_weight", None) if batch is not None else None
        for conv in self.gcns:
            x = self.dropout(torch.relu(conv(x, edge_index, edge_weight=edge_weight)))
        if self.use_edge_feats:
            x = self.last(x, edge_index, edge_feats)
        else:
            x = self.last(x, edge_index, edge_weight=edge_weight)
        return {"h": x}
