"""Encoder of E-GraphSAGE (Lo et al., NOMS 2022): "E-GraphSAGE: A Graph Neural Network based
Intrusion Detection System for IoT".

GraphSAGE layers whose messages are the concatenation of the neighbor embedding and the features
of the edge (`batch.edge_feats`, e.g. `edge_attrs`), averaged over the edges of each node:

    h_v' = act(W_self h_v + W_neigh mean_{u ~ v} [h_u || e_uv])

With `undirected` (as E-GraphSAGE and Anomal-E, whose graphs are undirected multigraphs), each
edge is aggregated in both directions; otherwise, only over incoming edges.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


class EGraphSAGELayer(MessagePassing):
    def __init__(self, in_dim, out_dim, edge_dim):
        super().__init__(aggr="mean")
        self.lin_neigh = nn.Linear(in_dim + edge_dim, out_dim)
        self.lin_self = nn.Linear(in_dim, out_dim)
        self.edge_dim = edge_dim

    def forward(self, x, edge_index, edge_feats):
        return self.lin_self(x) + self.lin_neigh(self.propagate(edge_index, x=x, edge_feats=edge_feats))

    def message(self, x_j, edge_feats):
        if edge_feats is None:
            return x_j
        return torch.cat([x_j, edge_feats], dim=-1)


class EGraphSAGE(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, edge_dim, activation, dropout, num_layers, undirected=True):
        super().__init__()
        self.undirected = undirected
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        edge_dim = edge_dim or 0
        dims = [in_dim] + [hid_dim] * (num_layers - 1) + [out_dim]
        self.layers = nn.ModuleList(
            [EGraphSAGELayer(dims[i], dims[i + 1], edge_dim) for i in range(num_layers)]
        )
        self.use_edge_feats = edge_dim > 0

    def forward(self, x, edge_index, edge_feats=None, **kwargs):
        edge_feats = edge_feats if self.use_edge_feats else None
        if self.undirected:
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
            edge_feats = torch.cat([edge_feats, edge_feats]) if edge_feats is not None else None
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index, edge_feats)
            if i < len(self.layers) - 1:
                x = self.dropout(self.activation(x))
        return {"h": x}
