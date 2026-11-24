"""Encoder of LARES (Bilot et al., ACSAC 2026): "LARES: Host-centered Lateral Movement Detection
via Inductive Graph Reasoning".

Hosts are embedded from their identity: as the one-hot node IDs followed by a linear layer of
LARES, a learned embedding of each node ID, initialized like a linear layer on one-hot vectors
(uniform in ±1/sqrt(N), so that hosts rarely seen during training stay close to the bias),
followed by dropout. A single sum-aggregation layer then combines the embedding of each host
with messages from its neighbors and the features of the edges:

    h_v' = tanh(Lin_out(h_v + sum_{u ~ v} tanh(Lin_node(w_uv * h_u) + Lin_edge(e_uv))))

where w_uv is the edge weight (`batch.edge_weight`, see the `merge_edges` transformation) and
e_uv the edge features (`batch.edge_feats`, e.g. `edge_attrs`). With `flow=target_to_source`
(LARES' default), a host aggregates the messages of the destinations of its outgoing edges;
with `source_to_target`, of the sources of its incoming edges. The output tanh keeps the
embeddings bounded whatever the degree of the hosts.

Node features (`x`) are only used with `use_node_features` (LARES uses the node IDs only).
"""

import math

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


class LaresEncoder(MessagePassing):
    def __init__(
        self,
        in_dim,
        out_dim,
        num_nodes,
        edge_dim=None,
        dropout=0.25,
        flow="target_to_source",
        use_node_features=False,
    ):
        super().__init__(aggr="add", flow=flow)
        bound = 1 / math.sqrt(num_nodes)
        self.id_embedding = nn.Embedding(num_nodes, out_dim)
        nn.init.uniform_(self.id_embedding.weight, -bound, bound)
        self.id_bias = nn.Parameter(torch.empty(out_dim).uniform_(-bound, bound))
        self.lin_x = nn.Linear(in_dim, out_dim) if use_node_features else None
        self.dropout = nn.Dropout(dropout)
        self.lin_node = nn.Linear(out_dim, out_dim)
        self.lin_edge = nn.Linear(edge_dim, out_dim, bias=False) if edge_dim else None
        self.lin_out = nn.Linear(out_dim, out_dim)

    def forward(self, x, edge_index, original_n_id, batch=None, edge_feats=None, **kwargs):
        h = self.id_embedding(original_n_id.long()) + self.id_bias
        if self.lin_x is not None:
            h = h + self.lin_x(x)
        h = self.dropout(h)

        edge_weight = getattr(batch, "edge_weight", None) if batch is not None else None
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.shape[1], device=h.device)
        if self.lin_edge is None:
            edge_feats = None
        aggr = self.propagate(edge_index, h=h, edge_weight=edge_weight, edge_feats=edge_feats)
        return {"h": torch.tanh(self.lin_out(h + aggr))}

    def message(self, h_j, edge_weight, edge_feats):
        m = self.lin_node(edge_weight.unsqueeze(-1) * h_j)
        if edge_feats is not None:
            m = m + self.lin_edge(edge_feats)
        return torch.tanh(m)
