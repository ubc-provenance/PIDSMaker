"""R-Caid GNN encoder (Goyal et al., IEEE S&P 2024).

Each layer embeds a node from two aggregations, concatenated and passed to an MLP update
(Section 4.2 and Figure 2 of the paper): x_i' = U(x_i || A(N_i) || A(R_i)), where

- A(N_i) is a graph attention over the local neighborhood: the sources of the edges of the
  provenance graph towards i (information flows along the direction of the edges);
- A(R_i) is a graph attention, with its own weights, over the pseudo-roots of i: the roots
  of the `rcaid_pseudo_graph` transformation that reach i, through the pseudo edges. A
  pseudo-root keeps the initial features of its root and is never updated, so the messages
  of the pseudo edges are always computed from the input features of the roots.

The pseudo edges are the edges without edge type (all-zero `edge_types` rows). The
transformation keeps the edges of the provenance graph with `keep_original_edges`; without
it (or without edge types), all the edges are used as both local and pseudo edges.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class _RcaidMLP(nn.Module):
    """Update function U of R-Caid: a 2-layer MLP."""

    def __init__(self, input_dim, output_dim):
        super(_RcaidMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.fc2 = nn.Linear(output_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def split_pseudo_edges(edge_index, edge_types):
    """(local edges, pseudo edges): the pseudo edges are those without edge type. Without edge
    types, or if no edge has a type, all the edges are both."""
    if edge_types is None or edge_types.dim() != 2 or edge_types.shape[1] == 0:
        return edge_index, edge_index
    pseudo = edge_types.sum(dim=1) == 0
    if bool(pseudo.all()):
        return edge_index, edge_index
    return edge_index[:, ~pseudo], edge_index[:, pseudo]


class RCaidGAT(nn.Module):
    """R-Caid encoder: `num_layers` layers of local + root attention with an MLP update.

    The paper selects 2 layers among 2, 3 and 4 and does not give the number of heads (1 here).
    """

    def __init__(self, in_dim, hid_dim, out_dim, dropout, num_layers=2, num_heads=1):
        super(RCaidGAT, self).__init__()
        self.local_convs = nn.ModuleList()
        self.root_convs = nn.ModuleList()
        self.updates = nn.ModuleList()
        for layer in range(num_layers):
            d_in = in_dim if layer == 0 else hid_dim
            d_out = out_dim if layer == num_layers - 1 else hid_dim
            self.local_convs.append(GATConv(d_in, hid_dim, heads=num_heads, concat=False))
            # Bipartite attention from the (never updated) pseudo-roots to the nodes
            self.root_convs.append(
                GATConv((in_dim, d_in), hid_dim, heads=num_heads, concat=False, add_self_loops=False)
            )
            self.updates.append(_RcaidMLP(d_in + 2 * hid_dim, d_out))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_types=None, **kwargs):
        local_edges, pseudo_edges = split_pseudo_edges(edge_index, edge_types)
        h = x
        num_layers = len(self.updates)
        for layer in range(num_layers):
            a_local = self.local_convs[layer](h, local_edges)
            a_root = self.root_convs[layer]((x, h), pseudo_edges)
            h = self.updates[layer](torch.cat([h, a_local, a_root], dim=-1))
            if layer < num_layers - 1:
                h = self.dropout(F.relu(h))
        return {"h": h}
