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
from torch_geometric.utils import scatter


def _edge_conditioned_conv(conv, x, edge_index, edge_attr):
    """Output of the NNConv `conv` (mean aggregation, edge network ending with a linear layer
    `conv.nn[-1]`), without building its (E, in, out) per-edge weight matrices.

    The weights of an edge are W(e) = sum_k h_k(e) A_k + B, with h(e) the input of the last
    linear layer of the edge network (its weight holds the A_k, its bias B). The message of
    an edge (j, i) is then x_j W(e) = sum_k h_k(e) (x_j A_k) + x_j B, where the x A_k and x B
    are computed once per node: (E, hidden, out) values instead of (E, in, out), which do not
    fit in memory for graphs with millions of edges.
    """
    last = conv.nn[-1]
    hidden, in_dim, out_dim = last.in_features, conv.in_channels_l, conv.out_channels
    h = conv.nn[:-1](edge_attr)  # (E, hidden)
    # A[i, o, k] = weight[i * out + o, k]: (in, out * hidden), then (N, out, hidden) per node
    a = last.weight.view(in_dim, out_dim, hidden).reshape(in_dim, out_dim * hidden)
    xa = (x @ a).view(-1, out_dim, hidden)
    xb = x @ last.bias.view(in_dim, out_dim) if last.bias is not None else None
    src, dst = edge_index
    msg = torch.einsum("eok,ek->eo", xa[src], h)
    if xb is not None:
        msg = msg + xb[src]
    out = scatter(msg, dst, dim=0, dim_size=x.size(0), reduce="mean")
    if conv.root_weight:
        out = out + conv.lin(x)
    if conv.bias is not None:
        out = out + conv.bias
    return out


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
            x = _edge_conditioned_conv(self.last, x, edge_index, edge_feats)
        else:
            x = self.last(x, edge_index, edge_weight=edge_weight)
        return {"h": x}
