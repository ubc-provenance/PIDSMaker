"""GCN encoder with edge weights, as used by Euler and Argus.

Edge weights are read from `batch.edge_weight` when available (e.g. computed by the
`merge_edges` transformation), and default to 1 otherwise.
"""

import torch.nn as nn
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, activation, dropout, num_layers):
        super().__init__()
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        dims = [in_dim] + [hid_dim] * (num_layers - 1) + [out_dim]
        self.convs = nn.ModuleList(
            [GCNConv(dims[i], dims[i + 1], add_self_loops=True) for i in range(num_layers)]
        )

    def forward(self, x, edge_index, batch=None, **kwargs):
        edge_weight = getattr(batch, "edge_weight", None) if batch is not None else None
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_weight=edge_weight)
            if i < len(self.convs) - 1:
                x = self.dropout(self.activation(x))
        return {"h": x}
