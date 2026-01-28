"""Graph Attention Network (GAT) encoder.

GAT uses attention mechanisms to weight neighbor contributions dynamically during
aggregation. Multi-head attention provides different representation subspaces.
"""

import torch.nn as nn
from torch_geometric.nn import GATConv


class GAT(nn.Module):
    """GAT encoder with multi-head attention layers.

    Applies attention-based aggregation where each node attends to its neighbors
    with learned attention weights. Supports multi-head attention for richer representations.
    """
    def __init__(
        self, in_dim, hid_dim, out_dim, activation, dropout, num_heads, concat, num_layers
    ):
        super().__init__()
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

        self.convs = nn.ModuleList()

        current_dim = in_dim
        for _ in range(num_layers - 1):
            out_channels = hid_dim
            conv = GATConv(
                current_dim,
                out_channels,
                heads=num_heads,
                dropout=dropout,
                concat=concat,
            )
            self.convs.append(conv)
            current_dim = hid_dim * num_heads if concat else hid_dim

        self.convs.append(
            GATConv(
                current_dim,
                out_dim,
                heads=1,
                concat=False,
                dropout=dropout,
            )
        )

    def forward(self, x, edge_index, **kwargs):
        """Forward pass through GAT layers.

        Args:
            x: Node features (N, in_dim)
            edge_index: Edge indices (2, E)
            **kwargs: Additional arguments (unused)

        Returns:
            dict: {'h': node embeddings (N, out_dim)}
        """
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = self.activation(x)
            x = self.dropout(x)

        x = self.convs[-1](x, edge_index)
        return {"h": x}
