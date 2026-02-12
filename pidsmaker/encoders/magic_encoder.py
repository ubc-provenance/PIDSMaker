"""MAGIC system GAT encoder with masked graph representation learning.

MAGIC-specific GAT variant supporting both encoding and decoding roles for
masked feature reconstruction and structure prediction objectives.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv


class MagicGAT(nn.Module):
    """GAT encoder/decoder for MAGIC masked graph learning.

    Multi-layer GAT with configurable heads, residual connections, and dropout.
    Can function as encoder (for feature extraction) or decoder (for reconstruction).
    """

    def __init__(
        self,
        in_dim,
        hid_dim,
        out_dim,
        n_layers,
        n_heads,
        feat_drop=0.1,
        attn_drop=0.0,
        negative_slope=0.2,
        residual=False,
        activation=None,
        concat_out=False,
        is_decoder=False,
        edge_dim=None,
    ):
        super(MagicGAT, self).__init__()
        self.out_dim = out_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.concat_out = concat_out
        self.gats = nn.ModuleList()
        self.is_decoder = is_decoder

        # First layer
        self.gats.append(
            GATConv(
                in_dim,
                hid_dim,
                heads=n_heads,
                concat=True,  # Old pipeline used True
                dropout=attn_drop,
                negative_slope=negative_slope,
                edge_dim=edge_dim,
                add_self_loops=False,
            )
        )

        # Hidden layers
        for _ in range(1, n_layers - 1):
            self.gats.append(
                GATConv(
                    hid_dim * n_heads,
                    hid_dim,
                    heads=n_heads,
                    concat=True,  # Old pipeline used True
                    dropout=attn_drop,
                    negative_slope=negative_slope,
                    edge_dim=edge_dim,
                    add_self_loops=False,
                )
            )

        # Last layer
        # Old pipeline used n_heads for the last layer too
        self.gats.append(
            GATConv(
                hid_dim * n_heads,
                out_dim,
                heads=n_heads,
                concat=True,
                dropout=attn_drop,
                negative_slope=negative_slope,
                edge_dim=edge_dim,
                add_self_loops=False,
            )
        )

        self.dropout = nn.Dropout(feat_drop)
        self.activation = activation
        self.residual = residual

        # Residual connection projection for the last layer
        # Output of last layer is now out_dim * n_heads due to concat=True and heads=n_heads
        self.last_linear = nn.Linear(hid_dim * n_heads, out_dim * n_heads)

        if not self.is_decoder:
            # MAGIC-style concatenation: Concat all layers + Linear Projection
            # Intermediate layers: hid_dim * n_heads
            # Last layer: out_dim * n_heads
            concat_dim = (n_layers - 1) * (hid_dim * n_heads) + (out_dim * n_heads)
            self.out_proj = nn.Linear(concat_dim, out_dim)

    def forward(self, x, edge_index, edge_attr=None, edge_feats=None, **kwargs):
        if edge_attr is None:
            edge_attr = edge_feats

        hidden_list = []
        h = x

        # Forward through GAT layers
        for layer in range(self.n_layers):
            h_in = h  # For residual connection
            h = self.dropout(h)
            h = self.gats[layer](h, edge_index, edge_attr=edge_attr)

            if self.residual and layer > 0:
                if layer == self.n_layers - 1:
                    h_in = self.last_linear(h_in)
                h = h + h_in

            if self.activation:
                h = self.activation(h)

            hidden_list.append(h)

        if not self.is_decoder:
            # Concatenate all layer outputs
            h = torch.cat(hidden_list, dim=-1)
            h = self.out_proj(h)

        return h if self.is_decoder else {"h": h}

    def reset_classifier(self, num_classes):
        self.head = nn.Linear(self.out_dim, num_classes)
