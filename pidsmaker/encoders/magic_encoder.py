"""MAGIC system GAT encoder with masked graph representation learning.

MAGIC-specific GAT variant supporting both encoding and decoding roles for
masked feature reconstruction and structure prediction objectives.
"""

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
                concat=self.concat_out,
                dropout=attn_drop,
                negative_slope=negative_slope,
            )
        )

        # Hidden layers
        for _ in range(1, n_layers - 1):
            self.gats.append(
                GATConv(
                    hid_dim * n_heads,
                    hid_dim,
                    heads=n_heads,
                    concat=self.concat_out,
                    dropout=attn_drop,
                    negative_slope=negative_slope,
                )
            )

        # Last layer
        self.gats.append(
            GATConv(
                hid_dim * n_heads,
                out_dim,
                heads=1,
                concat=self.concat_out,
                dropout=attn_drop,
                negative_slope=negative_slope,
            )
        )

        self.dropout = nn.Dropout(feat_drop)
        self.activation = activation
        self.residual = residual
        self.last_linear = nn.Linear(hid_dim * n_heads, out_dim)

    def forward(self, x, edge_index, **kwargs):
        hidden_list = []
        h = x

        # Forward through GAT layers
        for layer in range(self.n_layers):
            h_in = h  # For residual connection
            h = self.dropout(h)
            h = self.gats[layer](h, edge_index)

            if self.residual and layer > 0:  # Adding residual connection if enabled
                if layer == self.n_layers - 1:
                    h_in = self.last_linear(h_in)
                h = h + h_in

            if self.activation:
                h = self.activation(h)

            hidden_list.append(h)

        return h if self.is_decoder else {"h": h}

    def reset_classifier(self, num_classes):
        self.head = nn.Linear(self.out_dim, num_classes)
