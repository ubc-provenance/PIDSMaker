import torch.nn as nn
from torch_geometric.nn import TransformerConv


class GraphAttentionEmbedding(nn.Module):
    def __init__(
        self,
        in_dim,
        hid_dim,
        out_dim,
        edge_dim,
        dropout,
        activation,
        num_heads,
        concat,
        num_layers,
        flow="source_to_target",
    ):
        super().__init__()
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

        self.convs = nn.ModuleList()

        current_dim = in_dim
        for _ in range(num_layers - 1):
            out_channels = hid_dim
            conv = TransformerConv(
                current_dim,
                out_channels,
                heads=num_heads,
                dropout=dropout,
                edge_dim=edge_dim,
                concat=concat,
                flow=flow,
            )
            self.convs.append(conv)
            current_dim = hid_dim * num_heads if concat else hid_dim

        self.convs.append(
            TransformerConv(
                current_dim,
                out_dim,
                heads=1,
                concat=False,
                dropout=dropout,
                edge_dim=edge_dim,
                flow=flow,
            )
        )

    def forward(self, x, edge_index, edge_feats=None, **kwargs):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index, edge_feats)
            x = self.activation(x)
            x = self.dropout(x)

        x = self.convs[-1](x, edge_index, edge_feats)
        return {"h": x}
