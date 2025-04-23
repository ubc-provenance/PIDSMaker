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
        flow="source_to_target",
    ):
        super().__init__()

        conv2_in_dim = hid_dim * num_heads if concat else hid_dim
        self.conv = TransformerConv(
            in_dim,
            hid_dim,
            heads=num_heads,
            dropout=dropout,
            edge_dim=edge_dim,
            concat=concat,
            flow=flow,
        )
        self.conv2 = TransformerConv(
            conv2_in_dim,
            out_dim,
            heads=1,
            concat=False,
            dropout=dropout,
            edge_dim=edge_dim,
            flow=flow,
        )
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, x, edge_index, edge_feats=None, **kwargs):
        x = self.activation(self.conv(x, edge_index, edge_feats))
        x = self.dropout(x)
        x = self.conv2(x, edge_index, edge_feats)
        return {"h": x}
