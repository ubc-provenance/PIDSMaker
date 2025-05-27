import torch.nn as nn
from torch_geometric.nn import SAGEConv


class SAGE(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, activation, dropout, num_layers):
        super().__init__()
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

        self.convs = nn.ModuleList()

        current_dim = in_dim
        for _ in range(num_layers - 1):
            out_channels = hid_dim
            conv = SAGEConv(current_dim, out_channels, normalize=False)
            self.convs.append(conv)
            current_dim = hid_dim

        self.convs.append(SAGEConv(current_dim, out_dim, normalize=False))

    def forward(self, x, edge_index, **kwargs):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = self.activation(x)
            x = self.dropout(x)

        x = self.convs[-1](x, edge_index)
        return {"h": x}
