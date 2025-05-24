import torch
import torch.nn as nn
from torch_geometric.nn import GINConv, GINEConv


class GIN(nn.Module):
    def __init__(
        self,
        in_dim,
        hid_dim,
        out_dim,
        edge_dim,
        dropout,
        activation,
        num_layers,
        flow="target_to_source",
    ):
        super().__init__()
        self.activation = activation
        self.drop = nn.Dropout(dropout)
        
        self.convs = nn.ModuleList()
        
        current_dim = in_dim
        for i in range(num_layers):
            nn_seq = nn.Sequential(
                nn.Linear(current_dim, hid_dim),
                nn.ReLU(),
                nn.Linear(hid_dim, hid_dim)
            )

            if edge_dim is None:
                conv = GINConv(nn_seq)
            else:
                conv = GINEConv(nn_seq, edge_dim=edge_dim)
            self.convs.append(conv)
            current_dim = hid_dim
        
        self.fc1 = nn.Linear(hid_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, out_dim)

    def forward(self, x, edge_index, edge_feats, **kwargs):
        for conv in self.convs:
            x = conv(x, edge_index, edge_feats)
            x = self.activation(x)
            x = self.drop(x)
        
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return {"h": x}
