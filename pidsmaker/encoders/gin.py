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
        dropout=0.25,
        flow="target_to_source",
    ):
        super(GIN, self).__init__()

        nn1 = nn.Sequential(nn.Linear(in_dim, hid_dim), nn.ReLU(), nn.Linear(hid_dim, hid_dim))
        nn2 = nn.Sequential(nn.Linear(hid_dim, hid_dim), nn.ReLU(), nn.Linear(hid_dim, hid_dim))

        if edge_dim is None:
            self.conv1 = GINConv(nn1)
            self.conv2 = GINConv(nn2)
        else:
            self.conv1 = GINEConv(nn1, edge_dim=edge_dim)
            self.conv2 = GINEConv(nn2, edge_dim=edge_dim)

        self.fc1 = nn.Linear(hid_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, out_dim)

        self.drop = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_feats, **kwargs):
        x = self.conv1(x, edge_index, edge_feats)
        x = torch.tanh(x)
        x = self.drop(x)

        x = self.conv2(x, edge_index, edge_feats)
        x = torch.tanh(x)

        # NOTE: It's worst in inductive setting to use 2 dropouts
        # x = self.drop(x)

        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return {"h": x}
