import torch.nn as nn
from torch_geometric.nn import SAGEConv


class SAGE(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, activation, dropout):
        super(SAGE, self).__init__()
        self.conv1 = SAGEConv(in_dim, hid_dim, normalize=False)
        self.conv2 = SAGEConv(hid_dim, out_dim, normalize=False)
        self.activation = activation
        self.drop = nn.Dropout(dropout)

    def forward(self, x, edge_index, **kwargs):
        x = self.activation(self.conv1(x, edge_index))
        x = self.drop(x)
        x = self.conv2(x, edge_index)
        return {"h": x}
