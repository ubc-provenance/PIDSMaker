import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


class SumAggregation(MessagePassing):
    def __init__(
        self,
        in_dim,
        hid_dim,
        out_dim,
    ):
        super().__init__(aggr="sum")
        self.lin1 = nn.Linear(in_dim, hid_dim)
        self.lin2 = nn.Linear(hid_dim, out_dim)

    def forward(self, x, edge_index, **kwargs):
        x = torch.tanh(self.lin1(self.propagate(edge_index, x=x)))
        x = self.lin2(x)  # we need weights + tanh if "sum" aggreg is used, or too large gradients
        return {"h": x}
