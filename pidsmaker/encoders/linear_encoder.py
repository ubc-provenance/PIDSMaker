import torch.nn as nn


class LinearEncoder(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, *args, **kwargs):
        # Handle both tuples and lists (PyG batching may convert tuples to lists)
        if isinstance(x, (tuple, list)):
            h = self.dropout(self.lin1(x[0])), self.dropout(self.lin1(x[1]))
        else:
            h = self.dropout(self.lin1(x))
        return {"h": h}
