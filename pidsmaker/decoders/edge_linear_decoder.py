import torch.nn as nn


class EdgeLinearDecoder(nn.Module):
    def __init__(self, in_dim, dropout):
        super().__init__()

        self.lin_src = nn.Linear(in_dim, in_dim)
        self.lin_dst = nn.Linear(in_dim, in_dim)
        self.lin_final = nn.Linear(in_dim, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, h_src, h_dst):
        h = self.lin_src(self.drop(h_src)) + self.lin_dst(self.drop(h_dst))
        h = h.relu()
        return self.lin_final(h)
