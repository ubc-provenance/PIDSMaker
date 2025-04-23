import torch.nn as nn


class EdgeInnerProductDecoder(nn.Module):
    def __init__(self, dropout):
        super().__init__()

        self.drop = nn.Dropout(dropout)

    def forward(self, h_src, h_dst):
        return (self.drop(h_src) * self.drop(h_dst)).sum(dim=1)
