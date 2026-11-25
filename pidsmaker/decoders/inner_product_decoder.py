import torch.nn as nn


class EdgeInnerProductDecoder(nn.Module):
    """Scores edges by the inner product of the embeddings of their endpoints.

    With `negate`, edges are scored by the negated inner product: in ShadeWatcher, the inner
    product is the probability that two entities would *not* interact, low for observed edges.
    """

    def __init__(self, dropout, negate=False):
        super().__init__()

        self.drop = nn.Dropout(dropout)
        self.sign = -1.0 if negate else 1.0

    def forward(self, h_src, h_dst):
        return self.sign * (self.drop(h_src) * self.drop(h_dst)).sum(dim=1)
