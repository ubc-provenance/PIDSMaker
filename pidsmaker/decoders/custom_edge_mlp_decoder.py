import torch
import torch.nn as nn

from pidsmaker.encoders.custom_mlp import CustomMLPAsbtract


class CustomEdgeMLP(CustomMLPAsbtract):
    def __init__(self, in_dim, out_dim, architecture, dropout, src_dst_projection_coef):
        super().__init__(in_dim * 2 * src_dst_projection_coef, out_dim, architecture, dropout)

        self.lin_src = nn.Linear(in_dim, in_dim * src_dst_projection_coef)
        self.lin_dst = nn.Linear(in_dim, in_dim * src_dst_projection_coef)

    def forward(self, h_src, h_dst):
        h = torch.cat([self.lin_src(h_src), self.lin_dst(h_dst)], dim=-1)
        h = self.mlp(h)
        return h
