import torch
import torch.nn as nn


class EdgeEmbReconstruction(nn.Module):
    def __init__(self, decoder, loss_fn):
        super(EdgeEmbReconstruction, self).__init__()
        self.decoder = decoder
        self.loss_fn = loss_fn

    def forward(self, h_src, h_dst, inference, **kwargs):
        h_edge_hat = self.decoder(h_src=h_src, h_dst=h_dst)

        h_edge = torch.cat([h_src, h_dst], dim=-1)
        loss = self.loss_fn(h_edge_hat, h_edge, inference=inference)
        return {"loss": loss}
