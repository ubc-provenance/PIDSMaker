import torch.nn as nn


class NodeEmbReconstruction(nn.Module):
    def __init__(self, decoder, loss_fn):
        super(NodeEmbReconstruction, self).__init__()
        self.decoder = decoder
        self.loss_fn = loss_fn

    def forward(self, h, inference, **kwargs):
        h_hat = self.decoder(h)
        loss = self.loss_fn(h_hat, h, inference=inference)
        return {"loss": loss}
