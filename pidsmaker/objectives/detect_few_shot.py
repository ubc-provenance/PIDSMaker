import torch
import torch.nn as nn
import torch.nn.functional as F


class FewShotEdgeDetection(nn.Module):
    def __init__(self, decoder, loss_fn):
        super(FewShotEdgeDetection, self).__init__()
        self.decoder = decoder
        self.loss_fn = loss_fn

    def forward(self, h_src, h_dst, y_edge, inference, **kwargs):
        y_edge_hat = self.decoder(h_src=h_src, h_dst=h_dst)

        if inference:
            # To get an anomaly score instead of a loss, we get the complement of the softmax normalized score
            return {"loss": 1 - F.softmax(y_edge_hat, dim=1)[torch.arange(y_edge.shape[0]), y_edge]}

        return {
            "loss": self.loss_fn(y_edge_hat, y_edge, inference=inference)
        }  # weight=torch.tensor([1.0, 20000.0]).to(h_src.device))
