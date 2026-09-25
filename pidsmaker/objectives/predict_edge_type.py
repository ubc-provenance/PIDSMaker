import torch.nn as nn

from pidsmaker.utils.utils import compute_class_weights


class EdgeTypePrediction(nn.Module):
    def __init__(self, decoder, loss_fn, balanced_loss, edge_type_dim, multi_edge):
        super(EdgeTypePrediction, self).__init__()
        self.decoder = decoder
        self.loss_fn = loss_fn
        self.balanced_loss = balanced_loss
        self.edge_type_dim = edge_type_dim
        self.multi_edge = multi_edge

    def forward(self, h_src, h_dst, edge_type, inference, **kwargs):
        h = self.decoder(h_src=h_src, h_dst=h_dst)

        class_weights = (
            compute_class_weights(edge_type, num_classes=self.edge_type_dim)
            if self.balanced_loss
            else None
        )

        if not self.multi_edge:
            edge_type = edge_type.argmax(dim=1)
        
        loss = self.loss_fn(h, edge_type, inference=inference, class_weights=class_weights)
        return {"loss": loss}
