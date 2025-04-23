import torch.nn as nn
import torch.nn.functional as F

from pidsmaker.utils.utils import compute_class_weights


class NodeTypePrediction(nn.Module):
    def __init__(self, decoder, loss_fn, balanced_loss, node_type_dim):
        super(NodeTypePrediction, self).__init__()
        self.decoder = decoder
        self.loss_fn = loss_fn
        self.balanced_loss = balanced_loss
        self.node_type_dim = node_type_dim

    def forward(self, h, node_type, inference, **kwargs):
        h_hat = self.decoder(h)

        class_weights = (
            compute_class_weights(node_type, num_classes=self.node_type_dim)
            if self.balanced_loss
            else None
        )

        node_type_classes = node_type.argmax(dim=1)
        loss = self.loss_fn(h_hat, node_type_classes, inference=inference, weight=class_weights)
        return {"loss": loss, "out": F.log_softmax(h, dim=1)}
