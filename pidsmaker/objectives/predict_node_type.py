import torch.nn as nn
import torch.nn.functional as F

from pidsmaker.utils.utils import compute_class_weights

# Attribute of a batch holding the boolean mask of the nodes that contribute to the training loss
NODE_LOSS_MASK_KEY = "node_loss_mask"


class NodeTypePrediction(nn.Module):
    def __init__(self, decoder, loss_fn, balanced_loss, node_type_dim):
        super(NodeTypePrediction, self).__init__()
        self.decoder = decoder
        self.loss_fn = loss_fn
        self.balanced_loss = balanced_loss
        self.node_type_dim = node_type_dim

    def forward(self, h, node_type, inference, batch=None, **kwargs):
        h_hat = self.decoder(h)

        # During training, the loss can be restricted to some nodes: those that ThreaTrace's
        # previous submodels do not explain yet, or that FLASH still misclassifies
        node_loss_mask = getattr(batch, NODE_LOSS_MASK_KEY, None) if not inference else None
        h_loss, node_type_loss = h_hat, node_type
        if node_loss_mask is not None:
            h_loss, node_type_loss = h_hat[node_loss_mask], node_type[node_loss_mask]

        class_weights = (
            compute_class_weights(node_type_loss, num_classes=self.node_type_dim)
            if self.balanced_loss
            else None
        )

        node_type_classes = node_type_loss.argmax(dim=1)
        loss = self.loss_fn(h_loss, node_type_classes, inference=inference, weight=class_weights)
        return {"loss": loss, "out": F.log_softmax(h_hat, dim=1)}
