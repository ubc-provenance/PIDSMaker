import torch.nn as nn
import torch.nn.functional as F

from pidsmaker.detection.flash_detection import TRAIN_MASK_KEY
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
        if inference:
            loss = self.loss_fn(h_hat, node_type_classes, inference=True, weight=class_weights)
        else:
            # FLASH trains each round on the nodes it still misclassifies only
            batch = kwargs.get("batch")
            train_mask = getattr(batch, TRAIN_MASK_KEY, None) if batch is not None else None
            if train_mask is None:
                loss = self.loss_fn(h_hat, node_type_classes, inference=False, weight=class_weights)
            else:
                loss = self.loss_fn(
                    h_hat[train_mask], node_type_classes[train_mask], inference=False, weight=class_weights
                )
        return {"loss": loss, "out": F.log_softmax(h_hat, dim=1)}
