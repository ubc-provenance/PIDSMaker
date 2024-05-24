import torch
import torch.nn.functional as F


def sce_loss(x, y, alpha=3, inference=False, **kwargs):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    if not inference:
        loss = loss.mean()
    return loss

def mse_loss(x, y, inference=False, **kwargs):
    reduction = "none" if inference else "mean"
    return F.mse_loss(x, y, reduction=reduction)

def bce_contrastive(positive, negative, inference=False, **kwargs):
    reduction = "none" if inference else "mean"
    pos_loss = F.binary_cross_entropy_with_logits(positive, torch.ones_like(positive), reduction=reduction)
    
    if not inference:
        neg_loss = F.binary_cross_entropy_with_logits(negative, torch.zeros_like(negative), reduction=reduction)
        return (pos_loss + neg_loss) * 0.5

    return pos_loss

def cross_entropy(x, y, inference=False, **kwargs):
    reduction = "none" if inference else "mean"
    return F.cross_entropy(x, y, reduction=reduction)
