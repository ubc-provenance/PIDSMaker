import torch
import torch.nn.functional as F


def sce_loss(x, y, alpha=3, inference=False, **kwargs):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    if not inference:
        loss = loss.mean()
    return loss

def mse_loss(x, y, inference=False, reduction="mean", is_mae=False, **kwargs):
    loss_fn = F.l1_loss if is_mae else F.mse_loss
    if inference:
        losses = loss_fn(x, y, reduction="none")
        return torch.sum(losses, dim=1)
    
    return loss_fn(x, y, reduction=reduction)

def mse_loss_sum(x, y, **kwargs):
    return mse_loss(x, y, reduction="sum", **kwargs)

def mae_loss(x, y, **kwargs):
    return mse_loss(x, y, is_mae=True, **kwargs)

def bce_contrastive(positive, negative, inference=False, weight=None, **kwargs):
    reduction = "none" if inference else "mean"
    pos_loss = F.binary_cross_entropy_with_logits(positive, torch.ones_like(positive), reduction=reduction)
    
    if not inference:
        neg_loss = F.binary_cross_entropy_with_logits(negative, torch.zeros_like(negative), reduction=reduction, weight=weight)
        return (pos_loss + neg_loss) * 0.5

    return pos_loss

def cross_entropy(x, y, inference=False, weight=None, **kwargs):
    reduction = "none" if inference else "mean"
    return F.cross_entropy(x, y, reduction=reduction, weight=weight)

def binary_cross_entropy(x, y, inference=False, weight=None, **kwargs):
    reduction = "none" if inference else "mean"
    return F.binary_cross_entropy_with_logits(x, y, reduction=reduction, weight=weight)
