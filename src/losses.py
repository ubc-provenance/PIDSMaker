import torch.nn.functional as F


def sce_loss(x, y, alpha=3, inference=False):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    if not inference:
        loss = loss.mean()
    return loss
