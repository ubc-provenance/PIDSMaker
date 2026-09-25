import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable


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
    pos_loss = F.binary_cross_entropy_with_logits(
        positive, torch.ones_like(positive), reduction=reduction
    )

    if not inference:
        neg_loss = F.binary_cross_entropy_with_logits(
            negative, torch.zeros_like(negative), reduction=reduction, weight=weight
        )
        return (pos_loss + neg_loss) * 0.5

    return pos_loss


def cross_entropy(x, y, inference=False, weight=None, **kwargs):
    reduction = "none" if inference else "mean"
    return F.cross_entropy(x, y, reduction=reduction, weight=weight)


def binary_cross_entropy(x, y, inference=False, weight=None, **kwargs):
    reduction = "none" if inference else "mean"
    loss = F.binary_cross_entropy_with_logits(x, y, reduction=reduction, weight=weight)
    
    if inference:
        loss = loss.mean(dim=1)
    return loss

# Seems having an error (see GH issue)
# https://github.com/tomastokar/Additive-Margin-Softmax/blob/main/AMSloss.py
class AdMSoftmaxLoss(nn.Module):
    def __init__(self, emb_dim, num_classes, scale, margin):
        super(AdMSoftmaxLoss, self).__init__()
        self.scale = scale
        self.margin = margin
        self.emb_dim = emb_dim
        self.num_classes = num_classes
        self.embedding = nn.Embedding(num_classes, emb_dim, max_norm=1)

    def forward(self, x, labels, inference=False, **kwargs):
        x = F.normalize(x, dim=1)
        w = self.embedding.weight        
        cos_theta = torch.matmul(w, x.T).T
        psi = cos_theta - self.margin
        
        onehot = F.one_hot(labels, self.num_classes)
        logits = self.scale * torch.where(onehot == 1, psi, cos_theta)        
        err = cross_entropy(logits, labels, inference=inference)
        
        return err

# Seems a fixed version of AMS
# https://github.com/yzyouzhang/AIR-ASVspoof/blob/master/loss.py#L37
class AMSoftmax(nn.Module):
    def __init__(self, emb_dim, num_classes, scale, margin):
        super(AMSoftmax, self).__init__()
        self.emb_dim = emb_dim
        self.num_classes = num_classes
        self.scale = scale
        self.margin = margin
        self.centers = nn.Parameter(torch.randn(num_classes, emb_dim))

    def forward(self, feat, label, inference=False, **kwargs):
        batch_size = feat.shape[0]
        norms = torch.norm(feat, p=2, dim=-1, keepdim=True)
        nfeat = torch.div(feat, norms)

        norms_c = torch.norm(self.centers, p=2, dim=-1, keepdim=True)
        ncenters = torch.div(self.centers, norms_c)
        logits = torch.matmul(nfeat, torch.transpose(ncenters, 0, 1))

        y_onehot = torch.FloatTensor(batch_size, self.num_classes)
        y_onehot.zero_()
        y_onehot = Variable(y_onehot).cuda()
        y_onehot.scatter_(1, torch.unsqueeze(label, dim=-1), self.margin)
        margin_logits = self.scale * (logits - y_onehot)
        err = cross_entropy(margin_logits, label, inference=inference)

        return err
