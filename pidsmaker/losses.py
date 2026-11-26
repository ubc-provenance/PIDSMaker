import torch
import torch.nn.functional as F
import torch.nn as nn


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


def ap_contrastive(positive, negative, inference=False, margin=0.8, **kwargs):
    """Surrogate average precision loss of link prediction, as in Argus.

    Each positive edge should be ranked above the negative edges: the precision of a positive
    edge i is approximated by sum_{j in pos} l(s_j, s_i) / sum_{j in all} l(s_j, s_i), with the
    squared hinge l(s_j, s_i) = max(0, margin - (s_i - s_j))^2 on sigmoid scores. This is the
    mini-batch version of the AP loss of LibAUC (without its moving averages across batches).
    At inference, edges are scored like `bce_contrastive` (low probability = anomalous).
    """
    pos_loss = F.binary_cross_entropy_with_logits(
        positive, torch.ones_like(positive), reduction="none"
    )
    if inference:
        return pos_loss

    pos, neg = torch.sigmoid(positive.flatten()), torch.sigmoid(negative.flatten())
    pos_sum = _sum_squared_hinge(pos, pos, margin)
    all_sum = pos_sum + _sum_squared_hinge(pos, neg, margin)
    precision = pos_sum / (all_sum + 1e-12)
    return -precision.mean().to(positive.dtype)


def _sum_squared_hinge(queries, keys, margin):
    """For each query q: sum over the keys k of max(0, margin - (q - k))^2.

    Equal to building the (queries x keys) matrix of hinges and summing its rows, in
    O((Q + K) log K) time and O(Q + K) memory instead of O(Q * K): only the keys k > q - margin
    contribute, and their sum of (a + k)^2 = n * a^2 + 2a * sum(k) + sum(k^2), with a = margin - q,
    is read from suffix sums of the sorted keys. Sorting keeps the gradients of the keys, and the
    hinge is zero with a zero derivative at its boundary, so gradients are those of the matrix.
    Computed in float64, as the expansion sums many terms.
    """
    keys, _ = torch.sort(keys.double())
    queries = queries.double()
    zero = keys.new_zeros(1)
    suffix_k = torch.cat([keys.flip(0).cumsum(0).flip(0), zero])
    suffix_k2 = torch.cat([(keys**2).flip(0).cumsum(0).flip(0), zero])
    first = torch.searchsorted(keys.detach(), (queries - margin).detach(), right=True)
    count = (keys.shape[0] - first).double()
    a = margin - queries
    return count * a * a + 2 * a * suffix_k[first] + suffix_k2[first]


def bpr_contrastive(positive, negative, inference=False, pos_index=None, **kwargs):
    """Bayesian personalized ranking (BPR) loss of link prediction, as in ShadeWatcher.

    Each negative edge is paired with a positive edge (`pos_index[i]` is the positive edge of the
    negative edge i, by default the edge of the same index), whose score should be higher:
    softplus(s_neg - s_pos). At inference, edges are scored by their negated score (low score =
    anomalous).
    """
    if inference:
        return -positive
    if pos_index is None:
        if positive.shape[0] != negative.shape[0]:
            raise ValueError("The `bpr` loss requires a positive edge for each negative edge.")
        pos_index = torch.arange(positive.shape[0], device=positive.device)
    return F.softplus(negative.flatten() - positive.flatten()[pos_index]).mean()


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

        y_onehot = torch.zeros(batch_size, self.num_classes, device=feat.device)
        y_onehot.scatter_(1, torch.unsqueeze(label, dim=-1), self.margin)
        margin_logits = self.scale * (logits - y_onehot)
        err = cross_entropy(margin_logits, label, inference=inference)

        return err
