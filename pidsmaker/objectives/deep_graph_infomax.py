"""Deep Graph Infomax on edge embeddings, with PCA anomaly scores, as in Anomal-E.

Anomal-E (Caville et al., Knowledge-Based Systems 2022): "Anomal-E: A self-supervised network
intrusion detection system based on graph neural networks". The encoder (E-GraphSAGE) is trained
with Deep Graph Infomax (DGI): the embeddings of the edges of a graph, z_e = [h_src || h_dst],
should agree with the summary of the graph s = sigmoid(mean(z_e)), whereas the embeddings of a
corrupted graph (whose edge features are shuffled) should not. A bilinear discriminator
D(z, s) = z^T W s is trained with binary cross-entropy.

Edges are then scored by an unsupervised anomaly detector fitted on the embeddings of benign
training edges. Here, the detector is a PCA: its mean and covariance are accumulated over the
training edges of the last epoch, and the score of an edge is the reconstruction error of its
embedding from the top principal components.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def corrupt_batch(batch):
    """Copy of a batch whose node features and edge features are shuffled (DGI corruption)."""
    corrupted = batch.clone()
    if getattr(corrupted, "edge_feats", None) is not None:
        perm = torch.randperm(corrupted.edge_feats.shape[0], device=corrupted.edge_feats.device)
        corrupted.edge_feats = corrupted.edge_feats[perm]
    if getattr(corrupted, "x", None) is not None and isinstance(corrupted.x, torch.Tensor):
        perm = torch.randperm(corrupted.x.shape[0], device=corrupted.x.device)
        corrupted.x = corrupted.x[perm]
    return corrupted


class DeepGraphInfomax(nn.Module):
    needs_corrupted_embeddings = True

    def __init__(self, embed_dim, pca_components):
        super().__init__()
        edge_dim = 2 * embed_dim
        self.discriminator = nn.Bilinear(edge_dim, edge_dim, 1)
        self.pca_components = pca_components
        self.register_buffer("count", torch.zeros(()))
        self.register_buffer("sum", torch.zeros(edge_dim))
        self.register_buffer("outer", torch.zeros(edge_dim, edge_dim))
        self._pca = None

    def reset_state(self):
        """Called before each training epoch: the PCA is fitted on the last epoch only."""
        self.count.zero_()
        self.sum.zero_()
        self.outer.zero_()
        self._pca = None

    def _discriminate(self, z, summary):
        # Bilinear score z^T W s + b with the same summary s for all edges, computed as
        # z @ (W s) + b: `nn.Bilinear` on the expanded summary materializes an (E, d, d) tensor
        # in its backward pass (tens of GB on graphs with ~10^5 edges)
        ws = self.discriminator.weight[0] @ summary.reshape(-1)
        return z @ ws + self.discriminator.bias

    def _fit_pca(self):
        mean = self.sum / self.count.clamp(min=1)
        cov = self.outer / self.count.clamp(min=1) - torch.outer(mean, mean)
        _, eigvecs = torch.linalg.eigh(cov)  # ascending eigenvalues
        components = eigvecs[:, -self.pca_components :]
        self._pca = (mean, components)

    def forward(self, h_src, h_dst, inference, h_src_corrupted=None, h_dst_corrupted=None, **kwargs):
        z = torch.cat([h_src, h_dst], dim=-1)

        if inference:
            if self._pca is None:
                if self.count == 0:
                    raise RuntimeError("Deep Graph Infomax: the PCA has no training edge to fit on.")
                self._fit_pca()
            mean, components = self._pca
            centered = z - mean
            recon = centered @ components @ components.T
            return {"loss": ((centered - recon) ** 2).sum(dim=-1)}

        # Embeddings of benign training edges used to fit the PCA
        z_detached = z.detach()
        self.count += z_detached.shape[0]
        self.sum += z_detached.sum(dim=0)
        self.outer += z_detached.T @ z_detached
        self._pca = None

        summary = torch.sigmoid(z.mean(dim=0, keepdim=True))
        pos = self._discriminate(z, summary)
        z_corrupted = torch.cat([h_src_corrupted, h_dst_corrupted], dim=-1)
        neg = self._discriminate(z_corrupted, summary)
        loss = F.binary_cross_entropy_with_logits(
            pos, torch.ones_like(pos)
        ) + F.binary_cross_entropy_with_logits(neg, torch.zeros_like(neg))
        return {"loss": loss}
