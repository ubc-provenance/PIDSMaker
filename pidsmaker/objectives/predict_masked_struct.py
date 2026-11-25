import torch
import torch.nn as nn
from torch_geometric.utils import negative_sampling


class GMAEStructPrediction(nn.Module):
    """MAGIC's structure reconstruction: an MLP predicts whether node pairs are linked.

    As in MAGIC, `min(max_samples, num_nodes)` existing edges are sampled uniformly as positives,
    with as many uniform negative pairs. `max_samples=None` (or 0) uses all the edges.
    """

    def __init__(self, decoder, loss_fn, max_samples=None):
        super(GMAEStructPrediction, self).__init__()
        self.edge_recon_fc = decoder
        self.loss_fn = loss_fn
        self.max_samples = max_samples

    def sample_positive_edges(self, edge_index, num_nodes):
        num_edges = edge_index.shape[1]
        if not self.max_samples:
            return edge_index
        n = min(self.max_samples, num_nodes, num_edges)
        if n == num_edges:
            return edge_index
        idx = torch.randperm(num_edges, device=edge_index.device)[:n]
        return edge_index[:, idx]

    def forward(self, h, edge_index, inference, **kwargs):
        # The structure loss is a training signal only: it does not score nodes
        if inference:
            return {"loss": torch.zeros((h.shape[0],), device=h.device)}

        pos_edge_index = self.sample_positive_edges(edge_index, h.shape[0])
        neg_edge_index = negative_sampling(
            edge_index, num_nodes=h.shape[0], num_neg_samples=pos_edge_index.shape[1]
        )

        pos_samples = torch.cat([h[pos_edge_index[0]], h[pos_edge_index[1]]], dim=-1)
        neg_samples = torch.cat([h[neg_edge_index[0]], h[neg_edge_index[1]]], dim=-1)

        y_pred_pos = self.edge_recon_fc(pos_samples).squeeze(-1)
        y_pred_neg = self.edge_recon_fc(neg_samples).squeeze(-1)

        y_true = torch.cat([torch.ones_like(y_pred_pos), torch.zeros_like(y_pred_neg)])
        y_pred = torch.cat([y_pred_pos, y_pred_neg])

        loss = self.loss_fn(y_pred, y_true, inference=inference)
        return {"loss": loss}
