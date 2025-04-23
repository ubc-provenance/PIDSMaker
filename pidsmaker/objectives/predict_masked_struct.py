import torch
import torch.nn as nn
from torch_geometric.utils import negative_sampling


class GMAEStructPrediction(nn.Module):
    def __init__(self, decoder, loss_fn):
        super(GMAEStructPrediction, self).__init__()
        self.edge_recon_fc = decoder
        self.loss_fn = loss_fn

    def forward(self, h, edge_index, inference, **kwargs):
        pos_src, pos_dst = edge_index
        neg_edge_index = negative_sampling(
            edge_index, num_nodes=h.shape[0], num_neg_samples=pos_src.size(0)
        )
        h_src, h_dst = h[edge_index[0]], h[edge_index[1]]

        pos_samples = torch.cat([h_src, h_dst], dim=-1)
        neg_samples = torch.cat([h[neg_edge_index[0]], h[neg_edge_index[1]]], dim=-1)

        y_pred_pos = self.edge_recon_fc(pos_samples).squeeze(-1)
        y_pred_neg = self.edge_recon_fc(neg_samples).squeeze(-1)

        y_true = torch.cat([torch.ones_like(y_pred_pos), torch.zeros_like(y_pred_neg)])
        y_pred = torch.cat([y_pred_pos, y_pred_neg])

        # TODO: Here it returns a loss for each edge so we can't use it directly at node level
        # and combine it with node level
        if inference:
            losses = torch.zeros((h.shape[0],), device=h.device)
            return {"loss": losses}

        loss = self.loss_fn(y_pred, y_true, inference=inference)

        return {"loss": loss}
