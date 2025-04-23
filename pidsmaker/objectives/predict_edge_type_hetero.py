import torch
import torch.nn as nn


class EdgeTypePredictionHetero(nn.Module):
    """Adds a projection head after the decoder, between each possible pair of entites"""

    def __init__(self, decoder, loss_fn, edge_type_predictors, ntype2edgemap):
        super().__init__()
        self.decoder = decoder
        self.loss_fn = loss_fn
        self.edge_type_predictors = edge_type_predictors
        self.ntype2edgemap = ntype2edgemap

    def forward(self, h_src, h_dst, batch, inference, **kwargs):
        src_type_idx = batch.node_type_src_argmax
        dst_type_idx = batch.node_type_dst_argmax
        edge_type_classes = batch.edge_type_argmax

        h = self.decoder(h_src=h_src, h_dst=h_dst)

        losses = []
        for (src_type, dst_type), event_map in self.ntype2edgemap.items():
            mask = (src_type_idx == src_type) & (dst_type_idx == dst_type)
            if mask.any():
                layer = self.edge_type_predictors[f"{src_type}_{dst_type}"]
                out = layer(h[mask])

                reindexed_edge_types = event_map[edge_type_classes[mask]]
                loss = self.loss_fn(out, reindexed_edge_types, inference=inference)
                losses.append(loss)

        if inference:
            tot_loss = torch.cat(losses)
        else:
            tot_loss = torch.stack(losses).mean()
        return {"loss": tot_loss}
