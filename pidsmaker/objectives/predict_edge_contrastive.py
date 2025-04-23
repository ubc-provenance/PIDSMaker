import torch
import torch.nn as nn


class EdgeContrastivePrediction(nn.Module):
    def __init__(self, decoder, loss_fn, graph_reindexer):
        super().__init__()

        self.decoder = decoder
        self.loss_fn = loss_fn
        self.graph_reindexer = graph_reindexer

    def _fast_negative_sampling(self, edge_index, h_src, h_dst, aug_coef=1.0):
        (h_src, h_dst), edge_index, _ = self.graph_reindexer._reindex_graph(
            edge_index, h_src, h_dst, x_is_tuple=True
        )

        neg_dst_idx = torch.randperm(
            len(edge_index[0])
        )  # TODO: we may want to try also putting source nodes in the neg dst nodes
        neg_dst = edge_index[1, neg_dst_idx]
        neg_ei = torch.stack([edge_index[0], neg_dst])

        num_nodes = (edge_index.max() + 1).item()
        el_hash = lambda x: x[0, :] + x[1, :] * num_nodes
        el1d = el_hash(edge_index)

        neg_ei = neg_ei[:, neg_ei[0] != neg_ei[1]]  # remove self-loops
        neg_hash = el_hash(neg_ei)

        neg_samples = neg_ei[
            :, ~torch.isin(neg_hash, el1d)
        ]  # remove collision edges in positive edges

        neg_h_src = h_src[neg_samples[0]]
        neg_h_dst = h_dst[neg_samples[1]]

        return neg_h_src, neg_h_dst

    def forward(self, h_src, h_dst, edge_index, inference, **kwargs):
        pos_scores = self.decoder(h_src=h_src, h_dst=h_dst)

        if not inference:
            neg_h_src, neg_h_dst = self._fast_negative_sampling(edge_index, h_src, h_dst)
            neg_scores = self.decoder(h_src=neg_h_src, h_dst=neg_h_dst)
        else:
            neg_scores = None

        # class_weights = compute_class_weights(node_type, num_classes=self.node_type_dim) \
        # if self.balanced_loss else None

        loss = self.loss_fn(pos_scores, neg_scores, inference=inference)
        return {"loss": loss.squeeze()}
