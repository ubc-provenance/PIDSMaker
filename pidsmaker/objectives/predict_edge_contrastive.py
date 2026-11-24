import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score

NEG_SAMPLING_METHODS = ["shuffle_dst", "uniform", "uniform_dst"]


class EdgeContrastivePrediction(nn.Module):
    """Link prediction: edges of the graph (positives) are contrasted with random negative edges.

    Negative sampling (`neg_sampling`):
    - `shuffle_dst`: each source is paired with the destination of another edge, so negatives
      keep the degree distribution of destinations. On graphs where most edges go to a few
      hubs (e.g. authentications to domain controllers in LANL), negatives never contain
      edges to other nodes, which the model then learns to consider likely.
    - `uniform`: sources and destinations are drawn uniformly among the nodes embedded by the
      encoder (all the nodes of the dataset when the encoder embeds them all, see
      `node_id_embedding.all_nodes`), `neg_ratio` negatives per positive, as in Euler and Argus.
    - `uniform_dst`: the source of each positive edge is paired with destinations drawn uniformly
      among the embedded nodes (`neg_ratio` per positive), as in TGN and Jbeil.

    At validation, the AUC and AP of validation edges against uniform negatives measure how well
    the model predicts links (`get_val_score`, used by `best_model_selection: best_val_score`).
    """

    def __init__(self, decoder, loss_fn, graph_reindexer, neg_sampling="shuffle_dst", neg_ratio=1.0):
        super().__init__()

        if neg_sampling not in NEG_SAMPLING_METHODS:
            raise ValueError(f"Invalid neg_sampling {neg_sampling}, expected one of {NEG_SAMPLING_METHODS}")
        self.decoder = decoder
        self.loss_fn = loss_fn
        self.graph_reindexer = graph_reindexer
        self.neg_sampling = neg_sampling
        self.neg_ratio = neg_ratio
        self._val_pos, self._val_neg = [], []

    def _node_tables(self, edge_index, h, h_src, h_dst):
        """Node embedding tables (src, dst) indexed by `edge_index`."""
        if isinstance(h, torch.Tensor) and h.dim() == 2 and int(edge_index.max()) < h.size(0):
            return h, h, edge_index
        (h_src, h_dst), edge_index, _ = self.graph_reindexer._reindex_graph(
            edge_index, h_src, h_dst, x_is_tuple=True
        )
        return h_src, h_dst, edge_index

    def _uniform_negative_sampling(self, edge_index, h, h_src, h_dst, ratio, keep_src=False):
        table_src, table_dst, edge_index = self._node_tables(edge_index, h, h_src, h_dst)
        num_nodes = table_src.size(0)
        num_neg = max(1, int(round(edge_index.size(1) * ratio)))
        if keep_src:
            idx = torch.arange(num_neg, device=edge_index.device) % edge_index.size(1)
            src = edge_index[0, idx].long()
        else:
            src = torch.randint(0, num_nodes, (num_neg,), device=edge_index.device)
        dst = torch.randint(0, num_nodes, (num_neg,), device=edge_index.device)
        pos_hash = edge_index[0].long() * num_nodes + edge_index[1].long()
        keep = (src != dst) & ~torch.isin(src * num_nodes + dst, pos_hash)
        return table_src[src[keep]], table_dst[dst[keep]]

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

    def forward(self, h_src, h_dst, edge_index, inference, h=None, validation=False, **kwargs):
        pos_scores = self.decoder(h_src=h_src, h_dst=h_dst)

        if not inference:
            if self.neg_sampling in ("uniform", "uniform_dst"):
                neg_h_src, neg_h_dst = self._uniform_negative_sampling(
                    edge_index, h, h_src, h_dst, self.neg_ratio,
                    keep_src=self.neg_sampling == "uniform_dst",
                )
            else:
                neg_h_src, neg_h_dst = self._fast_negative_sampling(edge_index, h_src, h_dst)
            neg_scores = self.decoder(h_src=neg_h_src, h_dst=neg_h_dst)
        else:
            neg_scores = None
            if validation:
                # Link prediction quality: validation edges vs as many uniform negatives
                neg_h_src, neg_h_dst = self._uniform_negative_sampling(edge_index, h, h_src, h_dst, 1.0)
                self._val_pos.append(pos_scores.detach().reshape(-1).float().cpu())
                self._val_neg.append(
                    self.decoder(h_src=neg_h_src, h_dst=neg_h_dst).detach().reshape(-1).float().cpu()
                )

        # class_weights = compute_class_weights(node_type, num_classes=self.node_type_dim) \
        # if self.balanced_loss else None

        loss = self.loss_fn(pos_scores, neg_scores, inference=inference)
        return {"loss": loss.squeeze()}

    def get_val_score(self):
        """AUC + AP of the validation edges against uniform negatives (reset after each call)."""
        if not self._val_pos:
            return 0.0
        pos, neg = torch.cat(self._val_pos).numpy(), torch.cat(self._val_neg).numpy()
        self._val_pos, self._val_neg = [], []
        labels = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
        scores = np.r_[pos, neg]
        return float(roc_auc_score(labels, scores) + average_precision_score(labels, scores))
