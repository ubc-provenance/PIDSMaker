import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score

NEG_SAMPLING_METHODS = ["shuffle_dst", "uniform", "uniform_dst", "tgn"]


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
      among the embedded nodes (`neg_ratio` per positive).
    - `tgn`: the source of each edge is paired with a destination drawn uniformly among the
      destinations of the dataset and embedded by the TGN with the edges of the batch, as in TGN
      and Jbeil (requires `tgn_last_neighbor.negative_destinations`).

    Each negative edge is paired with a positive edge (that of its source, except with `uniform`),
    which pairwise losses such as `bpr` (ShadeWatcher) compare it to.

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

    def _uniform_negative_sampling(
        self, edge_index, h, h_src, h_dst, ratio, keep_src=False, return_pos_index=False
    ):
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
        if not keep.any():  # e.g. tiny graphs: keep the drawn edges rather than no negative
            keep = torch.ones_like(keep)
        neg_h_src, neg_h_dst = table_src[src[keep]], table_dst[dst[keep]]
        if not return_pos_index:
            return neg_h_src, neg_h_dst
        # Positive edge paired with each negative (that of its source with `keep_src`)
        pos_index = torch.arange(num_neg, device=edge_index.device) % edge_index.size(1)
        return neg_h_src, neg_h_dst, pos_index[keep]

    def _fast_negative_sampling(self, edge_index, h_src, h_dst, aug_coef=1.0, return_pos_index=False):
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

        # Positive edge paired with each negative (that of its source)
        pos_index = torch.arange(neg_ei.size(1), device=neg_ei.device)
        no_loop = neg_ei[0] != neg_ei[1]
        neg_ei, pos_index = neg_ei[:, no_loop], pos_index[no_loop]  # remove self-loops
        neg_hash = el_hash(neg_ei)

        no_collision = ~torch.isin(neg_hash, el1d)  # remove collision edges in positive edges
        neg_samples, pos_index = neg_ei[:, no_collision], pos_index[no_collision]

        neg_h_src = h_src[neg_samples[0]]
        neg_h_dst = h_dst[neg_samples[1]]

        if not return_pos_index:
            return neg_h_src, neg_h_dst
        return neg_h_src, neg_h_dst, pos_index

    def forward(self, h_src, h_dst, edge_index, inference, h=None, validation=False, batch=None, **kwargs):
        pos_scores = self.decoder(h_src=h_src, h_dst=h_dst)
        h_neg = getattr(batch, "h_neg_tgn", None) if batch is not None else None
        if self.neg_sampling == "tgn" and h_neg is None:
            raise ValueError(
                "neg_sampling=tgn requires the TGN negatives of "
                "`batching.intra_graph_batching.tgn_last_neighbor.negative_destinations`."
            )

        # Positive edge paired with each negative edge (by default, the edge of the same index)
        pos_index = None
        if not inference:
            if self.neg_sampling == "tgn":
                neg_h_src, neg_h_dst = h_src, h_neg
            elif self.neg_sampling in ("uniform", "uniform_dst"):
                neg_h_src, neg_h_dst, pos_index = self._uniform_negative_sampling(
                    edge_index, h, h_src, h_dst, self.neg_ratio,
                    keep_src=self.neg_sampling == "uniform_dst",
                    return_pos_index=True,
                )
            else:
                neg_h_src, neg_h_dst, pos_index = self._fast_negative_sampling(
                    edge_index, h_src, h_dst, return_pos_index=True
                )
            neg_scores = self.decoder(h_src=neg_h_src, h_dst=neg_h_dst)
        else:
            neg_scores = None
            if validation:
                # Link prediction quality: validation edges vs as many negatives
                if self.neg_sampling == "tgn":
                    neg_h_src, neg_h_dst = h_src, h_neg
                else:
                    neg_h_src, neg_h_dst = self._uniform_negative_sampling(edge_index, h, h_src, h_dst, 1.0)
                self._val_pos.append(pos_scores.detach().reshape(-1).float().cpu())
                self._val_neg.append(
                    self.decoder(h_src=neg_h_src, h_dst=neg_h_dst).detach().reshape(-1).float().cpu()
                )

        # class_weights = compute_class_weights(node_type, num_classes=self.node_type_dim) \
        # if self.balanced_loss else None

        pairing = {} if pos_index is None else {"pos_index": pos_index}
        loss = self.loss_fn(pos_scores, neg_scores, inference=inference, **pairing)
        if inference:
            # One loss per edge, also for batches of a single edge (`squeeze` would return a scalar)
            loss = loss.reshape(pos_scores.shape[0], -1)
            return {"loss": loss.squeeze(1) if loss.shape[1] == 1 else loss}
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
