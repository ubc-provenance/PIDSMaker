"""Prediction of the embedding of the next snapshot, and key indicators (ProGrapher).

The encoder (`textrcnn`) predicts the graph2vec embedding of the current snapshot from the
previous ones. The loss of a snapshot is the L2 distance between its actual and predicted
embeddings (Eq. 6 of ProGrapher): the anomaly score of the snapshot.

At inference, the objective returns one score per node of the snapshot, so that ProGrapher is
evaluated per time window (`tw_evaluation`, as the paper) or per node (`node_evaluation`):

1. Key indicators (Section 4.4). graph2vec predicts each rooted subgraph (RSG) r of a snapshot
   from its embedding E with p(r | E) = softmax_r(w_r . E) over the RSGs of the snapshot (Eq. 1),
   w_r being the output vector of r. The RSGs are ranked by the difference of their
   log-likelihood under the actual and the predicted embeddings, log p(r | E) - log p(r | E'),
   which reduces to w_r . (E - E') up to a constant of the snapshot: the RSGs best explained
   by the actual snapshot, but not by its prediction, come first.
2. Each node is ranked by the best rank of the RSGs rooted at it (one per WL depth): nodes
   matching the top RSGs are the key indicators.
3. The score of a node is `snapshot_score * (1 - rank / num_rsgs)`: the nodes of the top RSG
   get the score of the snapshot, the others a fraction decreasing with the rank of their RSG.
   With `top_k`, only the nodes matching the `top_k` first RSGs (the key indicators reported by
   ProGrapher) keep a non-zero score.

The maximum node score of a snapshot is thus its own score: reduced with a max, as
`tw_evaluation` does with `max_val_loss`, node scores give the snapshot-level detection of the
paper.

RSGs unseen when training graph2vec are ignored, as when inferring the snapshot embedding.
"""

import torch
import torch.nn as nn


class SnapshotEmbeddingPrediction(nn.Module):
    def __init__(self, rsg_vectors, top_k=0):
        """
        Args:
            rsg_vectors: (V, d) output vectors of the RSGs of graph2vec (`syn1neg`)
            top_k: if > 0, only the nodes matching the `top_k` first RSGs get a score
        """
        super().__init__()
        self.register_buffer("rsg_vectors", rsg_vectors.float(), persistent=False)
        self.top_k = top_k
        self.val_scores = []

    def forward(self, h, inference, batch, validation=False, **kwargs):
        target = batch.graph_emb.float().view(1, -1)
        pred = h.view(1, -1)
        score = torch.linalg.vector_norm(target - pred)

        if not inference:
            return {"loss": score}

        if validation:
            self.val_scores.append(score.item())
        node_scores = score * self.key_indicator_weights(batch, (target - pred).view(-1))
        return {"loss": node_scores}

    @torch.no_grad()
    def key_indicator_weights(self, batch, diff):
        """Weight in [0, 1] of each node of the batch (in the order of `batch.original_n_id`)
        given by the rank of its best RSG, the RSGs being ranked by w_r . diff."""
        table = batch.snapshot_rsg[0].long()  # (n, D + 2): node ID, then RSG IDs
        rsg_ids = table[:, 1:]
        known = rsg_ids >= 0

        # Rank of the unique RSGs of the snapshot
        uniq, inverse = torch.unique(rsg_ids[known], return_inverse=True)
        num_rsgs = uniq.numel()
        order = torch.argsort(self.rsg_vectors[uniq] @ diff, descending=True)
        rank = torch.empty_like(order)
        rank[order] = torch.arange(num_rsgs, device=order.device)

        # Best rank of the RSGs of each node (num_rsgs if none is known)
        node_rank = torch.full_like(rsg_ids, num_rsgs)
        node_rank[known] = rank[inverse]
        best_rank = node_rank.min(dim=1).values

        weights = 1.0 - best_rank.float() / max(num_rsgs, 1)
        if self.top_k > 0:
            weights[best_rank >= self.top_k] = 0.0

        # Nodes of the batch, found by ID in the table (sorted by node ID)
        n_id = batch.original_n_id.long()
        rows = torch.searchsorted(table[:, 0].contiguous(), n_id)
        rows = rows.clamp(max=table.size(0) - 1)
        if not torch.equal(table[rows, 0], n_id):
            raise ValueError("Nodes of the batch are missing from the RSGs of their snapshot.")
        return weights[rows]

    def get_val_score(self):
        """Validation score used by `best_model_selection=best_val_score`: the opposite of the
        mean prediction error on the (benign) validation snapshots, as ProGrapher selects its
        model on validation data."""
        score = -sum(self.val_scores) / len(self.val_scores) if self.val_scores else 0.0
        self.val_scores = []
        return score
