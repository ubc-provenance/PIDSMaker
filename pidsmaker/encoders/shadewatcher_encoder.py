"""Encoder of ShadeWatcher (Zeng et al., IEEE S&P 2022): "ShadeWatcher: Recommendation-guided
Cyber Threat Analysis using System Audit Records".

ShadeWatcher sees system entity interactions as a recommendation problem on a knowledge graph
(KG) made of the events of the provenance graph, `(h, r, t)` with `r` the event type, and of the
interactions between entities, `(h, interact, t)`. Here, each edge of the graph gives both
triples: relation 0 is `interact` and relation `i + 1` the edge type `i`.

1. First-order information: entities are embedded in a `d`-dimensional space (`e_h`, from the
   node features) and relations in a `k`-dimensional space (`e_r`), with a projection matrix
   `W_r` (d x k) per relation (TransR). The distance of a triple is
   `f(h, r, t) = ||e_h W_r + e_r - e_t W_r||^2`, trained with the pairwise loss
   `softplus(f(h, r, t) - f(h, r, t') + margin)` against corrupted tails `t'` drawn uniformly among
   the nodes of the graph. This loss is returned as `aux_loss` and added to the loss of the
   objectives by the model during training.
2. Higher-order information: GNN layers propagate the embeddings `z^(0) = e` along the KG. Each
   entity `h` aggregates its neighbors `t` (the destinations of its edges) with a relational
   attention computed from the TransR embeddings (without gradient, as the attentive matrix of
   ShadeWatcher), normalized by a softmax over the triples of `h`:

       pi(h, r, t) = (e_t W_r)^T tanh(e_h W_r + e_r)
       z_N(h)^(l-1) = sum_{(h, r, t)} pi(h, r, t) z_t^(l-1)
       z_h^(l) = LeakyReLU([z_h^(l-1) || z_N(h)^(l-1)] W^(l))      (GraphSAGE aggregator)

   followed by dropout. The final representation concatenates the initial embedding and the L2
   normalized output of each layer: `z*_h = e_h || z_h^(1) / ||z_h^(1)|| || ...`.

Interactions are scored by the inner product of `z*` (see `predict_edge_contrastive` with
`loss: bpr` in `config/shadewatcher.yml`). If the size of `z*` differs from `out_dim`, a linear
layer projects it to `out_dim` (not in ShadeWatcher, whose config sets `out_dim` to the size of
`z*`).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import softmax


class ShadeWatcherEncoder(nn.Module):
    def __init__(
        self,
        in_dim,
        emb_dim,
        out_dim,
        num_edge_types,
        hidden_dims=(32, 16),
        kg_dim=32,
        kg_neg_ratio=2,
        kg_margin=1.0,
        dropout=0.2,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.kg_dim = kg_dim
        self.kg_neg_ratio = kg_neg_ratio
        self.kg_margin = kg_margin
        # Relation 0 is `interact`, relation i + 1 the edge type i
        self.num_relations = num_edge_types + 1

        # Entity embeddings `e` from the node features (ShadeWatcher: one embedding per entity)
        self.entity_emb = nn.Linear(in_dim, emb_dim)

        # TransR
        self.rel_emb = nn.Parameter(torch.empty(self.num_relations, kg_dim))
        self.rel_proj = nn.Parameter(torch.empty(self.num_relations, emb_dim, kg_dim))
        nn.init.xavier_uniform_(self.rel_emb)
        nn.init.xavier_uniform_(self.rel_proj)

        # GNN layers with the GraphSAGE aggregator
        dims = [emb_dim] + list(hidden_dims)
        self.layers = nn.ModuleList(
            nn.Linear(2 * dims[i], dims[i + 1]) for i in range(len(hidden_dims))
        )
        self.dropout = nn.Dropout(dropout)

        concat_dim = sum(dims)
        self.out_proj = nn.Linear(concat_dim, out_dim) if concat_dim != out_dim else None

        for lin in [self.entity_emb, *self.layers]:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)

    def _project(self, e, rel):
        """Projects the entity embeddings `e` (T, d) in the space of their relation `rel` (T,)."""
        out = e.new_empty((e.size(0), self.kg_dim))
        for r in torch.unique(rel):
            mask = rel == r
            out[mask] = e[mask] @ self.rel_proj[r]
        return out

    def _kg_triples(self, edge_index, edge_types):
        """Unique (h, r, t) triples of the KG: an interaction and an event per edge."""
        if edge_types is None:
            edge_rel = torch.zeros(edge_index.size(1), dtype=torch.long, device=edge_index.device)
        else:
            edge_rel = edge_types.argmax(dim=-1).long() + 1
        interact_rel = torch.zeros_like(edge_rel)
        h = torch.cat([edge_index[0], edge_index[0]]).long()
        t = torch.cat([edge_index[1], edge_index[1]]).long()
        r = torch.cat([interact_rel, edge_rel])
        triples = torch.unique(torch.stack([h, r, t]), dim=1)
        return triples[0], triples[1], triples[2]

    def _transr_distance(self, e_h, e_t, r):
        return (self._project(e_h, r) + self.rel_emb[r] - self._project(e_t, r)).pow(2).sum(dim=-1)

    def _transr_loss(self, e, h, r, t):
        """Pairwise loss of TransR against tails corrupted uniformly among the nodes."""
        num_nodes = e.size(0)
        idx = torch.arange(h.size(0), device=h.device).repeat(self.kg_neg_ratio)
        neg_t = torch.randint(0, num_nodes, (idx.size(0),), device=h.device)
        # Corrupted triples observed in the KG are dropped
        key = lambda h_, r_, t_: (h_ * self.num_relations + r_) * num_nodes + t_
        keep = ~torch.isin(key(h[idx], r[idx], neg_t), key(h, r, t))
        if keep.any():
            idx, neg_t = idx[keep], neg_t[keep]

        pos_dist = self._transr_distance(e[h], e[t], r)
        neg_dist = self._transr_distance(e[h[idx]], e[neg_t], r[idx])
        return F.softplus(pos_dist[idx] - neg_dist + self.kg_margin).mean()

    def forward(self, x, edge_index, edge_types=None, inference=False, **kwargs):
        e = self.entity_emb(x)
        h, r, t = self._kg_triples(edge_index, edge_types)

        # Knowledge-aware attention over the triples of each head entity, from TransR
        with torch.no_grad():
            att = (
                self._project(e[t], r) * torch.tanh(self._project(e[h], r) + self.rel_emb[r])
            ).sum(-1)
            att = softmax(att, index=h, num_nodes=e.size(0))

        z = e
        outputs = [e]
        for layer in self.layers:
            z_neigh = torch.zeros_like(z).index_add_(0, h, att.unsqueeze(-1) * z[t])
            z = F.leaky_relu(layer(torch.cat([z, z_neigh], dim=-1)), negative_slope=0.2)
            z = self.dropout(z)
            outputs.append(F.normalize(z, p=2, dim=-1))

        out = torch.cat(outputs, dim=-1)
        if self.out_proj is not None:
            out = self.out_proj(out)

        res = {"h": out}
        if not inference and h.numel() > 0:
            res["aux_loss"] = self._transr_loss(e, h, r, t)
        return res
