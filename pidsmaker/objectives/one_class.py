"""Per-node-type Deep-SVDD hypersphere"""

import torch
import torch.nn as nn


class OneClass(nn.Module):
    def __init__(self, decoder, embed_dim, beta=0.5, eps=1e-3, warmup=2, num_node_types=1):
        super().__init__()
        self.decoder = decoder
        self.beta = beta
        self.eps = eps
        del warmup  # unused, matches OCRGCNBase's own dead arg
        self.num_node_types = num_node_types
        self.register_buffer("c", torch.zeros(num_node_types, embed_dim))
        self.register_buffer("r", torch.zeros(num_node_types))
        self._frozen = set()

    def freeze_type(self, t):
        self._frozen.add(t)

    def _node_type_idx(self, h, batch):
        node_type_argmax = getattr(batch, "node_type_argmax", None) if batch is not None else None
        if node_type_argmax is None:
            if self.num_node_types > 1:
                raise ValueError(
                    "OneClass needs `node_type_argmax` on the batch when num_node_types > 1, "
                    "otherwise every node silently shares a single hypersphere."
                )
            return torch.zeros(h.size(0), dtype=torch.long, device=h.device)
        return node_type_argmax

    def forward(self, h, inference, batch=None, **kwargs):
        h = self.decoder(h)
        node_type_idx = self._node_type_idx(h, batch)

        if not inference:
            with torch.no_grad():
                for nt in range(self.num_node_types):
                    if nt in self._frozen:
                        continue
                    mask = node_type_idx == nt
                    if not mask.any():
                        continue
                    c = h[mask].mean(0)
                    c[(c.abs() < self.eps) & (c < 0)] = -self.eps
                    c[(c.abs() < self.eps) & (c > 0)] = self.eps
                    self.c[nt] = c

        c_batch = self.c[node_type_idx]
        r_batch = self.r[node_type_idx]
        dist = torch.sum((h - c_batch) ** 2, dim=1)
        score = dist - r_batch ** 2

        if inference:
            return {"loss": score}

        if self._frozen:
            frozen_types = torch.tensor(sorted(self._frozen), device=h.device)
            active = ~torch.isin(node_type_idx, frozen_types)
        else:
            active = torch.ones_like(node_type_idx, dtype=torch.bool)
        loss = torch.mean(r_batch[active] ** 2 + (1.0 / self.beta) * torch.relu(score[active]))

        with torch.no_grad():
            for nt in range(self.num_node_types):
                if nt in self._frozen:
                    continue
                mask = node_type_idx == nt
                if not mask.any():
                    continue
                d_nt = dist[mask].detach()
                self.r[nt].copy_(torch.quantile(torch.sqrt(d_nt), 1 - self.beta))

        return {"loss": loss}
