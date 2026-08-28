"""RGCN encoder (OCR-APT OCRGCN backbone, arXiv:2510.15188)."""

import torch
import torch.nn as nn
from torch_geometric.nn import RGCNConv


class RGCN(nn.Module):
    # hidden layer weight-tied
    def __init__(self, in_dim, out_dim, num_relations, num_layers, activation, dropout):
        super().__init__()
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers

        self.conv_in = RGCNConv(in_dim, out_dim, num_relations)
        self.conv_hidden = RGCNConv(out_dim, out_dim, num_relations)

    def forward(self, x, edge_index, edge_types=None, **kwargs):
        rel = edge_types.argmax(dim=-1)  # one-hot -> relation index
        x = self.conv_in(x, edge_index, rel)
        for _ in range(self.num_layers - 1):
            x = self.activation(x)
            x = self.dropout(x)
            x = self.conv_hidden(x, edge_index, rel)
        return {"h": x}


class PerTypeRGCN(nn.Module):
    # One RGCN per node type
    def __init__(self, in_dim, out_dim, num_relations, num_layers, activation_factory,
                 dropout, num_node_types):
        super().__init__()
        self.num_node_types = num_node_types
        self.out_dim = out_dim
        self.encoders = nn.ModuleDict({
            str(t): RGCN(in_dim, out_dim, num_relations, num_layers, activation_factory(), dropout)
            for t in range(num_node_types)
        })
        self._frozen = set()

    def freeze_type(self, t):
        if t in self._frozen:
            return
        self._frozen.add(t)
        for p in self.encoders[str(t)].parameters():
            p.requires_grad_(False)

    def forward(self, x, edge_index, edge_types=None, node_type_argmax=None, **kwargs):
        if node_type_argmax is None:
            if self.num_node_types > 1:
                raise ValueError(
                    "PerTypeRGCN needs `node_type_argmax` on the batch when num_node_types > 1, "
                    "otherwise every node silently falls back to a single type."
                )
            node_type_argmax = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        h_out = None
        for t_str, enc in self.encoders.items():
            t = int(t_str)
            mask = node_type_argmax == t
            if not mask.any():
                continue
            if t in self._frozen:
                with torch.no_grad():
                    h_t = enc(x=x, edge_index=edge_index, edge_types=edge_types)["h"]
            else:
                h_t = enc(x=x, edge_index=edge_index, edge_types=edge_types)["h"]
            if h_out is None:
                h_out = torch.zeros(x.size(0), self.out_dim, device=x.device, dtype=h_t.dtype)
            h_out[mask] = h_t[mask]

        if h_out is None:
            h_out = torch.zeros(x.size(0), self.out_dim, device=x.device)
        return {"h": h_out}
