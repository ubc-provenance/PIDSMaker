import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.dense import HeteroDictLinear


class EventLinearEncoder(nn.Module):
    """Projects each (src_type, dst_type, edge_type) triplet with a separate linear layer"""

    def __init__(
        self, in_dim, out_dim, possible_events, node_map, edge_map, encoder, activation=False
    ):
        super().__init__()
        self.node_map = node_map
        self.edge_map = edge_map
        self.encoder = encoder
        self.activation = activation
        self.out_dim = out_dim

        possible_events_triplets = [
            (src_type, dst_type, event)
            for (src_type, dst_type), events in possible_events.items()
            for event in events
        ]

        in_dim_triplets = {"_".join(triplet): in_dim for triplet in possible_events_triplets}
        self.hetero_edge_proj = HeteroDictLinear(in_dim_triplets, out_dim)

    def forward(self, edge_feats, node_type, edge_types, edge_index, *args, **kwargs):
        src_type_argmax = node_type[edge_index[0]].max(dim=1).indices
        dst_type_argmax = node_type[edge_index[1]].max(dim=1).indices
        edge_type_argmax = edge_types.max(dim=1).indices

        triplets = torch.stack((src_type_argmax, dst_type_argmax, edge_type_argmax), dim=1)
        unique_triplets = torch.unique(triplets, dim=0).tolist()

        masks = {}
        edge_dict = {}
        for src_type, dst_type, edge_type in unique_triplets:
            mask = (
                (src_type_argmax == src_type)
                & (dst_type_argmax == dst_type)
                & (edge_type_argmax == edge_type)
            )
            mask = torch.where(mask)[0]

            key = "_".join(
                [self.node_map[src_type], self.node_map[dst_type], self.edge_map[edge_type]]
            )
            masks[key] = mask
            edge_dict[key] = edge_feats[mask]

        edge_dict_proj = self.hetero_edge_proj(edge_dict)
        assert len(edge_dict_proj) == len(edge_dict), (
            "Found src, dst, edge types that do not exist in `possible_events`"
        )

        out = torch.zeros((edge_feats.shape[0], self.out_dim), device=edge_feats.device)
        for key in edge_dict_proj:
            mask = masks[key]
            h_edge = edge_dict_proj[key]
            if self.activation:
                h_edge = F.relu(h_edge)
            out[mask] = h_edge

        x = self.encoder(
            *args,
            edge_index=edge_index,
            edge_feats=out,
            node_type=node_type,
            edge_types=edge_types,
            node_type_src_argmax=src_type_argmax,
            node_type_dst_argmax=dst_type_argmax,
            edge_type_argmax=edge_type_argmax,
            **kwargs,
        )
        return x

    def reset_state(self):
        if hasattr(self.encoder, "reset_state"):
            self.encoder.reset_state()
