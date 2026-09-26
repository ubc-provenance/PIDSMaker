import torch.nn as nn
from torch_geometric.nn import HGTConv

from pidsmaker.hetero import hetero_to_homo_features


class HeteroGraphTransformer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, num_layers, metadata, device, node_map):
        super().__init__()

        self.out_dim = out_dim
        self.device = device
        self.node_map = node_map

        self.lin_dict = nn.ModuleDict()
        node_types = metadata[0]
        for node_type in node_types:
            self.lin_dict[node_type] = nn.Linear(in_dim, out_dim)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(out_dim, out_dim, metadata=metadata, heads=num_heads)
            self.convs.append(conv)

    def forward(self, edge_index_dict, x_dict, node_type_argmax, x, **kwargs):
        x_dict = {node_type: self.lin_dict[node_type](x).relu_() for node_type, x in x_dict.items()}

        if len(edge_index_dict) > 0:
            for layer in self.convs:
                x_dict = layer(x_dict, edge_index_dict)

        x = hetero_to_homo_features(
            x_dict=x_dict,
            node_types=node_type_argmax,
            node_map=self.node_map,
            device=self.device,
            num_nodes=x.shape[0],
            out_dim=self.out_dim,
        )
        return {"h": x}
