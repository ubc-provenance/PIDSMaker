"""R-Caid system GAT encoder with residual MLP aggregation.

R-Caid-specific encoder combining multi-layer GAT with MLP-based aggregation
for root cause analysis and attack investigation in provenance graphs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class _RcaidMLP(nn.Module):
    """Internal MLP for R-Caid aggregation."""
    def __init__(self, input_dim, output_dim):
        super(_RcaidMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.fc2 = nn.Linear(output_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class RCaidGAT(nn.Module):
    """R-Caid GAT encoder with 3-layer attention and MLP aggregation.

    Combines three GAT layers with an MLP that aggregates intermediate and final
    representations for improved node embeddings in causal analysis tasks.
    """
    def __init__(self, in_dim, hid_dim, out_dim, dropout, num_heads=4):
        super(RCaidGAT, self).__init__()
        self.gat1 = GATConv(in_dim, hid_dim, heads=num_heads, concat=True)
        self.gat2 = GATConv(hid_dim * num_heads, hid_dim, heads=num_heads, concat=True)
        self.gat3 = GATConv(
            hid_dim * num_heads, out_dim, heads=1, concat=False
        )  # Output is not concatenated
        self.mlp = _RcaidMLP(hid_dim * num_heads + out_dim, out_dim)  # Input is concatenated
        self.dropout1 = nn.Dropout(dropout)

    def forward(self, x, edge_index, **kwargs):
        x1 = self.gat1(x, edge_index)
        x1 = F.relu(x1)
        # GAT Layer 2 with attention
        x2 = self.gat2(x1, edge_index)
        x2 = F.relu(x2)
        # Aggregation through attention in the third layer
        x3 = self.gat3(x2, edge_index)

        x3 = self.dropout1(x3)
        # Update through MLP (concatenate previous layer's output with the current output)
        mlp_input = torch.cat([x2, x3], dim=-1)

        out = self.mlp(mlp_input)

        return {"h": out}
