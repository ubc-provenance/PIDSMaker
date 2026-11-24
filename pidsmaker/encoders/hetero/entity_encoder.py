import torch
import torch.nn as nn
import torch.nn.functional as F


class EntityLinearEncoder(nn.Module):
    def __init__(self, in_dim, out_dim, encoder, activation=False, num_node_types=3):
        super().__init__()
        self.encoder = encoder
        self.activation = activation

        # One linear per entity type
        self.linears = nn.ModuleList([nn.Linear(in_dim, out_dim) for _ in range(num_node_types)])

    def forward(self, x, node_type, *args, **kwargs):
        node_type_idx = node_type.max(dim=1).indices
        out = torch.zeros_like(x)

        for i, layer in enumerate(self.linears):
            mask = node_type_idx == i
            if mask.any():
                h = layer(x[mask])
                if self.activation:
                    h = F.relu(h)
                out[mask] = h

        if self.encoder is not None:
            x = self.encoder(*args, x=out, node_type=node_type, **kwargs)
        return x

    def reset_state(self):
        if hasattr(self.encoder, "reset_state"):
            self.encoder.reset_state()
