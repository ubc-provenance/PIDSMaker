"""MAGIC system GAT encoder with masked graph representation learning.

Mirrors the GAT of the original MAGIC (Jia et al., USENIX Security 2024, `model/gat.py` and
`model/autoencoder.py`):
- attention over the source node, the destination node and the edge type (edge features);
- multi-head layers whose heads are concatenated: the hidden size is split among heads;
- a residual connection and an activation (one PReLU per layer) on every encoder layer;
- the encoder returns the concatenation of the outputs of all its layers, which feeds the
  structure reconstruction and, after a linear projection, the feature decoder. The k-NN
  detector scores this concatenation (MAGIC's paper) or only the last layer (MAGIC's code, see
  `detection_embedding` and `evaluation.node_evaluation.magic_embedding`);
- the decoder is a GAT whose last layer has no activation and no residual connection.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv


class MagicGAT(nn.Module):
    """GAT encoder/decoder for MAGIC masked graph learning.

    Encoder: `n_layers` GAT layers of `n_heads` concatenated heads; the hidden layers output
    `hid_dim` and the last one `out_dim` (each split among heads). The output `h` is the
    concatenation of all layer outputs, of size `(n_layers - 1) * hid_dim + out_dim`.

    Decoder (`is_decoder=True`): an optional linear projection without bias of the input
    (MAGIC's `encoder_to_decoder`, from `in_dim` to `hid_dim`), then `n_layers` GAT layers whose
    hidden layers have `hid_dim` units per head and whose last layer averages its heads into
    `out_dim`, without activation nor residual connection.
    """

    def __init__(
        self,
        in_dim,
        hid_dim,
        out_dim,
        n_layers,
        n_heads,
        feat_drop=0.1,
        attn_drop=0.0,
        negative_slope=0.2,
        residual=True,
        activation=None,
        is_decoder=False,
        edge_dim=None,
        in_proj_dim=None,
    ):
        super(MagicGAT, self).__init__()
        self.out_dim = out_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.is_decoder = is_decoder
        self.gats = nn.ModuleList()
        self.activations = nn.ModuleList()
        edge_dim = edge_dim or None

        # MAGIC's `encoder_to_decoder`: projects the concatenated encoder layers
        self.in_proj = None
        if in_proj_dim is not None:
            self.in_proj = nn.Linear(in_dim, in_proj_dim, bias=False)
            in_dim = in_proj_dim

        if not is_decoder:
            if hid_dim % n_heads or out_dim % n_heads:
                raise ValueError(
                    f"The hidden ({hid_dim}) and output ({out_dim}) sizes of the MAGIC encoder "
                    f"must be multiples of its number of heads ({n_heads})."
                )

        def make_activation():
            if activation is None:
                return nn.Identity()
            # One instance per layer: e.g. each PReLU learns its own slope
            return activation() if isinstance(activation, type) else _copy_module(activation)

        dim = in_dim
        for layer in range(n_layers):
            is_last = layer == n_layers - 1
            if is_decoder and is_last:
                per_head, heads, concat = out_dim, n_heads, False
                use_residual, act = False, nn.Identity()
            elif is_decoder:
                per_head, heads, concat = hid_dim, n_heads, True
                use_residual, act = residual, make_activation()
            else:
                layer_out = out_dim if is_last else hid_dim
                per_head, heads, concat = layer_out // n_heads, n_heads, True
                use_residual, act = residual, make_activation()

            conv = GATConv(
                dim,
                per_head,
                heads=heads,
                concat=concat,
                dropout=attn_drop,
                negative_slope=negative_slope,
                edge_dim=edge_dim,
                add_self_loops=False,
                residual=use_residual,
            )
            new_dim = per_head * heads if concat else per_head
            if use_residual and dim == new_dim:
                # As in MAGIC, the residual is the identity when the sizes match
                conv.res = nn.Identity()
            self.gats.append(conv)
            self.activations.append(act)
            dim = new_dim

        self.dropout = nn.Dropout(feat_drop)
        self.reset_parameters()

    def reset_parameters(self):
        # MAGIC initializes its GAT weights with Xavier normal and a ReLU gain
        gain = nn.init.calculate_gain("relu")
        for conv in self.gats:
            for lin in (conv.lin, getattr(conv, "lin_src", None), getattr(conv, "lin_dst", None),
                        conv.lin_edge, getattr(conv, "res", None)):
                if isinstance(lin, nn.Module) and getattr(lin, "weight", None) is not None:
                    nn.init.xavier_normal_(lin.weight, gain=gain)
            for att in (conv.att_src, conv.att_dst, getattr(conv, "att_edge", None)):
                if att is not None:
                    nn.init.xavier_normal_(att, gain=gain)
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)

    @property
    def concat_dim(self):
        """Size of the encoder output `h` (concatenation of all layer outputs)."""
        return sum(
            conv.heads * conv.out_channels if conv.concat else conv.out_channels
            for conv in self.gats
        )

    def forward(self, x, edge_index, edge_feats=None, **kwargs):
        h = x
        if self.in_proj is not None:
            h = self.in_proj(h)
        if self.gats[0].lin_edge is None:
            edge_feats = None

        hidden_list = []
        for conv, act in zip(self.gats, self.activations):
            h = conv(self.dropout(h), edge_index, edge_feats)
            h = act(h)
            hidden_list.append(h)

        if self.is_decoder:
            return h
        return {"h": torch.cat(hidden_list, dim=-1)}

    def detection_embedding(self, h):
        """Embedding of the k-NN detector: the output of the last encoder layer (`model.embed` of
        MAGIC), i.e. the last `out_dim` columns of `h`."""
        return h[..., -self.out_dim :]


def _copy_module(module):
    import copy

    return copy.deepcopy(module)
