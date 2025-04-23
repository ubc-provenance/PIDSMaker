import torch
import torch.nn as nn

from pidsmaker.encoders import GRU


class TGNEncoder(nn.Module):
    # Code adapted from https://github.com/pyg-team/pytorch_geometric/blob/master/examples/tgn.py
    def __init__(
        self,
        encoder,
        memory,
        time_encoder,
        in_dim,
        memory_dim,
        use_node_feats_in_gnn,
        edge_features,
        device,
        use_memory,
        use_time_enc,
        edge_dim,
        use_time_order_encoding,
        project_src_dst,
        node_map,
        edge_map,
    ):
        super(TGNEncoder, self).__init__()
        self.encoder = encoder
        self.memory = memory
        self.device = device

        self.edge_features = edge_features
        if "time_encoding" in self.edge_features:
            self.time_encoder = time_encoder

        self.use_memory = use_memory
        self.use_time_enc = use_time_enc

        self.use_node_feats_in_gnn = use_node_feats_in_gnn
        self.project_src_dst = project_src_dst
        if not self.use_memory or self.use_node_feats_in_gnn:
            if self.project_src_dst:
                self.src_linear = nn.Linear(in_dim, memory_dim)
                self.dst_linear = nn.Linear(in_dim, memory_dim)
            else:
                self.linear = nn.Linear(in_dim, memory_dim)

        self.use_time_order_encoding = use_time_order_encoding
        if self.use_time_order_encoding:
            self.gru = GRU(edge_dim, edge_dim, device)

        self.node_map = node_map
        self.edge_map = edge_map

    def forward(self, batch, inference=False, **kwargs):
        n_id = batch.n_id_tgn  # NOTE: this one may need to be updated with no __inc__, or memory is get for unknown nodes
        edge_index = batch.edge_index_tgn
        x = batch.x_tgn

        # NOTE: these are shape (N,) not (E,)
        x_s = batch.x_from_tgn
        x_d = batch.x_to_tgn

        x_proj = None
        if (not self.use_memory) or self.use_node_feats_in_gnn:
            if self.project_src_dst:
                x_proj = self.src_linear(x_s) + self.dst_linear(x_d)
            else:
                x_proj = self.linear(x)

        if self.use_memory:
            h, last_update = self.memory(n_id)
            if self.use_node_feats_in_gnn:
                h = h + x_proj
        else:
            h = x_proj

        # Edge features
        edge_feats = []
        if "edge_type_triplet" in self.edge_features or "edge_type" in self.edge_features:
            edge_feats.append(batch.edge_type_tgn)
        if "msg" in self.edge_features:
            edge_feats.append(batch.msg_tgn)
        if "time_encoding" in self.edge_features:  # or self.use_time_enc
            if not self.use_memory:
                last_update = self.memory.get_last_update(n_id)
            curr_t = batch.t_tgn
            rel_t = last_update[edge_index[0]] - curr_t
            rel_t_enc = self.time_encoder(rel_t.to(h.dtype))
            edge_feats.append(rel_t_enc)
        edge_feats = torch.cat(edge_feats, dim=-1) if len(edge_feats) > 0 else None

        if self.use_time_order_encoding:
            if len(edge_feats) > 0:  # GRU doesn't work with empty sequences
                edge_feats = self.gru(edge_feats)
                self.gru.detach_state()

        node_type = batch.node_type_tgn
        edge_type = batch.edge_type_tgn

        x_dict, edge_index_dict = None, None
        node_type_argmax = None

        tgn_kwargs = {
            "x": h,
            "edge_index": edge_index,
            "edge_feats": edge_feats,
            "node_type": node_type,
            "edge_types": edge_type,
            "original_n_id": n_id,
            "x_dict": x_dict,
            "edge_index_dict": edge_index_dict,
            "node_type_argmax": node_type_argmax,
        }
        kwargs = {**kwargs, **tgn_kwargs}
        h = self.encoder(**kwargs)["h"]

        h_src = h[batch.reindexed_edge_index_tgn[0]]
        h_dst = h[batch.reindexed_edge_index_tgn[1]]

        # in the neigh loader, n_id is original n_id and the n_id of neighbors, here we remove neighbors and keep original node IDs
        h = h[batch.reindexed_original_n_id_tgn]

        # Update memory and neighbor loader with ground-truth state.
        if self.use_memory or self.use_time_enc:
            self.memory.update_state(batch.src, batch.dst, batch.t, batch.msg)

        # Detaching memory is only useful for backprop in training
        if self.use_memory and not inference:
            self.memory.detach()

        return {"h": h, "h_src": h_src, "h_dst": h_dst}

    def reset_state(self):
        if self.use_memory or self.use_time_enc:  # if memory is used
            self.memory.reset_state()  # Flushes memory.

    def to_device(self, device):
        self.device = device
        if hasattr(self, "gru"):
            self.gru.device = device
        return self
