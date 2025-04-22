import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GATConv,
    GINConv,
    GINEConv,
    HGTConv,
    MessagePassing,
    SAGEConv,
    TransformerConv,
)
from torch_geometric.nn.dense import HeteroDictLinear

from pidsmaker.custom_mlp import CustomMLPAsbtract
from pidsmaker.hetero import _compute_hetero_features, hetero_to_homo_features
from pidsmaker.utils.dataset_utils import rel2id_darpa_tc


class CustomMLPEncoder(CustomMLPAsbtract):
    def __init__(self, in_dim, out_dim, architecture, dropout):
        super().__init__(in_dim, out_dim, architecture, dropout)

    def forward(self, x, **kwargs):
        h = self.mlp(x)
        return {"h": h}


class GraphAttentionEmbedding(nn.Module):
    def __init__(
        self,
        in_dim,
        hid_dim,
        out_dim,
        edge_dim,
        dropout,
        activation,
        num_heads,
        concat,
        flow="source_to_target",
    ):
        super().__init__()

        conv2_in_dim = hid_dim * num_heads if concat else hid_dim
        self.conv = TransformerConv(
            in_dim,
            hid_dim,
            heads=num_heads,
            dropout=dropout,
            edge_dim=edge_dim,
            concat=concat,
            flow=flow,
        )
        self.conv2 = TransformerConv(
            conv2_in_dim,
            out_dim,
            heads=1,
            concat=False,
            dropout=dropout,
            edge_dim=edge_dim,
            flow=flow,
        )
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, x, edge_index, edge_feats=None, **kwargs):
        x = self.activation(self.conv(x, edge_index, edge_feats))
        x = self.dropout(x)
        x = self.conv2(x, edge_index, edge_feats)
        return {"h": x}


class HeteroGraphTransformer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, num_layers, metadata, device, node_map):
        super().__init__()

        self.out_dim = out_dim
        self.device = device
        self.node_map = node_map

        self.lin_dict = torch.nn.ModuleDict()
        node_types = metadata[0]
        for node_type in node_types:
            self.lin_dict[node_type] = nn.Linear(in_dim, out_dim)

        self.convs = torch.nn.ModuleList()
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
        is_hetero,
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

        self.is_hetero = is_hetero
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

        # Hetero stuff
        if self.is_hetero:
            node_type_argmax = node_type.max(dim=1).indices
            edge_type_argmax = edge_type.max(dim=1).indices

            x_dict, edge_index_dict = _compute_hetero_features(
                edge_index=edge_index,
                x=h,
                node_type_argmax=node_type_argmax,
                edge_type_argmax=edge_type_argmax,
                node_map=self.node_map,
                edge_map=self.edge_map,
            )
        else:
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


class SAGE(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, activation, dropout):
        super(SAGE, self).__init__()
        self.conv1 = SAGEConv(in_dim, hid_dim, normalize=False)
        self.conv2 = SAGEConv(hid_dim, out_dim, normalize=False)
        self.activation = activation
        self.drop = nn.Dropout(dropout)

    def forward(self, x, edge_index, **kwargs):
        x = self.activation(self.conv1(x, edge_index))
        x = self.drop(x)
        x = self.conv2(x, edge_index)
        return {"h": x}


class GRU(nn.Module):
    def __init__(self, x_dim, h_dim, device, hidden_units=1):
        super(GRU, self).__init__()
        self.rnn = nn.GRU(x_dim, h_dim, num_layers=hidden_units)
        self.hidden_units = hidden_units

        self.h_dim = h_dim
        self.device = device
        self.reset_state()

    def reset_state(self):
        self.hidden = self._init_hidden()

    def detach_state(self):
        self.hidden = self.hidden.detach()

    def _init_hidden(self):
        return torch.zeros(self.hidden_units, self.h_dim, requires_grad=True).to(self.device)

    def forward(self, xs, h0=None, include_h=False):
        xs, self.hidden = self.rnn(xs, self.hidden)

        if not include_h:
            return xs
        return xs, self.hidden


class LSTM(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        cell_clip=None,
        type_specific_decoding=False,
        exclude_file=True,
        exclude_ip=True,
        typed_hidden_rep=False,
        edge_dim=None,
        full_param=False,
        num_edge_type=10,
    ):
        """
        LSTM Class initialiser
        Takes integer sizes of input features, output features and set up model linear network layers
        """
        super(LSTM, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self._cell_clip = cell_clip  # clip cell to avoid value overflow
        self._type_specific_decoding = (
            type_specific_decoding  # decode each type of nodes separately
        )
        self._exclude_file = (
            exclude_file  # if type_specific_decoding is true, exclude file node decoding
        )
        self._exclude_ip = exclude_ip  # if type_specific_decoding is true, exclude ip node decoding
        self._typed_hidden_rep = (
            typed_hidden_rep  # include edge type embedding in node hidden representation
        )
        self._edge_dim = (
            edge_dim if edge_dim is not None else in_features
        )  # TODO: edge_dim defaults to in_features
        # TODO: full parametrization implementation is incomplete due to runtime in-place operation error
        self._full_param = (
            full_param  # use full parametrization instead of typed hidden representation
        )
        self._num_edge_type = num_edge_type  # TODO: number of edge type defaults to 15

        self.W_iou = nn.Linear(self.in_features, 3 * self.out_features)
        if self._typed_hidden_rep:
            self.U_iou = torch.randn(
                self.out_features, 3 * self.out_features, self._edge_dim, requires_grad=True
            )
        else:
            if self._full_param:
                self.U_ious = nn.ModuleList()
                for _ in range(self._num_edge_type):
                    self.U_ious.append(nn.Linear(self.out_features, 3 * self.out_features))
            else:
                self.U_iou = nn.Linear(self.out_features, 3 * self.out_features)

        self.W_f = nn.Linear(self.in_features, self.out_features)
        if self._typed_hidden_rep:
            self.U_f = torch.randn(
                self.out_features, self.out_features, self._edge_dim, requires_grad=True
            )
        else:
            if self._full_param:
                self.U_fs = nn.ModuleList()
                for _ in range(self._num_edge_type):
                    self.U_fs.append(nn.Linear(self.out_features, self.out_features))
            else:
                self.U_f = nn.Linear(self.out_features, self.out_features)

    def forward(self, x, edge_index, edge_feats, edge_types, **kwargs):
        """Compute tree LSTM given a batch."""
        batch_size = x.shape[0]

        adjacency_list = torch.transpose(edge_index, 0, 1)
        features = x

        node_order = torch.zeros(batch_size)
        edge_order = torch.zeros(adjacency_list.shape[0])

        for i in range(adjacency_list.shape[0]):
            node_order[adjacency_list[i, 0]] = node_order[adjacency_list[i, 0]] + 1
            edge_order[i] = node_order[adjacency_list[i, 0]]

        # Retrieve device the model is currently loaded on to generate h, c, and h_sum result buffers
        device = next(self.parameters()).device
        tmp = torch.arange(1.0, 11.0).reshape(10, 1).to(device)
        edge_types = torch.matmul(edge_types, tmp)

        # h and c states for every node in the batch
        h = torch.zeros(batch_size, self.out_features, device=device)
        c = torch.zeros(batch_size, self.out_features, device=device)

        # h_sum storage buffer
        if self._typed_hidden_rep:
            h_sum = torch.zeros(batch_size, self.out_features, self._edge_dim, device=device)
        else:
            if self._full_param:
                raise NotImplementedError
            else:
                h_sum = torch.zeros(batch_size, self.out_features, device=device)

        # populate the h and c states respecting computation order

        for n in range(int(torch.max(node_order)) + 1):
            self._run_lstm(
                n,
                h,
                c,
                h_sum,
                features,
                node_order,
                adjacency_list,
                edge_order,
                edge_feats,
                edge_types,
            )

        assert not (torch.isnan(h).any() or torch.isinf(h).any()), "h has inf or nan"
        assert not (torch.isnan(c).any() or torch.isinf(c).any()), "c has inf or nan"

        return {"h": h}

    def _run_lstm(
        self,
        iteration,
        h,
        c,
        h_sum,
        features,
        node_order,
        adjacency_list,
        edge_order,
        edge_features,
        edge_types,
    ):
        """Helper function to evaluate all tree nodes currently able to be evaluated."""
        # N is the number of nodes in the tree
        # n is the number of nodes to be evaluated on in the current iteration
        # E is the number of edges in the tree
        # e is the number of edges to be evaluated on in the current iteration
        # F is the number of features in each node
        # M is the number of hidden neurons in the network

        # node_order is a tensor of size N x 1
        # edge_order is a tensor of size E x 1
        # features is a tensor of size N x F
        # adjacency_list is a tensor of size E x 2

        # node_mask is a tensor of size N x 1
        node_mask = node_order == iteration

        # edge_mask is a tensor of size E x 1
        edge_mask = edge_order == iteration

        # x is a tensor of size n x F
        x = features[node_mask, :]
        # print("x shape: {}".format(x.shape))

        # At iteration 0 none of the nodes should have children
        # Otherwise, select the child nodes needed for current iteration
        # and sum over their hidden states
        if iteration == 0:
            iou = self.W_iou(x)
        else:
            # adjacency_list is a tensor of size e x 2
            adjacency_list = adjacency_list[edge_mask, :]

            if self._typed_hidden_rep or self._full_param:
                # edge_types is a tensor of size e x 1
                edge_types = edge_types[edge_mask].long()
                # edge_features is a tensor of size e x F
                edge_features = edge_features[edge_types]
                edge_cnt = 0

            # parent_indexes and child_indexes are tensors of size e x 1
            # parent_indexes and child_indexes contain the integer indexes needed to index into
            # the feature and hidden state arrays to retrieve the data for those parent/child nodes.
            parent_indexes = adjacency_list[:, 0]
            child_indexes = adjacency_list[:, 1]

            # child_h and child_c are tensors of size e x 1
            child_h = h[child_indexes, :]
            child_c = c[child_indexes, :]

            # Add child hidden states to parent offset locations
            for pindex, cindex in zip(parent_indexes, child_indexes):
                if self._typed_hidden_rep:
                    h_sum[pindex, :, :] += torch.ger(h[cindex, :], edge_features[edge_cnt])
                    edge_cnt += 1
                else:
                    if self._full_param:
                        raise NotImplementedError
                    else:
                        h_sum[pindex, :] += h[cindex, :]

            # print("h_sum shape: {}".format(h_sum.shape))

            if self._typed_hidden_rep:
                # print("h_sum masked shape: {}".format(h_sum[node_mask, :, :].shape))
                # print("U_iou shape: {}".format(self.U_iou.shape))
                # TODO: is this correct?
                u_h_sum = torch.zeros(
                    h_sum[node_mask, :, :].shape[0], 3 * self.out_features, self._edge_dim
                )
                for d in range(self._edge_dim):
                    u_h_sum[:, :, d] = torch.mm(h_sum[node_mask, :, d], self.U_iou[:, :, d])
                # print("U_h_sum shape: {}".format(u_h_sum.shape))
                iou = self.W_iou(x) + torch.sum(u_h_sum, dim=2)
            else:
                if self._full_param:
                    raise NotImplementedError
                else:
                    iou = self.W_iou(x) + self.U_iou(h_sum[node_mask, :])

        # i, o and u are tensors of size n x M
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i = torch.sigmoid(i)
        o = torch.sigmoid(o)
        u = torch.tanh(u)

        # At iteration 0 none of the nodes should have children
        # Otherwise, calculate the forget states for each parent node and child node
        # and sum over the child memory cell states
        c[node_mask, :] = torch.mul(i, u)
        assert not (torch.isnan(c[node_mask, :]).any() or torch.isinf(c[node_mask, :]).any()), (
            "c mask has inf or nan"
        )

        if iteration > 0:
            # f is a tensor of size e x M
            if self._typed_hidden_rep:
                # TODO: is this correct?
                # print("child_h shape: {}".format(child_h.shape))
                # print("U_f shape: {}".format(self.U_f.shape))
                f_h = torch.zeros(child_h.shape[0], self.out_features, self._edge_dim)
                # print("f_h shape: {}".format(f_h.shape))
                h_e = torch.zeros(child_h.shape[0], self.out_features, self._edge_dim)
                for i in range(child_h.shape[0]):
                    h_e[i, :, :] = torch.ger(child_h[i, :], edge_features[i])
                # print("h_e shape: {}".format(h_e.shape))
                for d in range(self._edge_dim):
                    f_h[:, :, d] = torch.mm(h_e[:, :, d], self.U_f[:, :, d])
                f = self.W_f(features[parent_indexes, :]) + torch.sum(f_h, dim=2)
            else:
                if self._full_param:
                    u_f_list = list()
                    for i in range(child_h.shape[0]):
                        e_i = 0
                        for mod in self.U_fs:
                            # TODO: double check
                            if e_i == edge_types[i]:
                                u_f_list.append(mod(child_h[i]))
                            e_i += 1
                    u_f_stacked = torch.stack(u_f_list)
                    f = self.W_f(features[parent_indexes, :]) + u_f_stacked
                else:
                    f = self.W_f(features[parent_indexes, :]) + self.U_f(child_h)
            # print("f shape: {}".format(f.shape))
            f = torch.sigmoid(f)
            assert not (torch.isnan(f).any() or torch.isinf(f).any()), (
                "f has inf or nan at iteration {}".format(iteration)
            )

            # fc is a tensor of size e x M
            # if (child_c != child_c).any() and iteration == 1:
            #     print("Child_c: {}".format(child_c))
            fc = torch.mul(f, child_c)
            assert not (torch.isnan(fc).any() or torch.isinf(fc).any()), (
                "fc has inf or nan at iteration {}".format(iteration)
            )

            # if (fc != fc).any():
            #     print("Invalid fc at iteration {}".format(iteration))

            for cindex, pindex in enumerate(parent_indexes):
                c[pindex, :] += fc[cindex, :]
                if self._cell_clip is not None:
                    c[pindex, :] = torch.clamp_(c[pindex, :], -self._cell_clip, self._cell_clip)

        h[node_mask, :] = torch.mul(o, torch.tanh(c[node_mask]))


class _RcaidMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(_RcaidMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.fc2 = nn.Linear(output_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class RCaidGAT(nn.Module):
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


class SumAggregation(MessagePassing):
    def __init__(
        self,
        in_dim,
        hid_dim,
        out_dim,
    ):
        super().__init__(aggr="sum")
        self.lin1 = nn.Linear(in_dim, hid_dim)
        self.lin2 = nn.Linear(hid_dim, out_dim)

    def forward(self, x, edge_index, **kwargs):
        x = torch.tanh(self.lin1(self.propagate(edge_index, x=x)))
        x = self.lin2(x)  # we need weights + tanh if "sum" aggreg is used, or too large gradients
        return {"h": x}


class LinearEncoder(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, *args, **kwargs):
        if isinstance(x, tuple):
            h = self.dropout(self.lin1(x[0])), self.dropout(self.lin1(x[1]))
        else:
            h = self.dropout(self.lin1(x))
        return {"h": h}


class GIN(torch.nn.Module):
    def __init__(
        self,
        in_dim,
        hid_dim,
        out_dim,
        edge_dim,
        dropout=0.25,
        flow="target_to_source",
    ):
        super(GIN, self).__init__()

        nn1 = nn.Sequential(nn.Linear(in_dim, hid_dim), nn.ReLU(), nn.Linear(hid_dim, hid_dim))
        nn2 = nn.Sequential(nn.Linear(hid_dim, hid_dim), nn.ReLU(), nn.Linear(hid_dim, hid_dim))

        if edge_dim is None:
            self.conv1 = GINConv(nn1)
            self.conv2 = GINConv(nn2)
        else:
            self.conv1 = GINEConv(nn1, edge_dim=edge_dim)
            self.conv2 = GINEConv(nn2, edge_dim=edge_dim)

        self.fc1 = nn.Linear(hid_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, out_dim)

        self.drop = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_feats, **kwargs):
        x = self.conv1(x, edge_index, edge_feats)
        x = torch.tanh(x)
        x = self.drop(x)

        x = self.conv2(x, edge_index, edge_feats)
        x = torch.tanh(x)

        # NOTE: It's worst in inductive setting to use 2 dropouts
        # x = self.drop(x)

        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return {"h": x}


class MagicGAT(nn.Module):
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
        residual=False,
        activation=None,
        concat_out=False,
        is_decoder=False,
    ):
        super(MagicGAT, self).__init__()
        self.out_dim = out_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.concat_out = concat_out
        self.gats = nn.ModuleList()
        self.is_decoder = is_decoder

        # First layer
        self.gats.append(
            GATConv(
                in_dim,
                hid_dim,
                heads=n_heads,
                concat=self.concat_out,
                dropout=attn_drop,
                negative_slope=negative_slope,
            )
        )

        # Hidden layers
        for _ in range(1, n_layers - 1):
            self.gats.append(
                GATConv(
                    hid_dim * n_heads,
                    hid_dim,
                    heads=n_heads,
                    concat=self.concat_out,
                    dropout=attn_drop,
                    negative_slope=negative_slope,
                )
            )

        # Last layer
        self.gats.append(
            GATConv(
                hid_dim * n_heads,
                out_dim,
                heads=1,
                concat=self.concat_out,
                dropout=attn_drop,
                negative_slope=negative_slope,
            )
        )

        self.dropout = nn.Dropout(feat_drop)
        self.activation = activation
        self.residual = residual
        self.last_linear = nn.Linear(hid_dim * n_heads, out_dim)

    def forward(self, x, edge_index, **kwargs):
        hidden_list = []
        h = x

        # Forward through GAT layers
        for layer in range(self.n_layers):
            h_in = h  # For residual connection
            h = self.dropout(h)
            h = self.gats[layer](h, edge_index)

            if self.residual and layer > 0:  # Adding residual connection if enabled
                if layer == self.n_layers - 1:
                    h_in = self.last_linear(h_in)
                h = h + h_in

            if self.activation:
                h = self.activation(h)

            hidden_list.append(h)

        return h if self.is_decoder else {"h": h}

    def reset_classifier(self, num_classes):
        self.head = nn.Linear(self.out_dim, num_classes)


class AncestorEncoder(nn.Module):
    def __init__(self, in_dim, out_dim, edge_dim, encoder, num_nodes, device):
        super().__init__()
        self.rnn = nn.LSTM(in_dim + edge_dim, out_dim, batch_first=True)
        self.hidden_states = {}  # {process_id: (h, c)}
        self.embedding_store = torch.zeros((num_nodes, out_dim), device=device)
        self.edge_dim = edge_dim

        considered_events = ["EVENT_EXECUTE", "EVENT_CLONE"]
        self.considered_events = torch.tensor(
            [rel2id_darpa_tc[event] - 1 for event in considered_events], device=device
        )

        self.encoder = encoder
        self.linear = nn.Linear(out_dim + in_dim, out_dim)

    def forward(self, edge_index, *args, **kwargs):
        x = self._forward(edge_index=edge_index, *args, **kwargs)

        src, dst = edge_index
        for arg in ["x_src", "x_dst", "x"]:
            kwargs.pop(arg, None)
        res = self.encoder(edge_index=edge_index, x=x, x_src=x[src], x_dst=x[dst], **kwargs)
        return res

    def _forward(self, edge_index, edge_types, x, original_n_id, **kwargs):
        x_src, x_dst = x[edge_index[0]], x[edge_index[1]]
        mask = torch.isin(edge_types.argmax(dim=1), self.considered_events)
        edge_index, edge_types, x_src, x_dst = (
            edge_index[:, mask],
            edge_types[mask],
            x_src[mask],
            x_dst[mask],
        )  # NOTE: check here if the filtered edge_index makes sense

        batch_inputs = torch.cat([edge_types, x_src], dim=1)
        process_ids = original_n_id[edge_index[1]]
        process_features = x_dst

        if batch_inputs.shape[0] > 0:
            # TODO: multiple forward are needed to handle duplicate destination nodes
            x_rnn = self._process_batch(batch_inputs, process_ids, process_features)
            self.embedding_store[process_ids] = x_rnn.detach()

            embeddings = self.embedding_store[original_n_id]
            mask = embeddings.any(dim=1)
            new_x = x.clone()
            new_x[mask] = self.linear(torch.cat([new_x[mask], embeddings[mask]], dim=1))
            return new_x

        return x

    def _process_batch(self, emb_sequence, process_ids, process_features):
        """Runs an incremental forward pass for a batch of processes."""
        # Get hidden states for each process, initializing if missing
        h_0, c_0 = [], []
        for nid, feats in zip(process_ids, process_features):
            if nid not in self.hidden_states:
                # base_state = torch.cat([feats.reshape(1, 1, -1), torch.zeros((1, 1, self.edge_dim), device=feats.device)], dim=-1)
                # self.hidden_states[nid] = (
                #     base_state, # h_0
                #     base_state, # c_0
                # )
                self.hidden_states[nid] = (
                    torch.zeros(1, 1, self.rnn.hidden_size, device=feats.device),  # h_0
                    torch.zeros(1, 1, self.rnn.hidden_size, device=feats.device),  # c_0
                )
            h, c = self.hidden_states[nid]
            h_0.append(h)
            c_0.append(c)

        # Stack hidden states into batch format
        h_0 = torch.cat(h_0, dim=1)  # (1, batch_size, out_dim)
        c_0 = torch.cat(c_0, dim=1)  # (1, batch_size, out_dim)

        # Run RNN forward pass
        output, (new_h, new_c) = self.rnn(emb_sequence.unsqueeze(1), (h_0, c_0))

        # Update hidden states
        for i, nid in enumerate(process_ids):
            self.hidden_states[nid] = (
                new_h[:, i : i + 1, :].detach(),
                new_c[:, i : i + 1, :].detach(),
            )

        return output[:, -1, :]  # Return last output embedding

    def reset_state(self):
        self.hidden_states = {}
        self.embedding_store = {}
        if isinstance(self.encoder, TGNEncoder):
            self.encoder.reset_state()


class EntityLinearEncoder(nn.Module):
    def __init__(self, in_dim, out_dim, encoder, activation=False):
        super().__init__()
        self.encoder = encoder
        self.activation = activation

        # One linear per entity type
        # May be HeteroDictLinear for simplicity
        self.linears = nn.ModuleList(
            [
                nn.Linear(in_dim, out_dim),
                nn.Linear(in_dim, out_dim),
                nn.Linear(in_dim, out_dim),
            ]
        )

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
