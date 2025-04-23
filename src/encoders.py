from provnet_utils import *
from config import *
import torch.nn as nn
from torch_geometric.nn import SAGEConv, GATConv, MessagePassing, GINConv, GINEConv


class GRU(nn.Module):
    def __init__(self, x_dim, h_dim, device, hidden_units=1):
        """
        x_dim : int
            The input dimension
        h_dim : int
            The hidden dimension
        z_dim : int
            The output dimension
        hidden_units : int
            How many GRUs to use. 1 is usually sufficient to avoid
            loss of generality
        """
        super(GRU, self).__init__()

        self.rnn = nn.GRU(x_dim, h_dim, num_layers=hidden_units)
        self.hidden_units = hidden_units

        # self.drop = nn.Dropout(0.25)

        self.h_dim = h_dim
        self.device = device

        self.reset_state()

    def reset_state(self):
        self.hidden = self._init_hidden()

    def detach_state(self):
        self.hidden = self.hidden.detach()

    def _init_hidden(self):
        return torch.zeros(self.hidden_units, self.h_dim, requires_grad=True).to(
            self.device
        )

    def forward(self, xs, h0=None, include_h=False):
        """
        Forward method for GRU

        xs : torch.Tensor
            The T x N x X_dim input of node embeddings
        h0 : torch.Tensor
            A hidden state for the GRU
        include_h : bool
            If true, return hidden state as well as output
        """
        # xs = self.drop(xs)
        xs, self.hidden = self.rnn(xs, self.hidden)

        if not include_h:
            return xs

        return xs, self.hidden

class GraphAttentionEmbedding(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, edge_dim, dropout, activation, num_heads, concat):
        super(GraphAttentionEmbedding, self).__init__()
        
        conv2_in_dim = hid_dim * num_heads if concat else hid_dim
        self.conv = TransformerConv(in_dim, hid_dim, heads=num_heads, dropout=dropout, edge_dim=edge_dim, concat=concat)
        self.conv2 = TransformerConv(conv2_in_dim, out_dim, heads=1, concat=False, dropout=dropout, edge_dim=edge_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, x, edge_index, edge_feats=None, **kwargs):
        x = self.activation(self.conv(x, edge_index, edge_feats))
        x = self.dropout(x)
        x = self.conv2(x, edge_index, edge_feats)
        return x
    
class TGNEncoder(nn.Module):
    # Code adapted from https://github.com/pyg-team/pytorch_geometric/blob/master/examples/tgn.py
    def __init__(
        self,
        encoder,
        memory,
        neighbor_loader,
        time_encoder,
        in_dim,
        memory_dim,
        use_node_feats_in_gnn,
        graph_reindexer,
        edge_features,
        device,
        use_memory,
        num_nodes,
        use_time_enc,
        edge_dim,
        use_time_order_encoding,
        tgn_neighbor_n_hop,
        use_buggy_orthrus_TGN,
    ):
        super(TGNEncoder, self).__init__()
        self.encoder = encoder
        self.memory = memory
        self.neighbor_loader = neighbor_loader
        self.device = device
        self.assoc = torch.empty(num_nodes, dtype=torch.long, device=device)
        self.node_feat_cache = torch.empty((num_nodes, in_dim), device=device)
        
        self.edge_features = edge_features
        if "time_encoding" in self.edge_features:
            self.time_encoder = time_encoder

        self.graph_reindexer = graph_reindexer
        self.use_memory = use_memory
        self.use_time_enc = use_time_enc
        
        self.use_node_feats_in_gnn = use_node_feats_in_gnn
        if not self.use_memory or self.use_node_feats_in_gnn:
            self.src_linear = nn.Linear(in_dim, memory_dim)
            self.dst_linear = nn.Linear(in_dim, memory_dim)
        
        self.use_time_order_encoding = use_time_order_encoding
        if self.use_time_order_encoding:
            self.gru = GRU(edge_dim, edge_dim, device)
            
        self.tgn_neighbor_n_hop = tgn_neighbor_n_hop
        self.use_buggy_orthrus_TGN = use_buggy_orthrus_TGN

    def forward(self, edge_index, t, msg, x, full_data, inference=False, **kwargs):
        # NOTE: full_data is the full list of all edges in the entire dataset (train/val/test)
        
        src, dst = edge_index
        x_src, x_dst = x
        batch_edge_index = edge_index.clone()
        
        # Get the N last edges connecting nodes in n_id up to n-hop
        n_id = edge_index.unique()
        for _ in range(self.tgn_neighbor_n_hop):
            n_id, edge_index, e_id = self.neighbor_loader(n_id)

        self.assoc[n_id] = torch.arange(n_id.size(0), device=self.device)

        x_proj = None
        if (not self.use_memory) or self.use_node_feats_in_gnn:
            if self.use_buggy_orthrus_TGN:
                # This method is buggy because we only take the node features from the current batch, but the edge index also have the last
                # neighbors in int and we don't have their features. This means basically that the GNN aggregates mostly zeros, except if last 
                # neighbors are within the current batch.
                
                # x_src, x_dst = self.graph_reindexer.node_features_reshape(batch_edge_index, x_src, x_dst, max_num_node=n_id.max(), x_is_tuple=True)
                # x_proj = self.src_linear(x_src[n_id]) + self.dst_linear(x_dst[n_id])

                (x_src, x_dst), *_ = self.graph_reindexer._reindex_graph(batch_edge_index, x_src, x_dst, max_num_node=n_id.size(0), x_is_tuple=True)
                x_proj = self.src_linear(x_src) + self.dst_linear(x_dst)
            else:
                x, _, feats_n_id = self.graph_reindexer._reindex_graph(batch_edge_index, x_src, x_dst, x_is_tuple=False)
                self.node_feat_cache[feats_n_id] = x
                
                # Single linear
                # x_proj = self.src_linear(self.node_feat_cache[feats_n_id])
                
                # Src-dst linear
                x_src = torch.zeros((n_id.size(0), x.shape[1]), device=x.device)
                x_dst = x_src.clone()
                src_id, dst_id = edge_index[0].unique(), edge_index[1].unique()
                x_src[src_id] = self.node_feat_cache[n_id[src_id]]
                x_dst[dst_id] = self.node_feat_cache[n_id[dst_id]]
                x_proj = self.src_linear(x_src) + self.dst_linear(x_dst)

        if self.use_memory:
            # Get updated memory of all nodes involved in the computation.
            h, last_update = self.memory(n_id)
            # Adds the node features to the memory to avoid zero-vector for first-seen nodes
            if self.use_node_feats_in_gnn:
                h = h + x_proj
        else:
            h = x_proj
        
        # Edge features
        edge_feats = []
        if "edge_type" in self.edge_features:
            curr_msg = full_data.edge_type[e_id.cpu()].to(self.device)
            edge_feats.append(curr_msg)
        if "msg" in self.edge_features:
            # TGN uses the whole msg as edge feature (this is used by default in the TGN PyG implementation)
            curr_msg = full_data.msg[e_id.cpu()].to(self.device)
            edge_feats.append(curr_msg)
        if "time_encoding" in self.edge_features: # or self.use_time_enc
            if not self.use_memory:
                last_update = self.memory.get_last_update(n_id)
            curr_t = full_data.t[e_id.cpu()].to(self.device)
            rel_t = last_update[edge_index[0]] - curr_t
            rel_t_enc = self.time_encoder(rel_t.to(h.dtype))
            edge_feats.append(rel_t_enc)
        edge_feats = torch.cat(edge_feats, dim=-1) if len(edge_feats) > 0 else None
        
        if self.use_time_order_encoding:
            if len(edge_feats) > 0:  # GRU doesn't work with empty sequences
                edge_feats = self.gru(edge_feats)
                self.gru.detach_state()
        
        h = self.encoder(h, edge_index, edge_feats=edge_feats)

        h_src = h[self.assoc[src]]
        h_dst = h[self.assoc[dst]]

        # Update memory and neighbor loader with ground-truth state.
        if self.use_memory or self.use_time_enc:
            self.memory.update_state(src, dst, t, msg)
        self.neighbor_loader.insert(src, dst)
        
        # Detaching memory is only useful for backprop in training
        if self.use_memory and not inference:
            self.memory.detach()
        
        return h_src, h_dst

    def reset_state(self):
        if self.use_memory or self.use_time_enc: # if memory is used
            self.memory.reset_state()  # Flushes memory.
        self.neighbor_loader.reset_state()  # Empties the graph.
        
    def to_device(self, device):
        self.assoc = self.assoc.to(device)
        self.graph_reindexer.to(device)
        self.neighbor_loader.to(device)
        self.device = device
        if hasattr(self, "gru"):
            self.gru.device = device
        return self

class SAGE(nn.Module):
    def __init__(self, in_dim,  hid_dim, out_dim, activation, dropout):
        super(SAGE, self).__init__()
        self.conv1 = SAGEConv(in_dim, hid_dim, normalize=False)
        self.conv2 = SAGEConv(hid_dim, out_dim, normalize=False)
        self.activation = activation
        self.drop = nn.Dropout(dropout)
    
    def forward(self, x, edge_index, **kwargs):
        x = self.activation(self.conv1(x, edge_index))
        x = self.drop(x)
        x = self.conv2(x, edge_index)
        return x

class LSTM(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 cell_clip=None,
                 type_specific_decoding=False,
                 exclude_file=True,
                 exclude_ip=True,
                 typed_hidden_rep=False,
                 edge_dim=None,
                 full_param=False,
                 num_edge_type = 10
                 ):
        """
        LSTM Class initialiser
        Takes integer sizes of input features, output features and set up model linear network layers
        """
        super(LSTM, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self._cell_clip = cell_clip  # clip cell to avoid value overflow
        self._type_specific_decoding = type_specific_decoding  # decode each type of nodes separately
        self._exclude_file = exclude_file  # if type_specific_decoding is true, exclude file node decoding
        self._exclude_ip = exclude_ip  # if type_specific_decoding is true, exclude ip node decoding
        self._typed_hidden_rep = typed_hidden_rep  # include edge type embedding in node hidden representation
        self._edge_dim = edge_dim if edge_dim is not None else in_features  # TODO: edge_dim defaults to in_features
        # TODO: full parametrization implementation is incomplete due to runtime in-place operation error
        self._full_param = full_param  # use full parametrization instead of typed hidden representation
        self._num_edge_type = num_edge_type  # TODO: number of edge type defaults to 15


        self.W_iou = nn.Linear(self.in_features, 3 * self.out_features)
        if self._typed_hidden_rep:
            self.U_iou = torch.randn(self.out_features, 3 * self.out_features, self._edge_dim, requires_grad=True)
        else:
            if self._full_param:
                self.U_ious = nn.ModuleList()
                for _ in range(self._num_edge_type):
                    self.U_ious.append(nn.Linear(self.out_features, 3 * self.out_features))
            else:
                self.U_iou = nn.Linear(self.out_features, 3 * self.out_features)

        self.W_f = nn.Linear(self.in_features, self.out_features)
        if self._typed_hidden_rep:
            self.U_f = torch.randn(self.out_features, self.out_features, self._edge_dim, requires_grad=True)
        else:
            if self._full_param:
                self.U_fs = nn.ModuleList()
                for _ in range(self._num_edge_type):
                    self.U_fs.append(nn.Linear(self.out_features, self.out_features))
            else:
                self.U_f = nn.Linear(self.out_features, self.out_features)


    def forward(self, x, edge_index, edge_feats, edge_types,**kwargs):
        """Compute tree LSTM given a batch.
        """
        batch_size = x.shape[0]

        adjacency_list= torch.transpose(edge_index,0,1)
        features = x

        node_order = torch.zeros(batch_size)
        edge_order = torch.zeros(adjacency_list.shape[0])

        for i in range(adjacency_list.shape[0]):
            node_order[adjacency_list[i,0]] = node_order[adjacency_list[i,0]] + 1
            edge_order[i] = node_order[adjacency_list[i,0]]
    
        # Retrieve device the model is currently loaded on to generate h, c, and h_sum result buffers
        device = next(self.parameters()).device
        tmp = torch.arange(1.0, 11.0).reshape(10,1).to(device)
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
            self._run_lstm(n, h, c, h_sum, features, node_order, adjacency_list, edge_order, edge_feats, edge_types)


        assert not (torch.isnan(h).any() or torch.isinf(h).any()), "h has inf or nan"
        assert not (torch.isnan(c).any() or torch.isinf(c).any()), "c has inf or nan"

        return h


    def _run_lstm(self,
                  iteration,
                  h,
                  c,
                  h_sum,
                  features,
                  node_order,
                  adjacency_list,
                  edge_order,
                  edge_features,
                  edge_types):
        """Helper function to evaluate all tree nodes currently able to be evaluated.
        """
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
                u_h_sum = torch.zeros(h_sum[node_mask, :, :].shape[0], 3 * self.out_features, self._edge_dim)
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
        assert not (torch.isnan(c[node_mask, :]).any() or torch.isinf(c[node_mask, :]).any()), "c mask has inf or nan"

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
            assert not (torch.isnan(f).any() or torch.isinf(f).any()),\
                "f has inf or nan at iteration {}".format(iteration)

            # fc is a tensor of size e x M
            # if (child_c != child_c).any() and iteration == 1:
            #     print("Child_c: {}".format(child_c))
            fc = torch.mul(f, child_c)
            assert not (torch.isnan(fc).any() or torch.isinf(fc).any()),\
                "fc has inf or nan at iteration {}".format(iteration)

            # if (fc != fc).any():
            #     print("Invalid fc at iteration {}".format(iteration))

            for cindex, pindex in enumerate(parent_indexes):
                c[pindex, :] += fc[cindex, :]
                if self._cell_clip is not None:
                    c[pindex, :] = torch.clamp_(c[pindex, :], -self._cell_clip, self._cell_clip)

        h[node_mask, :] = torch.mul(o, torch.tanh(c[node_mask]))

class RcaidMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(RcaidMLP, self).__init__()
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
        self.gat3 = GATConv(hid_dim * num_heads, out_dim, heads=1, concat=False)  # Output is not concatenated
        self.mlp = RcaidMLP(hid_dim * num_heads + out_dim, out_dim)  # Input is concatenated
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

        return out

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
        return x
    
class LinearEncoder(nn.Module):    
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, out_dim)
    
    def forward(self, x, *args, **kwargs):
        if isinstance(x, tuple):
            return self.lin1(x[0]), self.lin1(x[1])
        return self.lin1(x)

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

        # hid_dim = hid_dim * 2

        nn1 = nn.Sequential(Linear(in_dim, hid_dim), nn.ReLU(), Linear(hid_dim, hid_dim))
        nn2 = nn.Sequential(Linear(hid_dim, hid_dim), nn.ReLU(), Linear(hid_dim, hid_dim))
        
        if edge_dim is None:
            self.conv1 = GINConv(nn1)
            self.conv2 = GINConv(nn2)
        else:
            self.conv1 = GINEConv(nn1, edge_dim=edge_dim)
            self.conv2 = GINEConv(nn2, edge_dim=edge_dim)

        self.fc1 = Linear(hid_dim, hid_dim)
        self.fc2 = Linear(hid_dim, out_dim)

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
        return x  # TODO: test to add tanh(), as in EULER
    
class MagicGAT(nn.Module):
    def __init__(self,
                 n_dim,
                 hidden_dim,
                 out_dim,
                 n_layers,
                 n_heads,
                 feat_drop=0.1,
                 attn_drop=0.0,
                 negative_slope=0.2,
                 residual=False,
                 activation=None,
                 concat_out=False
                 ):
        super(MagicGAT, self).__init__()
        self.out_dim = out_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.concat_out = concat_out
        self.gats = nn.ModuleList()

        # First layer
        self.gats.append(GATConv(n_dim, hidden_dim, heads=n_heads, concat=self.concat_out,
                                 dropout=attn_drop, negative_slope=negative_slope))

        # Hidden layers
        for _ in range(1, n_layers - 1):
            self.gats.append(GATConv(hidden_dim * n_heads, hidden_dim, heads=n_heads, concat=self.concat_out,
                                     dropout=attn_drop, negative_slope=negative_slope))

        # Last layer
        self.gats.append(GATConv(hidden_dim * n_heads, out_dim, heads=1, concat=self.concat_out,
                                 dropout=attn_drop, negative_slope=negative_slope))

        self.dropout = nn.Dropout(feat_drop)
        self.activation = activation
        self.residual = residual
        self.last_linear = nn.Linear(hidden_dim * n_heads, out_dim)

    def forward(self, x, edge_index, return_hidden=False, **kwargs):
        hidden_list = []
        h = x

        # Forward through GAT layers
        for layer in range(self.n_layers):
            h_in = h  # For residual connection
            h = self.dropout(h)
            h = self.gats[layer](h, edge_index)

            if self.residual and layer > 0:  # Adding residual connection if enabled
                if layer == self.n_layers -1:
                    h_in = self.last_linear(h_in)
                h = h + h_in

            if self.activation:
                h = self.activation(h)

            hidden_list.append(h)

        if return_hidden:
            return h, hidden_list
        else:
            return h

    def reset_classifier(self, num_classes):
        self.head = nn.Linear(self.out_dim, num_classes)
