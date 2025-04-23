import torch
import torch.nn as nn


class GLSTM(nn.Module):
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
        GLSTM Class initialiser
        Takes integer sizes of input features, output features and set up model linear network layers
        """
        super(GLSTM, self).__init__()
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
        """Compute tree GLSTM given a batch."""
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
