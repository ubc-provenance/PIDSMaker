import copy

import numpy as np
import torch
import torch.nn as nn

from pidsmaker.encoders import TGNEncoder
from pidsmaker.experiments.uncertainty import activate_dropout_inference


class Model(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        objectives: list[nn.Module],
        objective_few_shot: nn.Module,
        device,
        is_running_mc_dropout,
        use_few_shot,
        freeze_encoder,
        fuse_duplicate_edges_training,
        is_hybrid_loss,
    ):
        super(Model, self).__init__()

        self.encoder = encoder
        self.objectives = nn.ModuleList(objectives)
        self.device = device
        self.is_running_mc_dropout = is_running_mc_dropout

        self.objective_few_shot = objective_few_shot
        self.use_few_shot = use_few_shot
        self.few_shot_mode = False
        self.freeze_encoder = freeze_encoder
        self.fuse_duplicate_edges_training = fuse_duplicate_edges_training
        self.is_hybrid_loss = is_hybrid_loss

    def embed(self, batch, inference=False, **kwargs):
        train_mode = not inference
        edge_index = batch.edge_index
        with torch.set_grad_enabled(train_mode):
            res = self.encoder(
                edge_index=edge_index,
                t=batch.t,
                x_src=getattr(batch, "x_src", None),
                x_dst=getattr(batch, "x_dst", None),
                msg=getattr(batch, "msg", None),
                edge_feats=getattr(batch, "edge_feats", None),
                inference=inference,
                edge_types=batch.edge_type,
                node_type_src=getattr(batch, "node_type_src", None),
                node_type_dst=getattr(batch, "node_type_dst", None),
                batch=batch,
                # Reindexing attr
                x=getattr(batch, "x", None),
                original_n_id=getattr(batch, "original_n_id", None),
                node_type=getattr(batch, "node_type", None),
                node_type_argmax=getattr(batch, "node_type_argmax", None),
                # Hetero attr
                edge_index_dict=getattr(batch, "edge_index_dict", None),
                x_dict=getattr(batch, "x_dict", None),
            )
        h, h_src, h_dst = self.gather_h(batch, res)
        return h, h_src, h_dst

    def forward(self, batch, inference=False, validation=False):
        train_mode = not inference

        with torch.set_grad_enabled(train_mode):
            h, h_src, h_dst = self.embed(batch, inference=inference)

            if self.fuse_duplicate_edges_training:
                if not (inference or validation):
                    edge_index = batch.edge_index  # Shape: [2, num_edges]
                    edge_type = batch.edge_type  # Shape: [num_edges, d], one-hot encoded
                    num_edges, d = edge_type.shape

                    # Create edge tuple and find unique edges
                    edges_tuple = torch.stack([edge_index[0], edge_index[1]], dim=1)  # Shape: [num_edges, 2]
                    unique_edges, inverse_indices = torch.unique(edges_tuple, dim=0, return_inverse=True)

                    # Convert unique_edges to edge_index format
                    unique_edge_index = unique_edges.t()  # Shape: [2, num_unique_edges]

                    # Merge edge_type values for duplicate edges
                    # num_unique_edges = unique_edge_index.shape[1]
                    # new_y = torch.zeros(num_unique_edges, d, dtype=edge_type.dtype, device=edge_type.device)
                    # new_y.scatter_add_(0, inverse_indices.view(-1, 1).expand(-1, d), edge_type)
                    # new_y = (new_y > 0).float()  # Ensure one-hot

                    # Get indices of first occurrence of each unique edge
                    indices = torch.unique(inverse_indices, return_counts=False)

                    # Update batch
                    batch = batch[indices]
                    batch.edge_index = unique_edge_index
                    # batch.edge_type = new_y
                    h_src, h_dst = h_src[indices], h_dst[indices]
            
            # Train mode: loss | Inference mode: scores
            loss_or_scores = None

            for objective in self.objectives:
                results = objective(
                    h_src=h_src,  # shape (E, d)
                    h_dst=h_dst,  # shape (E, d)
                    h=h,  # shape (N, d)
                    edge_index=batch.edge_index,
                    edge_type=batch.edge_type,
                    y_edge=batch.y,
                    inference=inference,
                    x=getattr(batch, "x", None),
                    node_type=getattr(batch, "node_type", None),
                    node_type_src=getattr(batch, "node_type_src", None),
                    node_type_dst=getattr(batch, "node_type_dst", None),
                    validation=validation,
                    batch=batch,
                )
                loss = results["loss"]

                if loss_or_scores is None:
                    loss_or_scores = (
                        torch.zeros(1)
                        if train_mode
                        else torch.zeros(loss.shape[0], dtype=torch.float)
                    ).to(batch.edge_index.device)
                
                if loss.numel() != loss_or_scores.numel():
                    if self.is_hybrid_loss:
                        loss = self.node_to_edge_loss(batch, loss)
                    else:
                        raise TypeError(
                            f"Shapes of loss/score do not match ({loss.numel()} vs {loss_or_scores.numel()})"
                        )
                loss_or_scores = loss_or_scores + loss

            results["loss"] = loss_or_scores
            return results

    def get_val_ap(self):
        # If multiple objectives are used, we take the average of the val scores
        return np.mean([d.get_val_score() for d in self.objectives])

    def to_device(self, device):
        if self.device == device:
            return self

        for objective in self.objectives:
            objective.graph_reindexer.to(device)

        if isinstance(self.encoder, TGNEncoder):
            self.encoder.to_device(device)

        self.device = device
        return self.to(device)

    # override
    def eval(self):
        super().eval()

        if self.is_running_mc_dropout:
            activate_dropout_inference(self)

    def gather_h(self, batch, res):
        h = res["h"]
        h_src = res.get("h_src", None)
        h_dst = res.get("h_dst", None)

        if None in [h_src, h_dst]:
            h_src, h_dst = (
                (h[batch.edge_index[0]], h[batch.edge_index[1]])
                if isinstance(h, torch.Tensor)
                else h
            )

        return h, h_src, h_dst

    def to_fine_tuning(self, do: bool):
        if not self.use_few_shot:
            return
        if do and not self.few_shot_mode:
            if self.freeze_encoder:
                self.encoder.eval()
                for param in self.encoder.parameters():  # freeze the encoder
                    param.requires_grad = False

            # the objective is replaced by a copy of the objective_few_shot + the old objective is saved for later switch
            ssl_objective = (
                self.objectives
            )  # switch the pretext objective and fine-tuning objective
            self.objectives = copy.deepcopy(self.objective_few_shot)
            self.ssl_objective = ssl_objective
            self.few_shot_mode = True

        if not do and self.few_shot_mode:
            self.encoder.train()
            for param in self.encoder.parameters():
                param.requires_grad = True

            # the ssl objective is set back
            self.objectives = self.ssl_objective
            self.few_shot_mode = False

    def reset_state(self):
        if hasattr(self.encoder, "reset_state"):
            self.encoder.reset_state()
    
    def node_to_edge_loss(self, batch, loss):
        n_id = batch.original_n_id
        edge_index = batch.original_edge_index
        
        sorted_ids, perm = torch.sort(n_id)
        src_pos = torch.searchsorted(sorted_ids, edge_index[0])
        dst_pos = torch.searchsorted(sorted_ids, edge_index[1])

        src_idx = perm[src_pos]
        dst_idx = perm[dst_pos]

        edge_sums = loss.index_select(0, src_idx) + loss.index_select(0, dst_idx)
        
        return edge_sums
