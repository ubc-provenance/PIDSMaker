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

    def embed(self, batch, inference=False, **kwargs):
        train_mode = not inference
        edge_index = batch.edge_index
        with torch.set_grad_enabled(train_mode):
            res = self.encoder(
                edge_index=edge_index,
                t=batch.t,
                x_src=batch.x_src,
                x_dst=batch.x_dst,
                msg=batch.msg,
                edge_feats=getattr(batch, "edge_feats", None),
                inference=inference,
                edge_types=batch.edge_type,
                node_type_src=batch.node_type_src,
                node_type_dst=batch.node_type_dst,
                batch=batch,
                # Reindexing attr
                x=getattr(batch, "x", None),
                original_n_id=getattr(batch, "original_n_id", None),
                node_type=getattr(batch, "node_type", None),
                node_type_argmax=getattr(batch, "node_type_argmax", None),
            )
        h, h_src, h_dst = self.gather_h(batch, res)
        return h, h_src, h_dst

    def forward(self, batch, inference=False, validation=False):
        train_mode = not inference

        with torch.set_grad_enabled(train_mode):
            h, h_src, h_dst = self.embed(batch, inference=inference)

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
                    node_type_src=batch.node_type_src,
                    node_type_dst=batch.node_type_dst,
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
            if isinstance(h, torch.Tensor):
                # h is a single tensor with node embeddings - index by edge_index
                h_src, h_dst = h[batch.edge_index[0]], h[batch.edge_index[1]]
            elif isinstance(h, (tuple, list)):
                # h is (h_src_nodes, h_dst_nodes) with separate node embeddings - index each
                h_src, h_dst = h[0][batch.edge_index[0]], h[1][batch.edge_index[1]]
            else:
                h_src, h_dst = h

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
