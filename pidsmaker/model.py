"""PIDS Model combining encoder with multiple training objectives.

The Model class orchestrates encoder execution and applies multiple objectives
(reconstruction, prediction, contrastive learning) for joint training. Supports
few-shot learning mode and MC Dropout uncertainty quantification.
"""

import copy

import numpy as np
import torch
import torch.nn as nn

from pidsmaker.encoders import TGNEncoder
from pidsmaker.experiments.uncertainty import activate_dropout_inference


class Model(nn.Module):
    """Main PIDS model combining graph encoder with training objectives.

    Attributes:
        encoder: Neural network encoder (SAGE, GAT, TGN, etc.)
        objectives: List of training objectives (reconstruction, prediction, etc.)
        objective_few_shot: Few-shot detection objective (optional)
        device: PyTorch device (cuda/cpu)
        few_shot_mode: Whether currently in few-shot fine-tuning mode
    """

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
        """Generate node embeddings for batch using encoder.

        Args:
            batch: Data batch with edge_index, node features, timestamps, etc.
            inference: If True, run in inference mode (no gradients)
            **kwargs: Additional arguments passed to encoder

        Returns:
            tuple: (h, h_src, h_dst) where:
                - h: All node embeddings (N, d) or tuple of (src_nodes, dst_nodes)
                - h_src: Source node embeddings for edges (E, d)
                - h_dst: Destination node embeddings for edges (E, d)
        """
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
        """Forward pass: embed nodes and compute loss/scores across all objectives.

        Args:
            batch: Data batch with graph structure and features
            inference: If True, return anomaly scores; if False, return training loss
            validation: If True, compute validation metrics

        Returns:
            dict: Contains 'loss' key with:
                - Training mode: scalar loss (sum of all objective losses)
                - Inference mode: per-edge anomaly scores (E,)
        """
        train_mode = not inference

        # Apply masking to input features if training with reconstruction objective
        mask_nodes = None
        x_for_encoding = getattr(batch, "x", None)

        if train_mode and x_for_encoding is not None:
            # Check if we have a GMAEFeatReconstruction objective
            for objective in self.objectives:
                if hasattr(objective, "mask_input"):
                    x_for_encoding, mask_nodes = objective.mask_input(x_for_encoding)
                    break  # Only mask once

        with torch.set_grad_enabled(train_mode):
            # Pass masked input to encoder if masking was applied
            if x_for_encoding is not None and hasattr(batch, "x"):
                # Temporarily replace batch.x with masked version
                original_x = batch.x
                batch.x = x_for_encoding
                h, h_src, h_dst = self.embed(batch, inference=inference)
                batch.x = original_x  # Restore original for objectives
            else:
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
                    mask_nodes=mask_nodes,  # Pass mask_nodes to objectives
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
        """Get average validation score across all objectives.

        Returns:
            float: Mean validation score (average precision)
        """
        return np.mean([d.get_val_score() for d in self.objectives])

    def to_device(self, device):
        """Move model and associated components to specified device.

        Handles special device migration for TGN memory and graph reindexer.

        Args:
            device: Target PyTorch device

        Returns:
            Model: Self for chaining
        """
        if self.device == device:
            return self

        for objective in self.objectives:
            objective.graph_reindexer.to(device)

        if isinstance(self.encoder, TGNEncoder):
            self.encoder.to_device(device)

        self.device = device
        return self.to(device)

    def eval(self):
        """Set model to evaluation mode.

        Overrides default eval() to keep dropout active for MC Dropout uncertainty.
        """
        super().eval()

        if self.is_running_mc_dropout:
            activate_dropout_inference(self)

    def gather_h(self, batch, res):
        """Extract source and destination node embeddings from encoder output.

        Handles different encoder output formats:
        - Single tensor h (N, d): index by edge_index to get h_src, h_dst
        - Tuple (h_src_nodes, h_dst_nodes): separate embeddings for src/dst nodes
        - Pre-computed (h_src, h_dst): already indexed for edges

        Args:
            batch: Data batch with edge_index
            res: Encoder output dict with 'h', optionally 'h_src', 'h_dst'

        Returns:
            tuple: (h, h_src, h_dst) node embeddings
        """
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
        """Switch between self-supervised pretraining and few-shot fine-tuning.

        When entering few-shot mode:
        - Optionally freezes encoder weights
        - Replaces pretraining objectives with few-shot detection objective

        When exiting few-shot mode:
        - Unfreezes encoder
        - Restores pretraining objectives

        Args:
            do: True to enter few-shot mode, False to exit
        """
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
        """Reset encoder state (e.g., TGN memory) between evaluation windows."""
        if hasattr(self.encoder, "reset_state"):
            self.encoder.reset_state()
