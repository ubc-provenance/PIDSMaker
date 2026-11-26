"""Detection rule of FLASH (Rehman et al., IEEE S&P 2024).

In the original code (Flash-IDS, E3 notebooks), the GNN outputs softmax probabilities over the
node types, and the confidence of a node is the relative gap between its two highest
probabilities, min-max normalized over the batch:

    conf = (p1 - p2) / p1;  conf = (conf - conf.min()) / conf.max()

A node is benign if the GNN predicts its type correctly with a confidence above a threshold
(0.53 on THEIA), and is flagged otherwise.

PIDSMaker ranks nodes by a score, so the rule is turned into an anomaly score that increases
when the node looks less benign: `1 - conf` for a correct prediction and `1 + conf` for a wrong
one (a confidently wrong prediction is the most anomalous). A node is flagged by the original
rule iff its score is at least `1 - threshold`.
"""

import numpy as np
import torch

from pidsmaker.detection.node_scoring import NodeScorer
from pidsmaker.objectives.predict_node_type import NODE_LOSS_MASK_KEY


def flash_confidence(probs, eps=1e-6):
    """Normalized confidence of each row of `probs` (N, C) and the predicted classes."""
    top2, indices = probs.topk(2, dim=1)
    conf = (top2[:, 0] - top2[:, 1]) / (top2[:, 0] + eps)
    conf_max = conf.max() if conf.numel() else conf.new_zeros(())
    if conf_max > 0:
        conf = (conf - conf.min()) / conf_max
    return conf.clamp(min=0), indices[:, 0]


def flash_anomaly_score(probs, node_type_idx):
    """Anomaly score of each node (see the module docstring) and whether its type is predicted
    correctly."""
    conf, pred = flash_confidence(probs)
    correct = pred == node_type_idx
    score = torch.where(correct, 1 - conf, 1 + conf)
    return score, correct


def flash_flags(score, threshold):
    """Original rule: a node is flagged unless it is correctly predicted with conf > threshold."""
    return score >= 1 - threshold


def flash_drop_learned(probs, node_type_idx, drop_conf):
    """Training nodes that FLASH removes after an epoch: those whose type is predicted correctly,
    or predicted with a confidence of at least `drop_conf` (`(pred == y) | (conf >= 0.9)` in the
    CADETS and THEIA notebooks)."""
    conf, pred = flash_confidence(probs)
    return (pred == node_type_idx) | (conf >= drop_conf)


class FlashSnapshotEnsemble:
    """The sequence of GNN snapshots of FLASH.

    In the original code, the GNN is trained for N rounds (22 on CADETS, 20 on THEIA). Each
    round trains on the training nodes that are still misclassified, then removes the nodes it
    learned (see `flash_drop_learned`), and saves the model. At test time, a node is benign if
    any of the N snapshots predicts its type correctly with a confidence above the threshold.

    Here, a round is an epoch over the training graphs: the remaining nodes of each graph are
    stored in the CPU graph (`NODE_LOSS_MASK_KEY`) and only them contribute to the loss, and the
    score of a node at inference is the minimum of its `flash_anomaly_score` over the snapshots
    of the epochs trained so far (so it is flagged iff no snapshot is confidently correct).
    """

    def __init__(self, drop_conf):
        self.drop_conf = drop_conf
        self.states = []

    @staticmethod
    def _state(model):
        return {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def after_epoch(self, model, train_data, device):
        from pidsmaker.utils.data_utils import batch_to_device
        from pidsmaker.utils.utils import log

        self.states.append(self._state(model))
        was_training = model.training
        model.eval()
        remaining, total = 0, 0
        for dataset in train_data:
            for g in dataset:
                mask = getattr(g, NODE_LOSS_MASK_KEY, None)
                if mask is None or mask.any():
                    # The node features and types may only be built on the device copy
                    batch = batch_to_device(g, device)
                    out = model(batch, inference=True)["out"]
                    drop = flash_drop_learned(out.exp(), batch.node_type.argmax(1), self.drop_conf)
                    mask = ~drop.cpu() if mask is None else mask & ~drop.cpu()
                setattr(g, NODE_LOSS_MASK_KEY, mask)
                remaining += int(mask.sum())
                total += mask.numel()
        model.train(was_training)
        log(f"FLASH round {len(self.states) - 1}: {remaining}/{total} training nodes still misclassified")

    @torch.no_grad()
    def score(self, model, data, current_score):
        """Minimum anomaly score over the snapshots. `current_score` is the score given by the
        current weights of `model`, which are restored."""
        if len(self.states) <= 1:
            return current_score
        current = self._state(model)
        node_type_idx = data.node_type.argmax(1)
        best = current_score
        for state in self.states[:-1]:  # the last snapshot holds the current weights
            model.load_state_dict(state)
            out = model(data, inference=True)["out"]
            best = torch.minimum(best, flash_anomaly_score(out.exp(), node_type_idx)[0])
        model.load_state_dict(current)
        return best


def flash_snapshot_ensemble(cfg):
    """The `FlashSnapshotEnsemble` trained by the training loop, or None when the model is not
    FLASH's with snapshots (`predict_node_type.flash_snapshots` and `threshold_method: flash`)."""
    node_type_cfg = getattr(cfg.training.decoder, "predict_node_type", None)
    if (
        cfg._is_node_level
        and cfg.evaluation.node_evaluation.threshold_method == "flash"
        and node_type_cfg is not None
        and node_type_cfg.flash_snapshots
    ):
        return FlashSnapshotEnsemble(drop_conf=node_type_cfg.flash_drop_conf)
    return None


class FlashScorer(NodeScorer):
    """Writes the `flash_score` of the nodes, thresholded by `threshold_method: flash`, and
    whether their type is predicted correctly. With snapshots, the score is the minimum over
    the snapshots of the model's `flash_ensemble`."""

    score_column = "flash_score"

    def columns(self, model, data, results, split, node, loss):
        # `out` holds log-probabilities: the confidence of FLASH is computed on probabilities
        score, correct = flash_anomaly_score(results["out"].exp(), data.node_type.argmax(1))
        ensemble = getattr(model, "flash_ensemble", None)
        if ensemble is not None:
            score = ensemble.score(model, data, score)
        return {
            "node": node,
            "loss": loss,
            "flash_score": score.cpu().numpy().astype(float),
            "correct_pred": correct.long().cpu().numpy(),
        }

    def node_prediction(self, values, thr):
        # The node is flagged if, in one of its time windows, it is not predicted correctly
        # with a confidence above the threshold
        scores = values["flash_score"]
        tw_max = int(np.argmax(scores))
        pred_score = scores[tw_max]
        if flash_flags(np.float64(pred_score), thr):
            return pred_score, True, values["tw"][tw_max]
        return pred_score, False, None
