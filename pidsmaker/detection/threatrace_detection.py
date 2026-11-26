"""Detection rule of ThreaTrace (Wang et al., TIFS 2022, `test_darpatc.py`).

A node is anomalous when no submodel predicts its type with a top-1 / top-2 probability ratio of
at least `ratio_threshold`. The submodels are trained by
`training_methods.threatrace_training`.
"""

import math

import numpy as np
import torch

from pidsmaker.detection.node_scoring import NodeScorer


def threatrace_log_ratio(log_probs, node_type):
    """log(p_true / max_{c != true} p_c) per node, from log-probabilities (N, C).

    ThreaTrace explains a node when its type is predicted with a top-1 / top-2 probability
    ratio of at least `thre`. This is equivalent to `ratio >= thre` with the ratio above: when
    the prediction is right it is the top-1 / top-2 ratio, and when it is wrong it is <= 1.
    """
    log_p_true = log_probs.gather(1, node_type.unsqueeze(1)).squeeze(1)
    others = log_probs.scatter(1, node_type.unsqueeze(1), float("-inf"))
    return log_p_true - others.max(dim=1).values


def threatrace_anomaly_score(outs, node_type, ratio_threshold):
    """ThreaTrace's anomaly score, positive iff the node is anomalous.

    A node is anomalous when no submodel predicts its type with a probability ratio of at least
    `ratio_threshold` (test_darpatc.py). The score is
    `log(ratio_threshold) - max_k log(p_true / max_{c != true} p_c)` over the submodels `outs`
    (log-probabilities), so that anomalous nodes are exactly those with a score > 0, and nodes
    whose type is the least well predicted rank first.
    """
    best = torch.stack([threatrace_log_ratio(out, node_type) for out in outs]).max(dim=0).values
    return math.log(ratio_threshold) - best


class ThreaTraceScorer(NodeScorer):
    """Writes the `threatrace_score` of the nodes, thresholded by `threshold_method: threatrace`."""

    score_column = "threatrace_score"

    def columns(self, model, data, results, split, node, loss):
        # One output per submodel (ThreaTrace's multi-model scheme), or the single model's
        outs = results.get("outs", [results["out"]])
        score = threatrace_anomaly_score(
            outs, data.node_type.argmax(1), self.cfg.training.threatrace.ratio_threshold
        )
        return {
            "node": node,
            "loss": loss,
            "threatrace_score": score.cpu().numpy().astype(float),
        }

    def node_prediction(self, values, thr):
        # The score is positive iff no submodel explains the node's type in that
        # time window (see `threatrace_anomaly_score`): a node is detected if it is
        # anomalous in any time window, the one with the highest score being kept.
        scores = values["threatrace_score"]
        best = int(np.argmax(scores))
        pred_score = scores[best]
        return pred_score, pred_score > thr, values["tw"][best]
