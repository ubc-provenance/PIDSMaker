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

import torch


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
