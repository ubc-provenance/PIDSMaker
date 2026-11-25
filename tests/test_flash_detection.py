"""FLASH's detection rule follows the original code (Flash-IDS notebooks).
Runs without a database:

    python -m pytest tests/test_flash_detection.py
"""

import pytest
import torch

from pidsmaker.detection.flash_detection import flash_anomaly_score, flash_confidence, flash_flags

pytestmark = pytest.mark.no_database


def original_flags(probs, y, thr):
    """Test cell of the THEIA notebook, for one model: True for the flagged nodes."""
    sorted, indices = probs.sort(dim=1, descending=True)
    conf = (sorted[:, 0] - sorted[:, 1]) / sorted[:, 0]
    conf = (conf - conf.min()) / conf.max()
    pred = indices[:, 0]
    cond = (pred == y) & (conf > thr)
    return ~cond


def random_probs(n, c, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.softmax(3 * torch.randn(n, c, generator=g), dim=1), torch.randint(0, c, (n,), generator=g)


@pytest.mark.parametrize("seed", range(5))
def test_flags_match_original_rule(seed):
    probs, y = random_probs(500, 4, seed)
    for thr in [0.0, 0.3, 0.53, 0.9]:
        score, _ = flash_anomaly_score(probs, y)
        expected = original_flags(probs, y, thr)
        # Ignore nodes whose confidence is within float error of the threshold
        conf, _ = flash_confidence(probs)
        safe = (conf - thr).abs() > 1e-5
        assert torch.equal(flash_flags(score, thr)[safe], expected[safe])


def test_score_orders_nodes_by_anomaly():
    # Correct and confident < correct and unsure < wrong and unsure < wrong and confident. The
    # confidence is min-max normalized over the batch, so the last node gets a confidence of 0
    probs = torch.tensor([[0.98, 0.01, 0.01], [0.5, 0.45, 0.05], [0.5, 0.45, 0.05], [0.98, 0.01, 0.01], [0.34, 0.33, 0.33]])
    y = torch.tensor([0, 0, 1, 1, 0])
    score, correct = flash_anomaly_score(probs, y)
    assert correct.tolist() == [True, True, False, False, True]
    assert score[0] < score[1] < score[2] < score[3]
    # The confidence is a probability gap, never negative (log-probabilities made it negative)
    conf, _ = flash_confidence(probs)
    assert (conf >= 0).all() and (conf <= 1).all()
