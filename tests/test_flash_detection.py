"""FLASH's detection rule and snapshot sequence follow the original code (Flash-IDS notebooks).
Runs without a database:

    python -m pytest tests/test_flash_detection.py
"""

import pytest
import torch
import torch.nn as nn

from pidsmaker.detection.flash_detection import (
    FlashSnapshotEnsemble,
    flash_anomaly_score,
    flash_confidence,
    flash_drop_learned,
    flash_flags,
)
from pidsmaker.losses import cross_entropy
from pidsmaker.objectives.predict_node_type import NODE_LOSS_MASK_KEY, NodeTypePrediction

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


def test_drop_learned_matches_original():
    probs, y = random_probs(300, 3, 0)
    sorted, indices = probs.sort(dim=1, descending=True)
    conf = (sorted[:, 0] - sorted[:, 1]) / sorted[:, 0]
    conf = (conf - conf.min()) / conf.max()
    expected = (indices[:, 0] == y) | (conf >= 0.9)
    assert torch.equal(flash_drop_learned(probs, y, 0.9), expected)


class TinyModel(nn.Module):
    """A per-node classifier with the interface of `Model` used by the ensemble."""

    def __init__(self, num_types):
        super().__init__()
        self.lin = nn.Linear(num_types, num_types)
        self.objective = NodeTypePrediction(lambda h: h, cross_entropy, False, num_types)

    def forward(self, batch, inference=False, validation=False):
        return self.objective(self.lin(batch.x), batch.node_type, inference, batch=batch)


def make_graph(n, c, seed):
    from torch_geometric.data import Data

    g = torch.Generator().manual_seed(seed)
    y = torch.randint(0, c, (n,), generator=g)
    x = torch.nn.functional.one_hot(y, c).float() + torch.randn(n, c, generator=g)
    return Data(x=x, node_type=torch.nn.functional.one_hot(y, c).float(), edge_index=torch.zeros(2, 0, dtype=torch.long))


def test_masked_loss_uses_remaining_nodes_only():
    model = TinyModel(3)
    g = make_graph(50, 3, 0)
    mask = torch.zeros(50, dtype=torch.bool)
    mask[:10] = True
    setattr(g, NODE_LOSS_MASK_KEY, mask)
    out = model(g)
    expected = cross_entropy(model.lin(g.x[:10]), g.node_type[:10].argmax(1))
    assert out["loss"].item() == pytest.approx(expected.item())


def test_snapshot_ensemble():
    torch.manual_seed(0)
    model = TinyModel(3)
    graphs = [[make_graph(200, 3, 1), make_graph(100, 3, 2)]]
    ensemble = FlashSnapshotEnsemble(drop_conf=0.9)
    opt = torch.optim.Adam(model.parameters(), lr=0.05)

    states_scores = []
    test_g = make_graph(150, 3, 3)
    for _ in range(3):
        for g in graphs[0]:
            mask = getattr(g, NODE_LOSS_MASK_KEY, None)
            if mask is not None and not mask.any():
                continue
            opt.zero_grad()
            model(g)["loss"].backward()
            opt.step()
        before = getattr(graphs[0][0], NODE_LOSS_MASK_KEY, None)
        ensemble.after_epoch(model, graphs, torch.device("cpu"))
        after = getattr(graphs[0][0], NODE_LOSS_MASK_KEY)
        if before is not None:
            assert not (after & ~before).any()  # removed nodes never come back
        with torch.no_grad():
            out = model(test_g, inference=True)["out"]
        states_scores.append(flash_anomaly_score(out.exp(), test_g.node_type.argmax(1))[0])

    weights = {k: v.clone() for k, v in model.state_dict().items()}
    with torch.no_grad():
        current = flash_anomaly_score(model(test_g, inference=True)["out"].exp(), test_g.node_type.argmax(1))[0]
    score = ensemble.score(model, test_g, current)
    # Minimum over the snapshots, and the current weights are restored
    assert torch.allclose(score, torch.stack(states_scores).min(0).values)
    for k, v in model.state_dict().items():
        assert torch.equal(v, weights[k])
