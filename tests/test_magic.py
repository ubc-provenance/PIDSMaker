"""Unit tests of the MAGIC reimplementation that run without a database."""

import pytest
import torch

from pidsmaker.detection.magic_knn import MagicKNNDetector

pytestmark = pytest.mark.no_database


def _benign(n=2000, d=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, d, generator=g)


def test_knn_scores_benign_near_one_and_outliers_high():
    x_train = _benign()
    det = MagicKNNDetector(k=10).fit(x_train)
    benign_test = _benign(500, seed=1)
    outliers = torch.full((5, 8), 6.0)
    scores_benign = det.score(benign_test)
    scores_out = det.score(outliers)
    assert 0.8 < scores_benign.median().item() < 1.6
    assert scores_out.min().item() > 3 * scores_benign.max().item() / 2


def test_knn_reference_is_the_training_set_not_the_test_window():
    # A test window made only of a tight cluster of novel nodes: scored against itself (the
    # previous PIDSMaker behaviour) its nodes look typical, against the benign training set
    # (MAGIC) they are outliers.
    det = MagicKNNDetector(k=10).fit(_benign())
    g = torch.Generator().manual_seed(2)
    novel = 5.0 + 0.01 * torch.randn(300, 8, generator=g)
    assert det.score(novel).min().item() > 3.0
    self_ref = MagicKNNDetector(k=10).fit(novel)
    assert self_ref.score(novel).median().item() < 1.6


def test_knn_standardizes_with_training_statistics():
    x_train = _benign() * torch.tensor([100.0] + [1.0] * 7)
    det = MagicKNNDetector(k=5).fit(x_train)
    assert torch.allclose(det.reference.std(dim=0, unbiased=False), torch.ones(8), atol=1e-4)
    # Same scores whatever the scale of the input space
    x_test = _benign(100, seed=3)
    det2 = MagicKNNDetector(k=5).fit(x_train * 7.0)
    assert torch.allclose(det.score(x_test), det2.score(x_test * 7.0), rtol=1e-3, atol=1e-4)


def test_knn_matches_brute_force_with_chunking():
    x_train = _benign(300, d=4)
    x_test = _benign(50, d=4, seed=5)
    det = MagicKNNDetector(k=7, chunk_elems=300 * 3).fit(x_train)
    ref = det.reference
    q = (x_test - det.mean) / det.std
    expected = torch.cdist(q, ref).topk(7, largest=False).values.mean(dim=1) / det.mean_distance
    assert torch.allclose(det.score(x_test), expected, rtol=1e-4, atol=1e-5)


def test_knn_constant_dimension_and_reference_cap():
    x_train = _benign(1000, d=4)
    x_train[:, 2] = 1.0
    det = MagicKNNDetector(k=3, max_reference_size=200).fit(x_train)
    assert det.reference.shape[0] == 200
    assert torch.isfinite(det.score(_benign(10, d=4, seed=9))).all()
