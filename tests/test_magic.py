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
    assert torch.allclose(det.mean, x_train.mean(dim=0))
    assert torch.allclose(det.std, x_train.std(dim=0, unbiased=False))
    # Same scores whatever the scale of the input space
    x_test = _benign(100, seed=3)
    det2 = MagicKNNDetector(k=5).fit(x_train * 7.0)
    assert torch.allclose(det.score(x_test), det2.score(x_test * 7.0), rtol=1e-3, atol=1e-4)


def _brute_force_scores(x_train, x_test, k):
    mean, std = x_train.mean(dim=0), x_train.std(dim=0, unbiased=False).clamp_min(1e-6)
    ref, q = (x_train - mean) / std, (x_test - mean) / std
    return torch.cdist(q, ref).topk(k, largest=False).values.mean(dim=1)


def test_knn_with_duplicate_embeddings_is_exact():
    # Many nodes share the same embedding: the distinct-embedding search must give the same
    # k-NN distances as a search over all the nodes
    base = _benign(40, d=4)
    reps = torch.randint(1, 15, (40,), generator=torch.Generator().manual_seed(1))
    x_train = base.repeat_interleave(reps, dim=0)
    x_test = torch.cat([_benign(30, d=4, seed=4), base[:5]])
    det = MagicKNNDetector(k=10, chunk_elems=7 * 40).fit(x_train)
    assert det.reference.shape[0] == 40 and det.num_reference_nodes == x_train.shape[0]
    expected = _brute_force_scores(x_train, x_test, 10)
    assert torch.allclose(det.score(x_test) * det.mean_distance, expected, rtol=1e-4, atol=1e-5)


def test_knn_matches_brute_force_with_chunking():
    x_train = _benign(300, d=4)
    x_test = _benign(50, d=4, seed=5)
    det = MagicKNNDetector(k=7, chunk_elems=300 * 3).fit(x_train)
    expected = _brute_force_scores(x_train, x_test, 7) / det.mean_distance
    assert torch.allclose(det.score(x_test), expected, rtol=1e-4, atol=1e-5)


def test_knn_constant_dimension_and_reference_cap():
    x_train = _benign(1000, d=4)
    x_train[:, 2] = 1.0
    det = MagicKNNDetector(k=3, max_reference_size=200).fit(x_train)
    assert det.num_reference_nodes == 200
    assert torch.isfinite(det.score(_benign(10, d=4, seed=9))).all()


# --- Masked graph autoencoder ---------------------------------------------------------------

import torch.nn as nn  # noqa: E402

from pidsmaker.encoders.magic_encoder import MagicGAT  # noqa: E402
from pidsmaker.objectives.predict_masked_struct import GMAEStructPrediction  # noqa: E402


def _graph(n=50, e=200, d=6, e_dim=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=g)
    edge_index = torch.randint(0, n, (2, e), generator=g)
    edge_feats = nn.functional.one_hot(torch.randint(0, e_dim, (e,), generator=g), e_dim).float()
    return x, edge_index, edge_feats


def _encoder(**kw):
    args = dict(in_dim=6, hid_dim=64, out_dim=64, n_layers=3, n_heads=4, edge_dim=5,
                activation=nn.PReLU())
    args.update(kw)
    return MagicGAT(**args)


def test_magic_encoder_splits_hidden_size_among_heads_and_concats_layers():
    enc = _encoder()
    # As in MAGIC: 4 heads of 16 units = 64 per layer, not 4 x 64
    assert [(c.heads, c.out_channels, c.concat) for c in enc.gats] == [(4, 16, True)] * 3
    x, ei, ef = _graph()
    h = enc(x, ei, edge_feats=ef)["h"]
    assert h.shape == (50, 3 * 64) == (50, enc.concat_dim)
    assert enc.detection_embedding(h).shape == (50, 64)
    assert torch.equal(enc.detection_embedding(h), h[:, -64:])


def test_magic_encoder_residual_activation_and_edge_attention():
    enc = _encoder()
    # A residual on every layer: projection on the first, identity when sizes match
    assert enc.gats[0].res.weight.shape == (64, 6)
    assert all(isinstance(c.res, nn.Identity) for c in enc.gats[1:])
    # One PReLU per layer, including the last encoder layer
    assert len({id(a) for a in enc.activations}) == 3
    assert all(isinstance(a, nn.PReLU) for a in enc.activations)
    # The edge type takes part in the attention
    assert all(c.lin_edge is not None for c in enc.gats)
    enc.eval()
    x, ei, ef = _graph()
    h1 = enc(x, ei, edge_feats=ef)["h"]
    h2 = enc(x, ei, edge_feats=ef.roll(1, dims=1))["h"]
    assert not torch.allclose(h1, h2)


def test_magic_decoder_projects_concat_and_has_no_output_activation():
    dec = MagicGAT(in_dim=192, hid_dim=64, out_dim=6, n_layers=1, n_heads=1, edge_dim=5,
                   activation=nn.PReLU(), is_decoder=True, in_proj_dim=64)
    assert dec.in_proj.in_features == 192 and dec.in_proj.out_features == 64
    assert dec.in_proj.bias is None
    assert isinstance(dec.activations[0], nn.Identity)
    assert dec.gats[0].res is None
    x, ei, ef = _graph()
    out = dec(torch.randn(50, 192), ei, edge_feats=ef)
    assert out.shape == (50, 6)
    assert (out < 0).any()  # a linear output, not squashed by an activation


def test_magic_struct_prediction_samples_like_magic():
    obj = GMAEStructPrediction(decoder=nn.Linear(8, 1), loss_fn=lambda p, y, **_: p.sum() * 0,
                               max_samples=10000)
    ei = torch.randint(0, 30, (2, 500))
    assert obj.sample_positive_edges(ei, num_nodes=30).shape == (2, 30)
    assert obj.sample_positive_edges(ei[:, :10], num_nodes=30).shape == (2, 10)
    obj.max_samples = 20
    assert obj.sample_positive_edges(ei, num_nodes=30).shape == (2, 20)
    obj.max_samples = 0
    assert obj.sample_positive_edges(ei, num_nodes=30).shape == (2, 500)


def test_magic_detection_embedding_last_or_concat():
    from types import SimpleNamespace

    from pidsmaker.detection.magic_knn import magic_detection_embedding

    model = SimpleNamespace(encoder=_encoder())
    h = torch.randn(7, 192)

    def cfg(mode):
        return SimpleNamespace(
            evaluation=SimpleNamespace(node_evaluation=SimpleNamespace(magic_embedding=mode))
        )

    assert torch.equal(magic_detection_embedding(model, h, cfg("last")), h[:, -64:])
    assert torch.equal(magic_detection_embedding(model, h, cfg("concat")), h)
