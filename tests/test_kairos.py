"""Tests of the parts of Kairos that follow the original implementation, without a database."""

import pytest
import torch
from yacs.config import CfgNode as CN

from pidsmaker.encoders.custom_mlp import build_mlp_from_string
from pidsmaker.encoders.graph_attention import GraphAttentionEmbedding
from pidsmaker.tgn import LastNeighborLoader
from pidsmaker.utils.data_utils import _msg_has_edge_type

pytestmark = pytest.mark.no_database


def decoder_cfg(objective, edge_type_in_msg=None):
    cfg = CN()
    cfg.training = CN()
    cfg.training.decoder = CN()
    cfg.training.decoder.used_methods = objective
    cfg.training.decoder.predict_edge_type = CN()
    cfg.training.decoder.predict_edge_type.edge_type_in_msg = edge_type_in_msg
    return cfg


def test_msg_has_edge_type():
    # Removed from the messages when predicting edge types, unless kept as in Kairos
    assert not _msg_has_edge_type(decoder_cfg("predict_edge_type"))
    assert not _msg_has_edge_type(decoder_cfg("predict_edge_type", False))
    assert _msg_has_edge_type(decoder_cfg("predict_edge_type", True))
    assert _msg_has_edge_type(decoder_cfg("predict_edge_contrastive"))


def test_mlp_dropout_rates():
    mlp = build_mlp_from_string(
        "linear(2) | dropout(0.5) | tanh | linear(0.25) | dropout | tanh", 400, 7, dropout=0.1
    )
    dropouts = [m.p for m in mlp.mlp if isinstance(m, torch.nn.Dropout)]
    assert dropouts == [0.5, 0.1]
    linears = [(m.in_features, m.out_features) for m in mlp.mlp if isinstance(m, torch.nn.Linear)]
    assert linears == [(400, 800), (800, 200), (200, 7)]


@pytest.mark.parametrize("final_activation", [False, True])
def test_graph_attention_final_activation(final_activation):
    torch.manual_seed(0)
    gnn = GraphAttentionEmbedding(
        in_dim=8, hid_dim=8, out_dim=8, edge_dim=4, dropout=0.0, activation=torch.nn.ReLU(),
        num_heads=8, concat=True, num_layers=2, final_activation=final_activation,
    )
    # Kairos: TransformerConv(100, 100, heads=8) then TransformerConv(800, 100, heads=1)
    assert gnn.convs[0].heads == 8 and gnn.convs[1].in_channels == 64
    x = torch.randn(5, 8)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    h = gnn(x, edge_index, torch.randn(4, 4))["h"]
    assert bool((h >= 0).all()) == final_activation


def test_neighbor_loader_clear_keeps_edge_ids():
    loader = LastNeighborLoader(10, size=3)
    loader.insert(torch.tensor([0, 1]), torch.tensor([1, 2]))
    loader.clear()
    n_id, edge_index, e_id = loader(torch.tensor([0, 1, 2]))
    assert e_id.numel() == 0 and edge_index.shape[1] == 0
    # The IDs of the next edges still index the concatenation of all the edges
    loader.insert(torch.tensor([3]), torch.tensor([4]))
    _, _, e_id = loader(torch.tensor([3]))
    assert e_id.tolist() == [2]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_packed_batch_to_device_matches_to():
    from pidsmaker.utils.data_utils import CollatableTemporalData, batch_to_device

    g = torch.Generator().manual_seed(0)
    data = CollatableTemporalData(
        src=torch.randint(0, 100, (37,), generator=g),
        dst=torch.randint(0, 100, (37,), generator=g),
        t=torch.randint(0, 10**18, (37,), generator=g),
        msg=torch.randn(37, 5, generator=g),
    )
    data.y = torch.zeros(37, dtype=torch.bool)
    data.edge_index_tgn = torch.randint(0, 100, (2, 11), generator=g)[:, ::2]  # non-contiguous
    data.empty = torch.zeros(0, 3, dtype=torch.float16)
    data.scalar = torch.tensor(3.5, dtype=torch.float64)
    data.tgn_mode = True
    for _ in range(3):  # both pinned buffers, then the first one again
        out = batch_to_device(data, "cuda")
        ref = data.clone().to("cuda")
        assert out.tgn_mode is True and data.src.device.type == "cpu"
        for key, value in ref.items():
            if isinstance(value, torch.Tensor):
                assert out[key].dtype == value.dtype and out[key].shape == value.shape, key
                assert torch.equal(out[key], value), key



@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_batch_to_device_keeps_tensors_already_on_the_gpu():
    from pidsmaker.utils.data_utils import CollatableTemporalData, batch_to_device

    data = CollatableTemporalData(src=torch.arange(4), x=torch.ones(4, 2, device="cuda"))
    for device in ("cuda", "cuda:0"):
        out = batch_to_device(data, device)
        assert out.x is data.x and out.src.is_cuda and torch.equal(out.src.cpu(), data.src)

def encoder_cfg(encoders, objectives, use_memory=True, use_node_feats_in_gnn=False):
    cfg = decoder_cfg(objectives)
    cfg.training.decoder.use_few_shot = False
    cfg.training.encoder = CN()
    cfg.training.encoder.used_methods = encoders
    cfg.training.encoder.tgn = CN()
    cfg.training.encoder.tgn.use_memory = use_memory
    cfg.training.encoder.tgn.use_node_feats_in_gnn = use_node_feats_in_gnn
    return cfg


def test_per_batch_node_feats_used():
    from pidsmaker.utils.data_utils import _per_batch_node_feats_used

    # Kairos: the TGN embeds its memory, the objective only reads the embeddings of the edges
    assert _per_batch_node_feats_used(encoder_cfg("graph_attention,tgn", "predict_edge_type")) == (
        False,
        False,
    )
    assert _per_batch_node_feats_used(
        encoder_cfg("tgn,graph_attention", "predict_edge_type", use_memory=False)
    ) == (True, False)
    assert _per_batch_node_feats_used(
        encoder_cfg("tgn,graph_attention", "predict_edge_type", use_node_feats_in_gnn=True)
    ) == (True, False)
    assert _per_batch_node_feats_used(
        encoder_cfg("tgn,graph_attention", "reconstruct_node_features")
    ) == (False, True)
    assert _per_batch_node_feats_used(encoder_cfg("graph_attention", "predict_edge_type")) == (
        True,
        True,
    )
