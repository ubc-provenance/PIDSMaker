"""Tests of the ThreaTrace reimplementation against the original code
(https://github.com/threaTrace-detector/threaTrace, scripts/train_darpatc.py and test_darpatc.py).
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

from pidsmaker.config import get_runtime_required_args, get_yml_cfg
from pidsmaker.detection.evaluation_methods.evaluation_utils import get_threshold
from pidsmaker.detection.threatrace_detection import threatrace_anomaly_score
from pidsmaker.detection.training_methods.threatrace_training import (
    ThreaTraceEnsemble,
    train_submodels,
)
from pidsmaker.objectives.predict_node_type import NodeTypePrediction
from pidsmaker.utils.data_utils import get_selected_node_features

pytestmark = pytest.mark.no_database


def original_is_anomalous(out, y, thre):
    """The detection rule of test_darpatc.py: `pred` is set to 100 when the top-1 / top-2
    probability ratio is below `thre`, and the node stays in the alarms when `pred != y`."""
    pred = out.max(1)[1]
    pro = F.softmax(out, dim=1)
    pro1 = pro.max(1)
    for i in range(len(y)):
        pro[i][pro1[1][i]] = -1
    pro2 = pro.max(1)
    for i in range(len(y)):
        if pro1[0][i] / pro2[0][i] < thre:
            pred[i] = 100
    return pred != y


@pytest.mark.parametrize("thre", [1.0, 1.5, 3.0])
def test_score_matches_original_rule(thre):
    torch.manual_seed(0)
    out = F.log_softmax(torch.randn(2000, 4) * 2, dim=1)
    y = torch.randint(0, 4, (2000,))
    score = threatrace_anomaly_score([out], y, thre)
    assert torch.equal(score > 0, original_is_anomalous(out.clone(), y, thre))
    # The threshold applied by the evaluation
    assert get_threshold("unused", "threatrace") == 0.0


def test_score_is_anomalous_only_if_no_submodel_explains_the_node():
    torch.manual_seed(0)
    outs = [F.log_softmax(torch.randn(1000, 3) * 2, dim=1) for _ in range(3)]
    y = torch.randint(0, 3, (1000,))
    score = threatrace_anomaly_score(outs, y, 1.5)
    # test_darpatc.py removes the nodes explained by each submodel in turn: the alarms are
    # the nodes that every submodel flags
    expected = torch.stack([original_is_anomalous(o.clone(), y, 1.5) for o in outs]).all(0)
    assert torch.equal(score > 0, expected)
    assert expected.any() and not expected.all()


def _cfg(method, node_features):
    return SimpleNamespace(
        featurization=SimpleNamespace(used_method=method),
        batching=SimpleNamespace(node_features=node_features),
    )


def test_selected_node_features():
    # ThreaTrace's features: the in/out edge type counts, without the node type (the label)
    assert get_selected_node_features(_cfg("only_type", "edges_distribution")) == ["edges_distribution"]
    # Featurizations without embeddings drop `node_emb`, the node type is the fallback
    assert get_selected_node_features(_cfg("only_type", "node_emb,node_type")) == ["node_type"]
    assert get_selected_node_features(_cfg("only_type", "node_emb")) == ["node_type"]
    assert get_selected_node_features(_cfg("only_ones", "node_type")) == ["only_ones"]
    assert get_selected_node_features(_cfg("word2vec", "node_emb,node_type")) == ["node_emb", "node_type"]


def _yml_cfg(model, dataset, artifact_dir, custom_args=()):
    args, _ = get_runtime_required_args(return_unknown_args=True, args=[model, dataset])
    args.__dict__["artifact_dir"] = str(artifact_dir) + "/"  # not the default /home/artifacts
    for k, v in custom_args:
        args.__dict__[k] = v
    return get_yml_cfg(args)


def test_threatrace_config(tmp_path):
    cfg = _yml_cfg("threatrace", "CADETS_E3", tmp_path)
    assert get_selected_node_features(cfg) == ["edges_distribution"]
    assert cfg.training.used_method == "threatrace"
    assert cfg.batching.fix_buggy_graph_reindexer
    assert cfg.batching.edges_distribution_scaling == "none"  # raw counts
    # Original GraphSAGE: 2 layers, 32 hidden neurons, one output per node type
    assert cfg.training.encoder.sage.num_layers == 2
    assert cfg.training.node_hid_dim == 32
    assert cfg.training.node_out_dim == cfg.dataset.num_node_types
    assert cfg.training.threatrace.ratio_threshold == 1.5


def test_node_type_cannot_be_a_feature_of_node_type_prediction(tmp_path):
    with pytest.raises(ValueError, match="cannot be a node feature"):
        _yml_cfg(
            "threatrace", "CADETS_E3", tmp_path,
            [("batching.node_features", "node_type,edges_distribution")],
        )


def test_node_type_prediction_loss_mask():
    torch.manual_seed(0)
    objective = NodeTypePrediction(nn.Identity(), _ce, balanced_loss=False, node_type_dim=3)
    h = torch.randn(6, 3)
    node_type = F.one_hot(torch.tensor([0, 1, 2, 0, 1, 2]), 3).float()
    mask = torch.tensor([True, False, True, False, False, True])
    batch = SimpleNamespace(node_loss_mask=mask)
    loss = objective(h, node_type, inference=False, batch=batch)["loss"]
    assert torch.isclose(loss, F.cross_entropy(h[mask], node_type[mask].argmax(1)))
    # The mask only applies to the training
    scores = objective(h, node_type, inference=True, batch=batch)["loss"]
    assert scores.shape == (6,)


def _ce(x, y, inference=False, weight=None):
    return F.cross_entropy(x, y, reduction="none" if inference else "mean", weight=weight)


class ToyModel(nn.Module):
    """Node-type classifier on the node features only, with the interface of `Model`."""

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(2, 2)

    def forward(self, g, inference=False, validation=False):
        out = F.log_softmax(self.lin(g.x), dim=1)
        y = g.node_type.argmax(1)
        mask = getattr(g, "node_loss_mask", None)
        if inference:
            return {"loss": F.nll_loss(out, y, reduction="none"), "out": out}
        return {"loss": F.nll_loss(out[mask], y[mask]), "out": out}


def _toy_cfg(**threatrace):
    tt = dict(
        ratio_threshold=1.5, max_submodels=100, max_retrain_epochs=20,
        retrain_loss_threshold=1.0, max_rounds_without_progress=3,
    )
    tt.update(threatrace)
    return SimpleNamespace(
        training=SimpleNamespace(
            num_epochs=50,
            threatrace=SimpleNamespace(**tt),
            decoder=SimpleNamespace(few_shot=SimpleNamespace(include_attacks_in_ssl_training=False)),
        )
    )


def _toy_graph(x, y):
    return Data(
        x=x, node_type=F.one_hot(y, 2).float(), y=torch.zeros(1), edge_index=torch.zeros(2, 1).long()
    )


def test_train_submodels_until_all_nodes_are_explained():
    torch.manual_seed(0)
    x = torch.cat([torch.randn(50, 2) + 3, torch.randn(50, 2) - 3])
    y = torch.cat([torch.zeros(50), torch.ones(50)]).long()
    graphs = [_toy_graph(x, y)]
    model = ToyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    submodels = train_submodels(model, optimizer, graphs, _toy_cfg(), "cpu")
    ensemble = ThreaTraceEnsemble(model, submodels)
    ensemble.eval()
    outs = ensemble(graphs[0], inference=True)["outs"]
    assert len(outs) == len(submodels) >= 1
    assert not (threatrace_anomaly_score(outs, y, 1.5) > 0).any()


def test_train_submodels_stops_without_progress():
    torch.manual_seed(0)
    # Nodes with the same features and different types can never be explained
    x = torch.cat([torch.randn(50, 2) + 3, torch.zeros(20, 2)])
    y = torch.cat([torch.zeros(50), torch.zeros(10), torch.ones(10)]).long()
    graphs = [_toy_graph(x, y)]
    model = ToyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    cfg = _toy_cfg(max_retrain_epochs=5)
    submodels = train_submodels(model, optimizer, graphs, cfg, "cpu")
    assert 1 <= len(submodels) < cfg.training.threatrace.max_submodels

    model = ToyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    assert len(train_submodels(model, optimizer, graphs, _toy_cfg(max_submodels=1), "cpu")) == 1
