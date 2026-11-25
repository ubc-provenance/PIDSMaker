"""Tests of network datasets (e.g. LANL) and of the dataset schemas.

The LANL tests run on a small synthetic dataset in the LANL format (see `synthetic_lanl.py`),
loaded into a temporary database by the LANL preprocessing script, so that they don't
require the real dataset.
"""

import os
import shutil
import subprocess
import sys

import pytest
import torch
import wandb

from pidsmaker import main
from pidsmaker.config import (
    DATASET_DEFAULT_CONFIG,
    get_runtime_required_args,
    get_yml_cfg,
)
from pidsmaker.config.schemas import SCHEMAS, get_schema
from pidsmaker.utils.dataset_utils import encode_edge_attributes, get_rel2id

sys.path.insert(0, os.path.dirname(__file__))
import synthetic_lanl  # noqa: E402
from conftest import admin_execute, database_params  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Specific to the process: pytest-xdist workers and concurrent test sessions each have their own
TEST_DATABASE = f"pidsmaker_tests_lanl_{os.getpid()}"
# Set by `lanl_database`
TESTS_ARTIFACT_DIR = None
USE_CPU = False


def prepare_cfg(model, dataset="LANL", custom_args=(), test_mode=True):
    args, _ = get_runtime_required_args(return_unknown_args=True, args=[model, dataset])
    args.__dict__["artifact_dir"] = TESTS_ARTIFACT_DIR
    # Plotting dominates the evaluation time (figures are tested in test_framework.py)
    args.__dict__["evaluation.save_figures"] = False
    for k, v in custom_args:
        args.__dict__[k] = v
    cfg = get_yml_cfg(args)
    cfg._test_mode = test_mode
    if USE_CPU:
        cfg._use_cpu = True
    return cfg


@pytest.fixture(scope="module", autouse=True)
def lanl_database(tmp_path_factory, artifact_dir, device):
    """Loads the synthetic LANL dataset into the test database, used by the LANL dataset."""
    global TESTS_ARTIFACT_DIR, USE_CPU
    TESTS_ARTIFACT_DIR = os.path.join(artifact_dir, "network") + "/"
    USE_CPU = device == "cpu"
    data_dir = tmp_path_factory.mktemp("lanl")
    auth, redteam = synthetic_lanl.generate(str(data_dir))
    subprocess.run(
        [
            sys.executable,
            os.path.join(ROOT, "dataset_preprocessing/lanl/create_database_lanl.py"),
            "LANL",
            "--auth_file",
            auth,
            "--redteam_file",
            redteam,
            "--database",
            TEST_DATABASE,
            "--overwrite",
            # The default artifact directory (/home/artifacts/) may not be writable, e.g. in CI
            "--artifact_dir",
            TESTS_ARTIFACT_DIR,
        ],
        check=True,
        cwd=ROOT,
    )
    original = dict(DATASET_DEFAULT_CONFIG["LANL"])
    DATASET_DEFAULT_CONFIG["LANL"]["database"] = TEST_DATABASE
    DATASET_DEFAULT_CONFIG["LANL"]["database_all_file"] = TEST_DATABASE
    shutil.rmtree(TESTS_ARTIFACT_DIR, ignore_errors=True)
    wandb.init(mode="disabled")
    yield
    DATASET_DEFAULT_CONFIG["LANL"].update(original)
    shutil.rmtree(TESTS_ARTIFACT_DIR, ignore_errors=True)
    admin_execute(database_params(), f'DROP DATABASE IF EXISTS "{TEST_DATABASE}";')


class TestSchemas:
    def test_provenance_vocabularies_unchanged(self):
        # Provenance datasets keep the ids of their original vocabularies
        tc = SCHEMAS["darpa_tc"]
        assert tc.ntype2id == {1: "subject", "subject": 1, 2: "file", "file": 2, 3: "netflow", "netflow": 3}
        assert tc.num_edge_types == 10 and SCHEMAS["optc"].num_edge_types == 10
        assert SCHEMAS["carbon_black_edr"].num_edge_types == 33
        assert SCHEMAS["optc_tc_edge_types"].num_edge_types == 14

    def test_every_dataset_has_a_schema(self):
        for name, dataset in DATASET_DEFAULT_CONFIG.items():
            assert dataset["schema"] in SCHEMAS, name

    def test_edge_type_from_attribute(self):
        cfg = prepare_cfg("velox", custom_args=[("construction.edge_type", "logon_type")])
        schema = get_schema(cfg)
        assert schema.edge_type_attribute == "logon_type"
        assert get_rel2id(cfg)["Network"] == 1
        assert cfg.dataset.num_node_types == 1

    def test_invalid_edge_type(self):
        with pytest.raises(ValueError):
            prepare_cfg("velox", dataset="CLEARSCOPE_E3", custom_args=[("construction.edge_type", "logon_type")])

    def test_node_type_objective_needs_several_node_types(self):
        with pytest.raises(ValueError, match="at least two node types"):
            prepare_cfg("threatrace")

    def test_noise_events_filter(self):
        from types import SimpleNamespace

        from pidsmaker.config.schemas import drops_noise_events

        # OpTC noise events are dropped by default (also when the option is unset)
        optc = SCHEMAS["optc_flows"]
        assert drops_noise_events(SimpleNamespace()) and drops_noise_events(None)
        assert not drops_noise_events(SimpleNamespace(drop_noise_events=False))
        assert "NOT (" in optc.with_noise_dropped(True).operations_sql()
        assert optc.with_noise_dropped(False).operations_sql() == optc.operations_sql()
        # No effect on schemas without noise events
        for name, schema in SCHEMAS.items():
            if not schema.noise_events_sql:
                assert schema.with_noise_dropped(True) is schema, name

    def test_operations_filter(self):
        # OpTC flows keep FLOW_START events by default (as LARES and Argus)
        optc = SCHEMAS["optc_flows"]
        assert optc.operations_sql() == " AND operation IN ('FLOW_START')"
        assert "FLOW_MESSAGE" not in optc.event_filter_rel2id
        assert optc.with_operations("all").operations_sql() == ""
        both = optc.with_operations("FLOW_START,FLOW_MESSAGE")
        assert set(both.operations) == {"FLOW_START", "FLOW_MESSAGE"}
        # The filter is kept when another attribute is the edge type
        assert optc.with_edge_type("dst_port").operations_sql() == optc.operations_sql()
        # Edge types are not changed by the filter
        assert optc.with_operations("all").num_edge_types == optc.num_edge_types
        # Other schemas keep all events by default
        assert SCHEMAS["lanl_auth"].operations_sql() == ""
        with pytest.raises(ValueError):
            optc.with_operations("LogOn")

    def test_edge_attribute_encoding(self):
        # The attribute used as edge type is not an input feature
        cfg = prepare_cfg("velox", custom_args=[("construction.edge_type", "logon_type")])
        columns = {
            # An authentication (no flow attributes) and a flow (no authentication attributes)
            "auth_type": ["NTLM", None],
            "success": ["Success", None],
            "src_user_kind": ["user", None],
            "dst_user_kind": ["computer", None],
            "src_port": [None, "N"],
            "dst_port": [None, "445"],
            "protocol": [None, "6"],
            "duration": [None, 3.0],
            "packets": [None, 10.0],
            "bytes": [None, float("nan")],
        }
        feats = encode_edge_attributes(cfg, columns)
        categorical = [a for a in get_schema(cfg).edge_attributes if a.kind == "categorical" and a.name != "logon_type"]
        numeric = [a for a in get_schema(cfg).edge_attributes if a.kind == "numeric"]
        assert feats.shape == (2, sum(len(a.values) for a in categorical) + len(numeric))
        # One value per categorical attribute (null values are OTHER), numeric values are log1p
        assert torch.all(feats[:, : feats.shape[1] - len(numeric)].sum(dim=1) == len(categorical))
        assert torch.allclose(feats[1, -3:], torch.log1p(torch.tensor([3.0, 10.0, 0.0])))
        assert torch.all(feats[0, -3:] == 0)


NIDS_SYSTEMS = ["euler", "argus", "jbeil", "lares", "egraphsage", "anomal_e"]


class TestLANL:
    systems = ["velox", "orthrus", "kairos", "magic", "ocrapt", "shadewatcher"] + NIDS_SYSTEMS

    @pytest.mark.parametrize("system", systems)
    def test_systems(self, system):
        main.main(prepare_cfg(system))

    def test_edge_type_attribute(self):
        main.main(prepare_cfg("velox", custom_args=[("construction.edge_type", "logon_type")]))

    @pytest.mark.parametrize(
        "system,edge_features",
        [("orthrus", "edge_type,edge_attrs"), ("kairos", "msg,time_encoding,edge_attrs")],
    )
    def test_edge_attribute_features(self, system, edge_features):
        main.main(prepare_cfg(system, custom_args=[("batching.edge_features", edge_features)]))

    def test_edge_attributes_require_attributes(self):
        with pytest.raises(ValueError, match="edge attributes"):
            cfg = prepare_cfg(
                "orthrus", dataset="CLEARSCOPE_E3", custom_args=[("batching.edge_features", "edge_attrs")]
            )
            main.main(cfg)

    def test_labels_from_events(self):
        cfg = prepare_cfg("velox")
        main.main(cfg)
        from pidsmaker.utils.labelling import get_ground_truth, get_malicious_events

        events = get_malicious_events(cfg)
        assert len(events) == 25  # red team events of the synthetic dataset
        nids, _, _ = get_ground_truth(cfg)
        assert len(nids) == 26  # the attacker's computer and its 25 targets


    @pytest.mark.parametrize("benign", [True, False])
    def test_benign_training(self, tmp_path, benign):
        """With `benign_training`, the graphs of train/val dates have no malicious edge."""
        import glob

        from pidsmaker.preprocessing.build_graph_methods import build_default_graphs
        from pidsmaker.preprocessing.cleaning import Cleaner
        from pidsmaker.utils.graph_store import load_graph

        cfg = prepare_cfg("velox", custom_args=[("construction.benign_training", benign)], test_mode=False)
        # The first red team events of the synthetic dataset happen on 2015-01-03
        cfg.dataset.train_dates, cfg.dataset.val_dates = ["2015-01-03"], []
        cfg.dataset.test_dates = ["2015-01-04"]
        cfg.construction._graphs_dir = str(tmp_path)
        cleaner = Cleaner(cfg)
        indexid2msg = build_default_graphs.compute_indexid2msg(cfg=cfg, cleaner=cleaner)
        build_default_graphs.gen_edge_fused_tw(indexid2msg=indexid2msg, cfg=cfg, cleaner=cleaner, file2nodes={})

        def num_malicious(date):
            paths = glob.glob(os.path.join(str(tmp_path), f"graph_{date}", "*"))
            return sum(d["y"] for p in paths for *_, d in load_graph(p).edges(data=True))

        assert (num_malicious("2015-01-03") == 0) == benign
        assert num_malicious("2015-01-04") > 0  # test dates keep their malicious events


class TestNIDSOnProvenanceDatasets:
    """Network IDSs run on provenance datasets, which have no edge attributes: edge features
    that rely on them are replaced by the edge type."""

    @pytest.mark.parametrize("system", NIDS_SYSTEMS)
    def test_systems(self, system):
        custom_args = []
        if system in ["argus", "lares", "egraphsage", "anomal_e"]:
            custom_args.append(("batching.edge_features", "edge_type"))
        main.main(prepare_cfg(system, dataset="CLEARSCOPE_E3", custom_args=custom_args))


class TestMergeEdges:
    def test_merge_edges(self):
        import networkx as nx

        from pidsmaker.preprocessing.transformation_methods import transformation_merge_edges

        cfg = prepare_cfg("euler")
        G = nx.MultiDiGraph()
        G.add_nodes_from([(1, {"node_type": "host"}), (2, {"node_type": "host"}), (3, {"node_type": "host"})])
        G.add_edge(1, 2, time=3, label="LogOn", y=0)
        G.add_edge(1, 2, time=1, label="LogOff", y=1)
        G.add_edge(2, 3, time=2, label="LogOn", y=0)
        H = transformation_merge_edges.main(G, cfg)
        edges = list(H.edges(data=True))
        assert len(edges) == 2
        (u, v, first), (_, _, second) = edges
        assert (u, v) == (1, 2) and first["time"] == 1 and first["count"] == 2 and first["y"] == 1
        assert first["weight"] > 0.5 > second["weight"]

    def test_aggregated_attributes(self):
        import networkx as nx

        from pidsmaker.preprocessing.transformation_methods import transformation_merge_edges
        from pidsmaker.utils.dataset_utils import (
            aggregated_edge_attribute_columns,
            encode_aggregated_edge_attributes,
            get_edge_attrs_dim,
        )

        cfg = prepare_cfg("lares")
        G = nx.MultiDiGraph()
        G.add_nodes_from([(n, {"node_type": "host"}) for n in (1, 2, 3)])
        G.add_edge(1, 2, time=1, label="LogOn", y=0, success="Success", bytes=float("nan"))
        G.add_edge(1, 2, time=2, label="LogOn", y=1, success="Fail", bytes=10.0)
        G.add_edge(1, 2, time=3, label="LogOn", y=0, success="Success", bytes=30.0)
        G.add_edge(2, 3, time=4, label="LogOn", y=0, success="Success", bytes=None)
        H = transformation_merge_edges.main(G, cfg)
        (_, _, a), (_, _, b) = H.edges(data=True)
        # Counts of events per value of categorical attributes, stats of numeric attributes
        assert a["count"] == 3 and a["success=Success"] == 2 and a["success=Fail"] == 1
        assert a["bytes:n"] == 2 and a["bytes:mean"] == 20.0 and a["bytes:std"] == 10.0
        assert b["count"] == 1 and b["success=Success"] == 1 and b["bytes:n"] == 0

        columns = [c for cols in aggregated_edge_attribute_columns(cfg).values() for c, _ in cols]
        feats = encode_aggregated_edge_attributes(
            cfg, {c: [e[2].get(c) for e in H.edges(data=True)] for c in columns}
        )
        assert feats.shape == (2, get_edge_attrs_dim(cfg))
        assert ((feats >= 0) & (feats <= 1)).all()


    @pytest.mark.parametrize("aggregate,per_edge_type", [(True, False), (False, False), (True, True)])
    def test_merge_edges_from_columns(self, tmp_path, aggregate, per_edge_type):
        """Edges are merged the same from a graph and from its stored columns."""
        import math

        import networkx as nx
        import numpy as np

        from pidsmaker.preprocessing.transformation_methods import transformation_merge_edges
        from pidsmaker.utils.dataset_utils import get_input_edge_attributes
        from pidsmaker.utils.graph_store import load_graph, load_graph_columns, save_graph

        cfg = prepare_cfg("lares")
        cfg.transformation.merge_edges.aggregate_attributes = aggregate
        cfg.transformation.merge_edges.per_edge_type = per_edge_type
        rng = np.random.default_rng(0)
        attributes = get_input_edge_attributes(cfg, selected=False)
        G = nx.MultiDiGraph()
        G.add_nodes_from((str(n), {"node_type": "host", "label": f"C{n}"}) for n in range(12))
        for i in range(400):
            u, v = (str(x) for x in rng.integers(0, 6, 2))
            attrs = {}
            for attr in attributes:
                if attr.kind == "categorical":
                    attrs[attr.name] = [*attr.vocab, "unknown", None][rng.integers(len(attr.vocab) + 2)]
                else:
                    attrs[attr.name] = [None, float("nan"), float(rng.integers(100))][rng.integers(3)]
            G.add_edge(
                u, v, event_uuid=str(i), time=int(rng.integers(20)), label=["LogOn", "LogOff"][i % 2],
                y=int(rng.random() < 0.1), **attrs,
            )
        path = str(tmp_path / "g")
        save_graph(G, path)
        expected = transformation_merge_edges.main(load_graph(path), cfg)
        merged = transformation_merge_edges.merge_columns(load_graph_columns(path), cfg)
        # Merged once per (src, dst) (and edge type), ordered by first event, labels are maxima
        groups = {}
        for u, v, d in G.edges(data=True):
            key = (u, v, d["label"]) if per_edge_type else (u, v)
            groups.setdefault(key, []).append(d)
        assert merged.number_of_edges() == len(groups)
        for u, v, d in merged.edges(data=True):
            events = groups[(u, v, d["label"]) if per_edge_type else (u, v)]
            assert d["count"] == len(events) and d["y"] == max(e["y"] for e in events)
            assert d["time"] == min(e["time"] for e in events)

        assert list(merged.nodes(data=True)) == list(expected.nodes(data=True))
        a, b = list(merged.edges(keys=True, data=True)), list(expected.edges(keys=True, data=True))
        assert [e[:3] for e in a] == [e[:3] for e in b]
        for (*_, x), (*_, y) in zip(a, b):
            assert list(x) == list(y)
            for k in x:
                if isinstance(x[k], float) and isinstance(y[k], float):
                    assert (math.isnan(x[k]) and math.isnan(y[k])) or math.isclose(x[k], y[k], rel_tol=1e-9, abs_tol=1e-12), k
                else:
                    assert x[k] == y[k] and type(x[k]) is type(y[k]), k


class TestNIDSComponents:
    def test_uniform_negative_sampling(self):
        import torch

        from pidsmaker.objectives.predict_edge_contrastive import EdgeContrastivePrediction

        obj = EdgeContrastivePrediction(
            decoder=None, loss_fn=None, graph_reindexer=None, neg_sampling="uniform", neg_ratio=10
        )
        h = torch.randn(50, 4)
        edge_index = torch.tensor([[0, 1, 2], [3, 3, 3]])
        neg_src, neg_dst = obj._uniform_negative_sampling(edge_index, h, h[edge_index[0]], h[edge_index[1]], 10)
        # Negatives reach any node of h, not only the destinations of the graph
        assert 20 <= neg_src.shape[0] <= 30
        assert not torch.equal(neg_dst, h[3].expand_as(neg_dst))

    @pytest.mark.parametrize("num_pos,num_neg,margin", [(1, 1, 0.8), (40, 7, 0.8), (300, 900, 0.5), (500, 200, 1.0)])
    def test_ap_contrastive_matches_dense(self, num_pos, num_neg, margin):
        """The AP loss of Argus, computed with sorted suffix sums, equals the dense
        (positives x all edges) version, in value and gradients."""
        import torch

        from pidsmaker.losses import ap_contrastive

        def dense(positive, negative):
            pos, neg = torch.sigmoid(positive.flatten()), torch.sigmoid(negative.flatten())
            all_scores = torch.cat([pos, neg])
            hinge = torch.clamp(margin - (pos.unsqueeze(1) - all_scores.unsqueeze(0)), min=0) ** 2
            precision = hinge[:, : pos.shape[0]].sum(dim=1) / (hinge.sum(dim=1) + 1e-12)
            return -precision.mean()

        torch.manual_seed(0)
        pos = (torch.randn(num_pos, 1) * 3).requires_grad_()
        neg = (torch.randn(num_neg, 1) * 3 - 1).requires_grad_()
        expected, actual = dense(pos, neg), ap_contrastive(pos, neg, margin=margin)
        assert torch.allclose(actual, expected, atol=1e-6)
        for g_actual, g_expected in zip(torch.autograd.grad(actual, [pos, neg]), torch.autograd.grad(expected, [pos, neg])):
            assert torch.allclose(g_actual, g_expected, atol=1e-6)

    def test_contrastive_inference_single_edge(self):
        """At inference, the objective returns one loss per edge, also for a single edge."""
        import torch

        from pidsmaker.objectives.predict_edge_contrastive import EdgeContrastivePrediction

        obj = EdgeContrastivePrediction(
            decoder=lambda h_src, h_dst: (h_src * h_dst).sum(-1, keepdim=True),
            loss_fn=lambda pos, neg, inference: -pos,
            graph_reindexer=None,
            neg_sampling="uniform",
            neg_ratio=1,
        )
        for num_edges in (1, 3):
            h = torch.randn(num_edges, 4)
            edge_index = torch.zeros(2, num_edges, dtype=torch.long)
            loss = obj(h, h, edge_index, inference=True, h=h)["loss"]
            assert loss.shape == (num_edges,)

    def test_edge_autoencoder(self):
        import torch
        from types import SimpleNamespace

        from pidsmaker.objectives.autoencode_edge_emb import EdgeEmbAutoencoder

        obj = EdgeEmbAutoencoder(node_dim=8, hid_dim=16, edge_dim=3)
        h_src, h_dst = torch.randn(5, 8), torch.randn(5, 8)
        batch = SimpleNamespace(edge_feats=torch.rand(5, 3))
        assert obj(h_src, h_dst, inference=False, batch=batch)["loss"].dim() == 0
        assert obj(h_src, h_dst, inference=True, batch=batch)["loss"].shape == (5,)

    def test_argus_edge_conditioned_conv(self):
        """The ARGUS convolution equals NNConv without building per-edge weight matrices."""
        import torch
        import torch.nn as nn
        from torch_geometric.nn import NNConv

        from pidsmaker.encoders.argus_encoder import _edge_conditioned_conv

        torch.manual_seed(0)
        conv = NNConv(16, 12, nn.Sequential(nn.Linear(5, 8), nn.ReLU(), nn.Linear(8, 16 * 12)), aggr="mean")
        x = torch.randn(40, 16, requires_grad=True)
        edge_index, edge_attr = torch.randint(0, 40, (2, 300)), torch.randn(300, 5)
        expected, out = conv(x, edge_index, edge_attr), _edge_conditioned_conv(conv, x, edge_index, edge_attr)
        assert torch.allclose(out, expected, atol=1e-5)
        params = [x, *conv.parameters()]
        for a, b in zip(torch.autograd.grad(out.sum(), params), torch.autograd.grad(expected.sum(), params)):
            assert torch.allclose(a, b, atol=1e-4)

    def test_two_stage_host_selection(self):
        from pidsmaker.detection.evaluation_methods.host_two_stage_evaluation import select_hosts

        scores = {1: 0.1, 2: 0.2, 3: 5.0, 4: 5.2, 5: 0.3}
        # Top cluster of the 3 highest scores (3, 4) and the host ranked just below them (5)
        assert sorted(select_hosts(scores, alpha=3)) == [3, 4, 5]

    def test_label_free_epoch_selection(self, tmp_path):
        import json
        from types import SimpleNamespace

        from pidsmaker.tasks.evaluation import _select_epoch_without_labels

        epochs = ["model_epoch_1", "model_epoch_11", "model_epoch_3"]
        (tmp_path / "val_scores").mkdir()
        for d, score in [("model_epoch_1", 1.2), ("model_epoch_11", 1.5), ("model_epoch_3", 1.9)]:
            (tmp_path / "val_scores" / f"{d}.json").write_text(json.dumps({"val_score": score}))
        cfg = SimpleNamespace(
            evaluation=SimpleNamespace(best_model_selection="last_epoch"),
            training=SimpleNamespace(_edge_losses_dir=str(tmp_path)),
        )
        # Epochs are compared numerically, not as strings
        assert _select_epoch_without_labels(cfg, epochs) == "model_epoch_11"
        cfg.evaluation.best_model_selection = "best_val_score"
        assert _select_epoch_without_labels(cfg, epochs) == "model_epoch_3"


def test_ranking_metrics():
    """Hand-computed values of the ranking metrics, with ties."""
    import math

    from pidsmaker.detection.evaluation_methods import evaluation_utils as eu

    # Ranking a(TP, attack 0), b(FP), c(TP, attack 1), d(FP): points (precision, % detected)
    # (0, 0), (1, 50), (.5, 50), (2/3, 100), (.5, 100) -> area 25 + 100/6 + 25
    adp = eu.plot_detected_attacks_vs_precision(
        [0.9, 0.8, 0.7, 0.6], ["a", "b", "c", "d"], {"a": {0}, "c": {1}}, [1, 0, 1, 0], None
    )
    assert math.isclose(adp, (25 + 100 / 6 + 25) / 100)

    # The attacks are all detected at c, tied with b: ties are ranked by descending item, so
    # c comes first and b is not counted
    fps, tps, precision, recall = eu.get_metrics_if_all_attacks_detected(
        [0.9, 0.5, 0.5, 0.1], ["a", "b", "c", "d"], {0: {"c"}, 1: {"a"}}
    )
    assert (fps, tps) == (0, 1) and math.isclose(precision, 1) and math.isclose(recall, 0.5)
    # Nodes of an attack absent from the ranking: all items are counted
    assert eu.get_metrics_if_all_attacks_detected([0.9, 0.1], ["a", "b"], {0: {"a"}, 1: {"z"}})[:2] == (1, 1)

    # Discrimination: max normalized score of each attack minus the mean of the top-k benign ones
    scores, nodes, y = [1.0, 0.5, 0.25, 0.75], ["a", "b", "c", "d"], [1, 0, 1, 0]
    discrim = eu.compute_discrimination_score(scores, nodes, {"a": {0}, "c": {1}}, y, k=2)
    norm = 1.0 + 1e-6
    assert math.isclose(discrim["discrim_score_att_0"], 1 / norm - 0.625 / norm)
    assert math.isclose(discrim["discrim_score_att_1"], 0.25 / norm - 0.625 / norm)
    tp = eu.compute_discrimination_tp(scores, nodes, {"a": {0}, "c": {1}}, y, k=2)
    assert tp["discrim_tp_att_0"] == 1 and tp["discrim_tp_att_1"] == 0


@pytest.mark.parametrize("seed", [0, 1])
def test_metrics_of_lazy_edges(seed):
    """The metrics are the same on the lazy edge sequence of the edge evaluation as on lists."""
    import numpy as np

    from pidsmaker.detection.evaluation_methods import evaluation_utils as eu
    from pidsmaker.detection.evaluation_methods.edge_evaluation import EdgeTuples

    rng = np.random.default_rng(seed)
    n = 3000
    edges = EdgeTuples(
        rng.integers(0, 40, n), rng.integers(0, 40, n), rng.integers(0, 100, n),
        rng.integers(1, 3, n), {1: "LogOn", 2: "LogOff"},
    )
    edge_list = list(edges)
    scores = np.round(rng.random(n), 1)  # many ties
    y = (rng.random(n) < 0.03).astype(np.int8)
    attack_to_mal_edges = {0: set(), 1: set()}
    for i in np.flatnonzero(y):
        attack_to_mal_edges[int(edges.time[i] >= 50)].add(edge_list[i])
    edge2attack = eu.transform_attack2nodes_to_node2attacks(attack_to_mal_edges)

    for fn, args in (
        (eu.plot_detected_attacks_vs_precision, (edge2attack, y, None)),
        (eu.compute_discrimination_score, (edge2attack, y)),
        (eu.compute_discrimination_tp, (edge2attack, y)),
    ):
        assert fn(scores, edges, *args) == fn(scores.tolist(), edge_list, *[a.tolist() if isinstance(a, np.ndarray) else a for a in args])
    assert eu.get_metrics_if_all_attacks_detected(scores, edges, attack_to_mal_edges) == (
        eu.get_metrics_if_all_attacks_detected(scores.tolist(), edge_list, attack_to_mal_edges)
    )
