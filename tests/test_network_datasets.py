"""Tests of network datasets (e.g. LANL) and of the dataset schemas.

The LANL tests run on a small synthetic dataset in the LANL format (see `synthetic_lanl.py`),
loaded into the `lanl_pytest` database by the LANL preprocessing script, so that they don't
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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS_ARTIFACT_DIR = "/home/artifacts/tests_network/"
TEST_DATABASE = "lanl_pytest"


def prepare_cfg(model, dataset="LANL", custom_args=(), test_mode=True):
    args, _ = get_runtime_required_args(return_unknown_args=True, args=[model, dataset])
    args.__dict__["artifact_dir"] = TESTS_ARTIFACT_DIR
    for k, v in custom_args:
        args.__dict__[k] = v
    cfg = get_yml_cfg(args)
    cfg._test_mode = test_mode
    return cfg


@pytest.fixture(scope="module", autouse=True)
def lanl_database(tmp_path_factory):
    """Loads the synthetic LANL dataset into the test database, used by the LANL dataset."""
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
    systems = ["velox", "orthrus", "kairos", "magic", "ocrapt"] + NIDS_SYSTEMS

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

    def test_edge_autoencoder(self):
        import torch
        from types import SimpleNamespace

        from pidsmaker.objectives.autoencode_edge_emb import EdgeEmbAutoencoder

        obj = EdgeEmbAutoencoder(node_dim=8, hid_dim=16, edge_dim=3)
        h_src, h_dst = torch.randn(5, 8), torch.randn(5, 8)
        batch = SimpleNamespace(edge_feats=torch.rand(5, 3))
        assert obj(h_src, h_dst, inference=False, batch=batch)["loss"].dim() == 0
        assert obj(h_src, h_dst, inference=True, batch=batch)["loss"].shape == (5,)

    def test_two_stage_host_selection(self):
        from pidsmaker.detection.evaluation_methods.host_two_stage_evaluation import select_hosts

        scores = {1: 0.1, 2: 0.2, 3: 5.0, 4: 5.2, 5: 0.3}
        # Top cluster of the 3 highest scores (3, 4) and the host ranked just below them (5)
        assert sorted(select_hosts(scores, alpha=3)) == [3, 4, 5]
