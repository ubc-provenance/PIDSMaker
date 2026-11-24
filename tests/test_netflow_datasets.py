"""Tests of the NetFlow v3 datasets (e.g. CIC_IDS2018) and of the `netflow_v3` schema.

The tests run on a small synthetic dataset in the NetFlow v3 format (see
`synthetic_netflow_v3.py`), loaded by the NetFlow v3 preprocessing script into test databases, so
that they don't require the real dataset.
"""

import importlib.util
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import psycopg2
import pytest
import torch
import wandb

from pidsmaker import main
from pidsmaker.config import DATASET_DEFAULT_CONFIG, get_runtime_required_args, get_yml_cfg
from pidsmaker.config.schemas import NETFLOW_PORTS, NETFLOW_PROTOCOLS, SCHEMAS, get_schema
from pidsmaker.utils.dataset_utils import encode_edge_attributes, get_edge_attrs_dim

sys.path.insert(0, os.path.dirname(__file__))
import synthetic_netflow_v3  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "dataset_preprocessing/netflow/create_database_netflow_v3.py")
TESTS_ARTIFACT_DIR = "/home/artifacts/tests_netflow/"
TEST_DATABASE = "cic_ids2018_pytest"
LOADER_TEST_DATABASE = "cic_ids2018_loader_pytest"
NIDS_SYSTEMS = ["euler", "argus", "jbeil", "lares", "egraphsage", "anomal_e"]


def prepare_cfg(model, dataset="CIC_IDS2018", custom_args=(), test_mode=True):
    args, _ = get_runtime_required_args(return_unknown_args=True, args=[model, dataset])
    args.__dict__["artifact_dir"] = TESTS_ARTIFACT_DIR
    for k, v in custom_args:
        args.__dict__[k] = v
    cfg = get_yml_cfg(args)
    cfg._test_mode = test_mode
    return cfg


def load(path, database, chunk_size=1_000_000):
    subprocess.run(
        [sys.executable, SCRIPT, "CIC_IDS2018", "--input", path, "--database", database,
         "--chunk_size", str(chunk_size), "--overwrite"],
        check=True,
        cwd=ROOT,
    )  # fmt: skip


def query(database, sql):
    cfg = prepare_cfg("velox")
    connect = psycopg2.connect(
        dbname=database, host=cfg.database.host, user=cfg.database.user,
        password=cfg.database.password, port=cfg.database.port,
    )  # fmt: skip
    cur = connect.cursor()
    cur.execute(sql)
    rows = cur.fetchall()
    connect.close()
    return rows


def loader_module():
    spec = importlib.util.spec_from_file_location("create_database_netflow_v3", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def flows():
    return synthetic_netflow_v3.generate()


@pytest.fixture(scope="module", autouse=True)
def cic_database(tmp_path_factory, flows):
    """Loads the synthetic flows into the test database, used by the CIC_IDS2018 dataset."""
    path = synthetic_netflow_v3.write(flows, str(tmp_path_factory.mktemp("cic") / "flows.csv"))
    load(path, TEST_DATABASE)
    original = dict(DATASET_DEFAULT_CONFIG["CIC_IDS2018"])
    DATASET_DEFAULT_CONFIG["CIC_IDS2018"]["database"] = TEST_DATABASE
    DATASET_DEFAULT_CONFIG["CIC_IDS2018"]["database_all_file"] = TEST_DATABASE
    shutil.rmtree(TESTS_ARTIFACT_DIR, ignore_errors=True)
    wandb.init(mode="disabled")
    yield
    DATASET_DEFAULT_CONFIG["CIC_IDS2018"].update(original)
    shutil.rmtree(TESTS_ARTIFACT_DIR, ignore_errors=True)


class TestNetflowV3Schema:
    def test_schema(self):
        schema = SCHEMAS["netflow_v3"]
        assert DATASET_DEFAULT_CONFIG["CIC_IDS2018"]["schema"] == "netflow_v3"
        assert schema.num_node_types == 1 and schema.num_edge_types == len(NETFLOW_PROTOCOLS) == 5
        # Edge types are the transport protocols, with ids starting at 1
        assert [schema.rel2id[p] for p in NETFLOW_PROTOCOLS] == [1, 2, 3, 4, 5]
        assert schema.rel2id[1] == "TCP" and schema.rel2id["OTHER"] == 5
        assert schema.label_column == "label"
        attributes = {a.name: a for a in schema.edge_attributes}
        numeric = ["in_bytes", "out_bytes", "in_pkts", "out_pkts", "duration"]
        assert list(attributes) == ["src_port", "dst_port"] + numeric
        assert all(a.column == a.name for a in schema.edge_attributes)
        assert attributes["src_port"].kind == attributes["dst_port"].kind == "categorical"
        assert attributes["dst_port"].values == list(NETFLOW_PORTS) + ["OTHER"]
        assert all(attributes[n].kind == "numeric" for n in numeric)

    def test_dates(self):
        cic = DATASET_DEFAULT_CONFIG["CIC_IDS2018"]
        splits = [cic["train_dates"], cic["val_dates"], cic["test_dates"], cic["unused_dates"]]
        dates = [d for split in splits for d in split]
        # The splits are disjoint and cover all the days of the dataset
        assert len(dates) == len(set(dates)) == 17
        assert min(dates) == cic["start_date"] and max(dates) == cic["end_date"]
        assert set(synthetic_netflow_v3.DAYS) == set(dates) - set(cic["unused_dates"])

    def test_edge_attribute_encoding(self):
        cfg = prepare_cfg("euler")
        columns = {
            "src_port": ["49152+", "1-1023", None],
            "dst_port": ["443", "unknown", "0"],
            "in_bytes": [100.0, 0.0, None],
            "out_bytes": [10.0, 5.0, None],
            "in_pkts": [2.0, 1.0, None],
            "out_pkts": [1.0, 1.0, None],
            "duration": [0.0, 3.0, None],
        }
        feats = encode_edge_attributes(cfg, columns)
        num_ports = len(NETFLOW_PORTS) + 1
        assert feats.shape == (3, 2 * num_ports + 5) and get_edge_attrs_dim(cfg) == feats.shape[1]
        ports = list(NETFLOW_PORTS) + ["OTHER"]
        # One-hot ports, unknown and null values are OTHER
        assert feats[0, ports.index("49152+")] == 1 and feats[0, num_ports + ports.index("443")] == 1
        assert feats[1, num_ports + ports.index("OTHER")] == 1 and feats[2, ports.index("OTHER")] == 1
        assert torch.all(feats[:, : 2 * num_ports].sum(dim=1) == 2)
        # Numeric attributes are log1p, null values are 0
        assert torch.allclose(feats[0, -5:], torch.log1p(torch.tensor([100.0, 10.0, 2.0, 1.0, 0.0])))
        assert torch.all(feats[2, -5:] == 0)


class TestNetflowV3Loader:
    def test_port_categories(self):
        categories = loader_module().port_categories(
            pd.Series([0, 22, 53, 111, 443, 1023, 1024, 3389, 8081, 49151, 49152, 65535, np.nan])
        )
        assert list(categories) == [
            "0", "22", "53", "1-1023", "443", "1-1023", "1024-49151", "3389", "1024-49151",
            "1024-49151", "49152+", "49152+", None,
        ]  # fmt: skip
        # Every category is in the vocabulary of the ports
        assert set(categories[:-1]) <= set(NETFLOW_PORTS)

    @pytest.mark.parametrize("extension", ["csv", "csv.gz", "zip", "parquet"])
    def test_load(self, tmp_path, flows, extension):
        if extension == "parquet":
            pytest.importorskip("pyarrow")
        path = synthetic_netflow_v3.write(flows, str(tmp_path / f"flows.{extension}"))
        # Small chunks: hosts are numbered consistently across chunks
        load(path, LOADER_TEST_DATABASE, chunk_size=500)
        columns = (
            "src_node, src_index_id, operation, dst_node, dst_index_id, event_uuid, timestamp_rec, "
            "src_port, dst_port, in_bytes, out_bytes, in_pkts, out_pkts, duration, label"
        )
        rows = query(
            LOADER_TEST_DATABASE, f"SELECT {columns} FROM event_table ORDER BY event_uuid::int"
        )
        assert len(rows) == len(flows)
        (src, src_id, operation, dst, dst_id, uuid, t, src_port, dst_port, in_bytes, out_bytes,
         in_pkts, out_pkts, duration, label) = map(list, zip(*rows))  # fmt: skip
        # Flows in file order, with their hosts, times (ns), numeric attributes and labels
        assert uuid == [str(i) for i in range(len(flows))]
        assert src == flows["IPV4_SRC_ADDR"].tolist() and dst == flows["IPV4_DST_ADDR"].tolist()
        assert t == (flows["FLOW_START_MILLISECONDS"] * 1_000_000).tolist()
        assert in_bytes == flows["IN_BYTES"].tolist() and out_bytes == flows["OUT_BYTES"].tolist()
        assert in_pkts == flows["IN_PKTS"].tolist() and out_pkts == flows["OUT_PKTS"].tolist()
        assert duration == flows["FLOW_DURATION_MILLISECONDS"].tolist()
        assert label == flows["Label"].tolist() and sum(label) == 130
        # Edge types are the protocols (IGMP is OTHER)
        protocols = {6: "TCP", 17: "UDP", 1: "ICMP", 58: "ICMPv6", 2: "OTHER"}
        assert operation == flows["PROTOCOL"].map(protocols).tolist()
        assert set(operation) == set(NETFLOW_PROTOCOLS)
        # Service ports are kept, other ports are grouped by range
        port2category = {p: str(p) for p in (0, 22, 53, 80, 443, 445)}
        port2category.update({111: "1-1023", 8081: "1024-49151"})
        assert dst_port == flows["L4_DST_PORT"].map(port2category).tolist()
        assert set(src_port) == {"0", "1024-49151", "49152+"}

        # Hosts are numbered in order of appearance, and events refer to them by index
        hosts = query(
            LOADER_TEST_DATABASE,
            "SELECT node_uuid, name, index_id FROM host_node_table ORDER BY index_id",
        )
        ips = list(dict.fromkeys(flows[["IPV4_SRC_ADDR", "IPV4_DST_ADDR"]].to_numpy().ravel()))
        assert [h[0] for h in hosts] == [h[1] for h in hosts] == ips
        assert [h[2] for h in hosts] == list(range(len(ips)))
        ip2id = {ip: i for i, ip in enumerate(ips)}
        assert list(map(int, src_id)) == [ip2id[ip] for ip in src]
        assert list(map(int, dst_id)) == [ip2id[ip] for ip in dst]


class TestCIC:
    @pytest.mark.parametrize("benign", [True, False])
    def test_benign_training(self, tmp_path, flows, benign):
        """With `benign_training`, the graphs of train/val dates have no malicious flow."""
        import glob

        from pidsmaker.preprocessing.build_graph_methods import build_default_graphs
        from pidsmaker.preprocessing.cleaning import Cleaner
        from pidsmaker.utils.graph_store import load_graph

        # Test mode: graphs of the first date of each split (2018-02-14, 2018-02-20, 2018-02-21)
        cfg = prepare_cfg("euler", custom_args=[("construction.benign_training", benign)])
        cfg.construction._graphs_dir = str(tmp_path)
        cleaner = Cleaner(cfg)
        indexid2msg = build_default_graphs.compute_indexid2msg(cfg=cfg, cleaner=cleaner)
        build_default_graphs.gen_edge_fused_tw(
            indexid2msg=indexid2msg, cfg=cfg, cleaner=cleaner, file2nodes={}
        )

        def edges(date):
            paths = glob.glob(os.path.join(str(tmp_path), f"graph_{date}", "*"))
            return [d for p in paths for *_, d in load_graph(p).edges(data=True)]

        # The SSH brute force of the train date is dropped, the DDoS of the test date is kept
        train, test = edges("2018-02-14"), edges("2018-02-21")
        assert sum(d["y"] for d in train) == (0 if benign else 40)
        assert sum(d["y"] for d in test) == 60
        # All the other flows of the day are kept (except self-loops, removed by Euler's cleaning)
        time = pd.to_datetime(flows["FLOW_START_MILLISECONDS"], unit="ms", utc=True)
        day = time.dt.tz_convert("Etc/GMT+4").dt.strftime("%Y-%m-%d")
        kept = flows[(day == "2018-02-14") & (flows["IPV4_SRC_ADDR"] != flows["IPV4_DST_ADDR"])]
        assert len(train) == len(kept) - (kept["Label"].sum() if benign else 0)
        # Flows carry their protocol and attributes
        assert {d["label"] for d in test} <= set(NETFLOW_PROTOCOLS)
        assert all(d["dst_port"] in NETFLOW_PORTS for d in test)

    @pytest.mark.parametrize("system", NIDS_SYSTEMS)
    def test_systems(self, system):
        main.main(prepare_cfg(system))

    def test_labels_from_events(self):
        from pidsmaker.utils.labelling import get_malicious_events

        cfg = prepare_cfg("euler")
        # All the malicious flows of the database, sorted by time
        events = get_malicious_events(cfg)
        assert len(events) == 130 and [e[2] for e in events] == sorted(e[2] for e in events)
        assert get_schema(cfg).name == "netflow_v3"

