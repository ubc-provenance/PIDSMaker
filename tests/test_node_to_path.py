"""Tests of the node -> path mapping of the evaluation (only the nodes of the graphs are
loaded, the others are read from the database on access), with a fake database."""

import pickle
from types import SimpleNamespace

import pytest

from pidsmaker.config.schemas import NodeTypeSchema
from pidsmaker.utils import dataset_utils
from pidsmaker.utils.dataset_utils import NodeToPathType, _node_info

pytestmark = pytest.mark.no_database

SUBJECT = NodeTypeSchema(
    name="subject", id=1, table="subject_node_table",
    columns=(("path", "path"), ("cmd_line", "cmd")), display_extra=(("cmd", "cmd_line"),),
)
FILE = NodeTypeSchema(name="file", id=2, table="file_node_table", columns=(("path", "path"),))
ROWS = {
    "subject_node_table": {1: (1, "/bin/sh", "sh -c x"), 2: (2, "/bin/ls", "ls")},
    "file_node_table": {3: (3, "/etc/passwd")},
}


class FakeCursor:
    def __init__(self, queries):
        self.queries, self.row = queries, None

    def execute(self, sql, params):
        table = sql.split(" FROM ")[1].split()[0]
        self.queries.append((table, params[0]))
        self.row = ROWS[table].get(params[0])

    def fetchone(self):
        return self.row


@pytest.fixture
def mapping(monkeypatch):
    queries = []

    class FakeConnection:
        def close(self):
            pass

    monkeypatch.setattr(
        dataset_utils, "init_database_connection", lambda cfg: (FakeCursor(queries), FakeConnection())
    )
    monkeypatch.setattr(
        dataset_utils, "get_schema", lambda cfg: SimpleNamespace(node_types=[SUBJECT, FILE])
    )
    loaded = {1: _node_info(SUBJECT, ROWS["subject_node_table"][1])}
    return NodeToPathType(cfg=None, data=loaded), queries


def test_loaded_and_database_nodes(mapping):
    node_to_path, queries = mapping
    assert node_to_path[1] == {"path": "/bin/sh", "type": "subject", "cmd": "sh -c x"}
    assert queries == []  # loaded nodes don't query the database
    assert node_to_path[3] == {"path": "/etc/passwd", "type": "file"}
    assert node_to_path.get(2) == {"path": "/bin/ls", "type": "subject", "cmd": "ls"}
    num_queries = len(queries)
    assert node_to_path[3]["path"] == "/etc/passwd"  # read once, then kept
    assert len(queries) == num_queries


def test_absent_nodes(mapping):
    node_to_path, queries = mapping
    with pytest.raises(KeyError):
        node_to_path[42]
    assert node_to_path.get(42, "?") == "?"
    assert node_to_path.get("not an id") is None
    assert len(queries) == 2  # one query per node type, once


def test_pickled_as_dict(mapping):
    node_to_path, _ = mapping
    loaded = pickle.loads(pickle.dumps(node_to_path))
    assert type(loaded) is dict and loaded == dict(node_to_path)
