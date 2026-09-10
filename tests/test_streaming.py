"""Functional tests for the real-time streaming layer.

These exercise the whole path a live provenance record takes - Avro encoding as
SPADE's Kafka storage produces it, decoding, translation into PIDSMaker's node and
edge model, and assembly into time window graphs - without needing a broker or a
database.
"""

import io
import json
import os
from types import SimpleNamespace

import pytest

from pidsmaker.streaming.adapters import build_adapter
from pidsmaker.streaming.adapters.spade import SpadeKafkaAdapter, parse_operation
from pidsmaker.streaming.decoders import build_decoder, load_schema
from pidsmaker.streaming.records import StreamEvent, StreamNode, parse_timestamp_to_ns
from pidsmaker.streaming.sources.file_source import FileSource
from pidsmaker.streaming.state import (
    apply_stream_state,
    load_stream_state,
    save_stream_state,
)
from pidsmaker.streaming.stream import ProvenanceStream

SCHEMA_NAME = "spade.storage.Kafka"

# A few seconds of activity as SPADE's Audit reporter would report it: bash forks
# curl, curl reads a config file and talks to a remote host.
PROCESS_BASH = {
    "type": "Process",
    "pid": "1000",
    "ppid": "1",
    "name": "bash",
    "exe": "/usr/bin/bash",
    "command line": "-bash",
    "source": "syscall",
}
PROCESS_CURL = {
    "type": "Process",
    "pid": "1001",
    "ppid": "1000",
    "name": "curl",
    "exe": "/usr/bin/curl",
    "command line": "curl https://example.com",
    "source": "syscall",
}
ARTIFACT_FILE = {
    "type": "Artifact",
    "subtype": "file",
    "path": "/etc/ssl/certs/ca-certificates.crt",
    "version": "0",
    "epoch": "0",
    "permissions": "0644",
}
ARTIFACT_SOCKET = {
    "type": "Artifact",
    "subtype": "network socket",
    "remote address": "93.184.216.34",
    "remote port": "443",
    "local address": "10.0.2.15",
    "local port": "51000",
    "protocol": "tcp",
}
AGENT = {"type": "Agent", "uid": "1000", "gid": "1000"}


def vertex_record(annotations, vertex_hash):
    return {"element": {"annotations": annotations, "hash": vertex_hash}}


def edge_record(annotations, child, parent, edge_hash):
    return {
        "element": {
            "annotations": annotations,
            "childVertexHash": child,
            "parentVertexHash": parent,
            "hash": edge_hash,
        }
    }


@pytest.fixture(scope="module")
def schema():
    return load_schema(SCHEMA_NAME)


def encode_avro(record, schema, branch=None):
    """Encodes a record the way SPADE's Kafka server writer does: bare Avro binary.

    `branch` names the union member to write, which is what SPADE's own writer
    selects from the record's Java class. Left unset, fastavro picks the first
    branch that fits - the `Edge` one, since its endpoint fields default to null -
    which is a useful second case to decode correctly.
    """
    import fastavro

    if branch is not None:
        record = {"element": (branch, record["element"])}
    buffer = io.BytesIO()
    fastavro.schemaless_writer(buffer, schema, record)
    return buffer.getvalue()


class TestSpadeAdapter:
    def test_process_vertex_becomes_a_subject(self):
        adapter = SpadeKafkaAdapter()
        (node,) = adapter.handle(vertex_record(PROCESS_BASH, "h_bash"))

        assert isinstance(node, StreamNode)
        assert node.node_type == "subject"
        assert node.attrs["path"] == "/usr/bin/bash"
        assert node.attrs["cmd_line"] == "-bash"

    def test_file_artifact_becomes_a_file(self):
        adapter = SpadeKafkaAdapter()
        (node,) = adapter.handle(vertex_record(ARTIFACT_FILE, "h_file"))

        assert node.node_type == "file"
        assert node.attrs["path"] == "/etc/ssl/certs/ca-certificates.crt"

    def test_network_socket_becomes_a_netflow(self):
        adapter = SpadeKafkaAdapter()
        (node,) = adapter.handle(vertex_record(ARTIFACT_SOCKET, "h_socket"))

        assert node.node_type == "netflow"
        assert node.attrs["remote_ip"] == "93.184.216.34"
        assert node.attrs["remote_port"] == "443"
        assert node.attrs["local_port"] == "51000"

    def test_agents_are_skipped(self):
        adapter = SpadeKafkaAdapter()

        assert adapter.handle(vertex_record(AGENT, "h_agent")) == []
        assert adapter.stats["skipped_vertex_Agent"] == 1

    def test_read_flows_from_the_file_to_the_process(self):
        # SPADE reports `Used(child=process, parent=artifact)`; PIDSMaker orients
        # edges along the information flow, so a read must come out as file -> subject.
        adapter = SpadeKafkaAdapter()
        adapter.handle(vertex_record(PROCESS_CURL, "h_curl"))
        adapter.handle(vertex_record(ARTIFACT_FILE, "h_file"))

        (event,) = adapter.handle(
            edge_record(
                {"type": "Used", "operation": "read", "time": "1756456789.123"},
                child="h_curl",
                parent="h_file",
                edge_hash="e1",
            )
        )

        assert isinstance(event, StreamEvent)
        assert event.src_key == adapter.hash_to_key["h_file"]
        assert event.dst_key == "h_curl"
        assert event.operation == "EVENT_READ"
        assert event.timestamp == 1756456789123000000

    def test_write_flows_from_the_process_to_the_file(self):
        adapter = SpadeKafkaAdapter()
        adapter.handle(vertex_record(PROCESS_CURL, "h_curl"))
        adapter.handle(vertex_record(ARTIFACT_FILE, "h_file"))

        (event,) = adapter.handle(
            edge_record(
                {"type": "WasGeneratedBy", "operation": "write", "time": "1756456790"},
                child="h_file",
                parent="h_curl",
                edge_hash="e2",
            )
        )

        assert event.src_key == "h_curl"
        assert event.dst_key == adapter.hash_to_key["h_file"]
        assert event.operation == "EVENT_WRITE"

    def test_fork_flows_from_the_parent_to_the_child(self):
        adapter = SpadeKafkaAdapter()
        adapter.handle(vertex_record(PROCESS_BASH, "h_bash"))
        adapter.handle(vertex_record(PROCESS_CURL, "h_curl"))

        (event,) = adapter.handle(
            edge_record(
                {"type": "WasTriggeredBy", "operation": "fork", "time": "1756456791"},
                child="h_curl",
                parent="h_bash",
                edge_hash="e3",
            )
        )

        assert (event.src_key, event.dst_key) == ("h_bash", "h_curl")
        assert event.operation == "EVENT_CLONE"

    def test_qualified_operations_are_mapped(self):
        assert parse_operation("open (read)") == ("open", "read")
        assert parse_operation("mmap (write)") == ("mmap", "write")
        assert parse_operation("read") == ("read", None)

        adapter = SpadeKafkaAdapter()
        adapter.handle(vertex_record(PROCESS_CURL, "h_curl"))
        adapter.handle(vertex_record(ARTIFACT_FILE, "h_file"))

        (event,) = adapter.handle(
            edge_record(
                {"type": "Used", "operation": "open (read)", "time": "1756456792"},
                child="h_curl",
                parent="h_file",
                edge_hash="e4",
            )
        )
        assert event.operation == "EVENT_OPEN"

    def test_unmapped_operations_are_dropped_by_default(self):
        adapter = SpadeKafkaAdapter()
        adapter.handle(vertex_record(PROCESS_BASH, "h_bash"))
        adapter.handle(vertex_record(PROCESS_CURL, "h_curl"))

        assert (
            adapter.handle(
                edge_record(
                    {"type": "WasTriggeredBy", "operation": "madvise", "time": "1756456793"},
                    child="h_curl",
                    parent="h_bash",
                    edge_hash="e5",
                )
            )
            == []
        )
        assert adapter.stats["skipped_operation_madvise"] == 1

    def test_unmapped_operations_are_kept_when_asked(self):
        adapter = SpadeKafkaAdapter(keep_unmapped_operations=True)
        adapter.handle(vertex_record(PROCESS_BASH, "h_bash"))
        adapter.handle(vertex_record(PROCESS_CURL, "h_curl"))

        (event,) = adapter.handle(
            edge_record(
                {"type": "WasTriggeredBy", "operation": "madvise", "time": "1756456793"},
                child="h_curl",
                parent="h_bash",
                edge_hash="e5",
            )
        )
        assert event.operation == "EVENT_OTHER"

    def test_artifact_versions_collapse_onto_one_node(self):
        # SPADE emits a fresh vertex per write when `versions=true`; without merging,
        # one file would look like a new node on every modification.
        adapter = SpadeKafkaAdapter(merge_artifact_versions=True)
        (v0,) = adapter.handle(vertex_record({**ARTIFACT_FILE, "version": "0"}, "h_v0"))
        (v1,) = adapter.handle(vertex_record({**ARTIFACT_FILE, "version": "1"}, "h_v1"))

        assert v0.key == v1.key

        unmerged = SpadeKafkaAdapter(merge_artifact_versions=False)
        (u0,) = unmerged.handle(vertex_record({**ARTIFACT_FILE, "version": "0"}, "h_v0"))
        (u1,) = unmerged.handle(vertex_record({**ARTIFACT_FILE, "version": "1"}, "h_v1"))
        assert u0.key != u1.key

    def test_version_edges_become_self_loops_and_are_dropped(self):
        adapter = SpadeKafkaAdapter(merge_artifact_versions=True)
        adapter.handle(vertex_record({**ARTIFACT_FILE, "version": "0"}, "h_v0"))
        adapter.handle(vertex_record({**ARTIFACT_FILE, "version": "1"}, "h_v1"))

        assert (
            adapter.handle(
                edge_record(
                    {"type": "WasDerivedFrom", "operation": "update", "time": "1756456794"},
                    child="h_v1",
                    parent="h_v0",
                    edge_hash="e6",
                )
            )
            == []
        )
        # Counted apart from genuine self-loops: on a real capture these are pure
        # version bookkeeping, and they can outnumber every other dropped edge.
        assert adapter.stats["skipped_edge_version_update"] == 1
        assert adapter.stats["skipped_edge_self_loop"] == 0

    def test_edges_with_unknown_endpoints_are_dropped(self):
        # What a detector joining a topic mid-stream sees: the vertices were
        # published before it subscribed.
        adapter = SpadeKafkaAdapter()
        assert (
            adapter.handle(
                edge_record(
                    {"type": "Used", "operation": "read", "time": "1756456795"},
                    child="never_seen",
                    parent="never_seen_either",
                    edge_hash="e7",
                )
            )
            == []
        )
        assert adapter.stats["skipped_edge_unknown_endpoint"] == 1

    def test_edges_without_a_time_are_dropped(self):
        adapter = SpadeKafkaAdapter()
        adapter.handle(vertex_record(PROCESS_BASH, "h_bash"))
        adapter.handle(vertex_record(PROCESS_CURL, "h_curl"))

        assert (
            adapter.handle(
                edge_record(
                    {"type": "WasTriggeredBy", "operation": "fork"},
                    child="h_curl",
                    parent="h_bash",
                    edge_hash="e8",
                )
            )
            == []
        )
        assert adapter.stats["skipped_edge_missing_time"] == 1


class TestDecoding:
    def test_avro_binary_roundtrip(self, schema):
        # The exact wire format of SPADE's Kafka server writer: bare Avro binary,
        # no schema envelope, decoded with the schema we ship.
        decoder = build_decoder("avro", SCHEMA_NAME)
        payload = encode_avro(
            vertex_record(PROCESS_BASH, "h_bash"), schema, branch="spade.storage.kafka.Vertex"
        )

        adapter = SpadeKafkaAdapter()
        (node,) = adapter.handle(decoder.decode(payload))
        assert node.node_type == "subject"
        assert node.attrs["path"] == "/usr/bin/bash"

    def test_avro_binary_edge_roundtrip(self, schema):
        decoder = build_decoder("avro", SCHEMA_NAME)
        adapter = SpadeKafkaAdapter()
        adapter.handle(decoder.decode(encode_avro(vertex_record(PROCESS_CURL, "h_curl"), schema)))
        adapter.handle(decoder.decode(encode_avro(vertex_record(ARTIFACT_FILE, "h_file"), schema)))

        payload = encode_avro(
            edge_record(
                {"type": "Used", "operation": "read", "time": "1756456789.123"},
                child="h_curl",
                parent="h_file",
                edge_hash="e1",
            ),
            schema,
            branch="spade.storage.kafka.Edge",
        )
        (event,) = adapter.handle(decoder.decode(payload))
        assert event.operation == "EVENT_READ"
        assert event.dst_key == "h_curl"

    def test_a_vertex_written_into_the_edge_branch_is_still_a_vertex(self):
        # Both endpoint fields default to null, so a writer that does not name the
        # union member produces a vertex carrying null endpoints. It must not be
        # mistaken for an edge.
        adapter = SpadeKafkaAdapter()
        (node,) = adapter.handle(
            {
                "element": {
                    "annotations": ARTIFACT_FILE,
                    "childVertexHash": None,
                    "parentVertexHash": None,
                    "hash": "h_file",
                }
            }
        )
        assert node.node_type == "file"

    def test_avro_json_is_understood(self):
        # What SPADE's file writer produces when its output path ends in `.json`.
        decoder = build_decoder("avro_json", SCHEMA_NAME)
        payload = json.dumps(
            {
                "element": {
                    "spade.storage.kafka.Vertex": {
                        "annotations": {"map": ARTIFACT_FILE},
                        "hash": "h_file",
                    }
                }
            }
        ).encode()

        adapter = SpadeKafkaAdapter()
        (node,) = adapter.handle(decoder.decode(payload))
        assert node.node_type == "file"
        assert node.attrs["path"] == ARTIFACT_FILE["path"]

    def test_plain_json_union_wrappers_are_unwrapped(self):
        # A plain JSON decoder leaves Avro's union tags in place; the adapter still
        # has to make sense of them.
        decoder = build_decoder("json")
        payload = json.dumps(
            {
                "element": {
                    "spade.storage.kafka.Edge": {
                        "annotations": {"map": {"type": "Used", "operation": "read", "time": "1"}},
                        "childVertexHash": {"string": "h_curl"},
                        "parentVertexHash": {"string": "h_file"},
                        "hash": "e9",
                    }
                }
            }
        ).encode()

        adapter = SpadeKafkaAdapter()
        adapter.handle(vertex_record(PROCESS_CURL, "h_curl"))
        adapter.handle(vertex_record(ARTIFACT_FILE, "h_file"))
        (event,) = adapter.handle(decoder.decode(payload))
        assert event.operation == "EVENT_READ"

    def test_unknown_format_is_rejected(self):
        with pytest.raises(ValueError):
            build_decoder("protobuf")


class TestTimestamps:
    @pytest.mark.parametrize(
        "value,expected",
        [
            ("1756456789.123", 1756456789123000000),  # SPADE audit: fractional seconds
            (1756456789, 1756456789000000000),  # seconds
            (1756456789123, 1756456789123000000),  # milliseconds
            (1756456789123456, 1756456789123456000),  # microseconds
            (1756456789123456789, 1756456789123456789),  # nanoseconds
        ],
    )
    def test_units_are_inferred(self, value, expected):
        assert parse_timestamp_to_ns(value) == expected

    @pytest.mark.parametrize("value", [None, "", "not-a-time", 0, -1])
    def test_unusable_values_fall_back(self, value):
        assert parse_timestamp_to_ns(value, default=None) is None


class TestFileSourceReplay:
    def test_replays_avro_json_lines_through_the_whole_chain(self, tmp_path):
        records = [
            {
                "element": {
                    "spade.storage.kafka.Vertex": {
                        "annotations": {"map": PROCESS_BASH},
                        "hash": "h_bash",
                    }
                }
            },
            {
                "element": {
                    "spade.storage.kafka.Vertex": {
                        "annotations": {"map": PROCESS_CURL},
                        "hash": "h_curl",
                    }
                }
            },
            {
                "element": {
                    "spade.storage.kafka.Edge": {
                        "annotations": {
                            "map": {
                                "type": "WasTriggeredBy",
                                "operation": "fork",
                                "time": "1756456789",
                            }
                        },
                        "childVertexHash": {"string": "h_curl"},
                        "parentVertexHash": {"string": "h_bash"},
                        "hash": "e1",
                    }
                }
            },
        ]
        capture = tmp_path / "capture.json"
        capture.write_text("\n".join(json.dumps(r) for r in records) + "\n")

        stream = ProvenanceStream(
            source=FileSource(str(capture), fmt="avro_json"),
            decoder=build_decoder("avro_json", SCHEMA_NAME),
            adapter=build_adapter("spade"),
        )
        results = [r for r in stream if r is not None]

        assert [type(r).__name__ for r in results] == ["StreamNode", "StreamNode", "StreamEvent"]
        assert results[-1].operation == "EVENT_CLONE"

    def test_a_corrupt_record_does_not_stop_the_stream(self, tmp_path):
        capture = tmp_path / "capture.json"
        capture.write_text(
            "\n".join(
                [
                    "{ this is not json",
                    json.dumps(
                        {
                            "element": {
                                "spade.storage.kafka.Vertex": {
                                    "annotations": {"map": PROCESS_BASH},
                                    "hash": "h_bash",
                                }
                            }
                        }
                    ),
                ]
            )
            + "\n"
        )

        stream = ProvenanceStream(
            source=FileSource(str(capture), fmt="avro_json"),
            decoder=build_decoder("avro_json", SCHEMA_NAME),
            adapter=build_adapter("spade"),
        )
        results = [r for r in stream if r is not None]

        assert len(results) == 1
        assert stream.stats["errors"] == 1

    def test_errors_can_be_made_fatal(self, tmp_path):
        capture = tmp_path / "capture.json"
        capture.write_text("{ this is not json\n")

        stream = ProvenanceStream(
            source=FileSource(str(capture), fmt="avro_json"),
            decoder=build_decoder("avro_json", SCHEMA_NAME),
            adapter=build_adapter("spade"),
            on_error="raise",
        )
        with pytest.raises(Exception):
            list(stream)


class TestStreamState:
    def test_state_lets_a_later_run_resolve_earlier_vertices(self, tmp_path):
        # A detector started after the capture never sees the vertices SPADE
        # published during it, unless it is handed the state the ingestion wrote.
        first = SpadeKafkaAdapter()
        first.handle(vertex_record(PROCESS_CURL, "h_curl"))
        first.handle(vertex_record(ARTIFACT_FILE, "h_file"))

        key_to_node = {
            "h_curl": (
                "subject",
                {"path": "/usr/bin/curl", "cmd_line": "curl https://example.com"},
            ),
            first.hash_to_key["h_file"]: ("file", {"path": ARTIFACT_FILE["path"]}),
        }
        state_file = tmp_path / "stream_state.pkl"
        save_stream_state(str(state_file), first, key_to_node)

        second = SpadeKafkaAdapter()
        assert (
            second.handle(
                edge_record(
                    {"type": "Used", "operation": "read", "time": "1756456789"},
                    child="h_curl",
                    parent="h_file",
                    edge_hash="e1",
                )
            )
            == []
        )

        state = load_stream_state(str(state_file))
        second.hash_to_key.update(state["hash_to_key"])

        (event,) = second.handle(
            edge_record(
                {"type": "Used", "operation": "read", "time": "1756456789"},
                child="h_curl",
                parent="h_file",
                edge_hash="e2",
            )
        )
        assert event.dst_key == "h_curl"

    def test_state_from_another_adapter_is_refused(self, tmp_path):
        adapter = SpadeKafkaAdapter()
        state_file = tmp_path / "stream_state.pkl"
        save_stream_state(str(state_file), adapter, {})

        state = load_stream_state(str(state_file))
        state["adapter"] = "camflow"
        with pytest.raises(ValueError):
            apply_stream_state(state, adapter, builder=None)

    def test_missing_state_file_is_not_an_error(self, tmp_path):
        assert load_stream_state(str(tmp_path / "nope.pkl")) is None


def make_construction_cfg(time_window_size=15.0, fuse_edge=True, use_hashed_label=False):
    """A config carrying just what the graph builder reads from `construction`."""
    from yacs.config import CfgNode as CN

    cfg = CN()
    cfg.construction = CN()
    cfg.construction.time_window_size = time_window_size
    cfg.construction.fuse_edge = fuse_edge
    cfg.construction.use_hashed_label = use_hashed_label
    cfg.construction.node_label_features = CN()
    cfg.construction.node_label_features.subject = "type, path, cmd_line"
    cfg.construction.node_label_features.file = "type, path"
    cfg.construction.node_label_features.netflow = "type, remote_ip, remote_port"
    return cfg


class TestGraphBuilder:
    MINUTE = 60_000_000_000

    def build(self, **kwargs):
        from pidsmaker.streaming.graph_builder import StreamingGraphBuilder

        return StreamingGraphBuilder(make_construction_cfg(**kwargs))

    def add_nodes(self, builder):
        builder.add_node(StreamNode("p", "subject", {"path": "/usr/bin/curl", "cmd_line": "curl"}))
        builder.add_node(StreamNode("f", "file", {"path": "/etc/passwd"}))
        builder.add_node(StreamNode("n", "netflow", {"remote_ip": "8.8.8.8", "remote_port": "53"}))

    def test_labels_match_the_offline_ones(self):
        # The featurizers were trained on labels built this way by the construction
        # task; a live window has to produce byte-identical ones.
        builder = self.build()
        self.add_nodes(builder)

        assert builder.indexid2msg["0"] == ["subject", "subject /usr/bin/curl curl"]
        assert builder.indexid2msg["1"] == ["file", "file /etc/passwd"]
        assert builder.indexid2msg["2"] == ["netflow", "netflow 8.8.8.8 53"]

    def test_hashed_labels(self):
        builder = self.build(use_hashed_label=True)
        self.add_nodes(builder)

        label = builder.indexid2msg["1"][1]
        assert label != "file /etc/passwd"
        assert set(label) <= set("0123456789abcdef")

    def test_a_window_closes_when_an_event_crosses_its_boundary(self):
        builder = self.build(time_window_size=15.0)
        self.add_nodes(builder)
        start = 1_756_456_789_000_000_000

        assert builder.add_event(StreamEvent("e1", "f", "p", "EVENT_READ", start)) is None
        assert builder.add_event(StreamEvent("e2", "f", "p", "EVENT_OPEN", start + 60)) is None

        window = builder.add_event(
            StreamEvent("e3", "p", "n", "EVENT_CONNECT", start + 16 * self.MINUTE)
        )
        assert window is not None
        assert window.graph.number_of_edges() == 2
        assert window.num_events == 2
        # The event that closed the window belongs to the next one.
        assert builder.num_pending_events == 1

    def test_flush_returns_the_pending_window(self):
        builder = self.build()
        self.add_nodes(builder)
        builder.add_event(StreamEvent("e1", "f", "p", "EVENT_READ", 1_756_456_789_000_000_000))

        window = builder.flush()
        assert window.graph.number_of_edges() == 1
        assert builder.flush() is None

    def test_edges_are_fused_like_the_offline_construction(self):
        # A process reading the same file a hundred times in a row is one edge.
        builder = self.build(fuse_edge=True)
        self.add_nodes(builder)
        start = 1_756_456_789_000_000_000
        for i in range(5):
            builder.add_event(StreamEvent(f"e{i}", "f", "p", "EVENT_READ", start + i))
        builder.add_event(StreamEvent("e5", "f", "p", "EVENT_OPEN", start + 10))

        window = builder.flush()
        assert window.graph.number_of_edges() == 2
        assert window.num_events == 6

    def test_fusion_can_be_turned_off(self):
        builder = self.build(fuse_edge=False)
        self.add_nodes(builder)
        start = 1_756_456_789_000_000_000
        for i in range(5):
            builder.add_event(StreamEvent(f"e{i}", "f", "p", "EVENT_READ", start + i))

        assert builder.flush().graph.number_of_edges() == 5

    def test_graph_carries_the_attributes_the_pipeline_expects(self):
        builder = self.build()
        self.add_nodes(builder)
        start = 1_756_456_789_000_000_000
        builder.add_event(StreamEvent("e1", "f", "p", "EVENT_READ", start))

        graph = builder.flush().graph
        node_attrs = graph.nodes["1"]
        assert set(node_attrs) == {"node_type", "label"}

        _, _, edge_attrs = list(graph.edges(data=True))[0]
        assert set(edge_attrs) == {"event_uuid", "time", "label", "y"}
        assert edge_attrs["label"] == "EVENT_READ"
        assert edge_attrs["y"] == 0

    def test_out_of_order_events_join_the_open_window(self):
        builder = self.build()
        self.add_nodes(builder)
        start = 1_756_456_789_000_000_000
        builder.add_event(StreamEvent("e1", "f", "p", "EVENT_READ", start))
        builder.add_event(StreamEvent("e2", "p", "n", "EVENT_CONNECT", start - 1_000_000_000))

        window = builder.flush()
        assert window.graph.number_of_edges() == 2
        assert builder.num_late_events == 1
        # Events are sorted by time before the graph is built.
        assert window.start_ns < start

    def test_events_with_unknown_nodes_are_ignored(self):
        builder = self.build()
        self.add_nodes(builder)

        assert builder.add_event(StreamEvent("e", "ghost", "p", "EVENT_READ", 1)) is None
        assert builder.num_pending_events == 0

    def test_node_state_round_trips(self):
        builder = self.build()
        self.add_nodes(builder)

        state = builder.node_state()
        assert state["f"] == ("file", {"path": "/etc/passwd"})
        assert set(state) == {"p", "f", "n"}


class TestRealSpadeOutput:
    """Replays output produced by SPADE itself, to pin the wire format.

    `tests/fixtures/spade_kafka_sample.json` is the reference output of SPADE's Kafka
    storage file writer, taken from `vagrant/kafka/data/kafka-expected-output.json`
    in the SPADE repository: two processes, a file artifact, and the edges between
    them. If SPADE's schema or encoding ever changes, this test is what notices.
    """

    SAMPLE = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "fixtures", "spade_kafka_sample.json"
    )

    def test_sample_replays_into_nodes_and_events(self):
        stream = ProvenanceStream(
            source=FileSource(self.SAMPLE, fmt="avro_json"),
            decoder=build_decoder("avro_json", SCHEMA_NAME),
            adapter=build_adapter("spade"),
        )
        records = [r for r in stream if r is not None]

        nodes = [r for r in records if isinstance(r, StreamNode)]
        events = [r for r in records if isinstance(r, StreamEvent)]

        assert [n.node_type for n in nodes] == ["subject", "subject", "file"]
        assert nodes[2].attrs["path"] == "/etc/hostname"

        # `Used(child=cat, parent=/etc/hostname)` with operation `read`: the file is
        # the source of the flow, the process its destination.
        read = events[1]
        assert read.operation == "EVENT_READ"
        assert read.src_key == nodes[2].key
        assert read.dst_key == nodes[1].key


class TestStreamingVizCollector:
    """The collector that feeds real-time predictions to the web embedding viewer.

    These pin the data contract — the per-(node, window) records and the flat
    `stream_scores.jsonl` — without invoking the heavy dimensionality-reduction
    pass in `finalize()`, which is exercised end to end elsewhere.
    """

    @staticmethod
    def _cfg():
        return SimpleNamespace(
            dataset=SimpleNamespace(name="SPADE_AUDIT"),
            featurization=SimpleNamespace(used_method="word2vec"),
            _model="orthrus",
        )

    @staticmethod
    def _window(interval, edges):
        """A minimal stand-in for a TimeWindow: an interval and a networkx graph."""
        import networkx as nx

        g = nx.MultiDiGraph()
        for u, v, op in edges:
            g.add_edge(u, v, time=1, label=op)
        return SimpleNamespace(interval=interval, graph=g)

    def _collector(self, tmp_path):
        from pidsmaker.streaming.viz import StreamingVizCollector

        return StreamingVizCollector(self._cfg(), artifact_dir=str(tmp_path))

    def test_collects_one_snapshot_per_node_per_window(self, tmp_path):
        col = self._collector(tmp_path)
        indexid2msg = {"1": ["subject", "/usr/bin/curl"], "2": ["file", "/etc/passwd"]}
        vecs = {1: [0.1, 0.2, 0.3], 2: [0.4, 0.5, 0.6]}

        col.add_window(
            window=self._window("w0", [(1, 2, "EVENT_READ")]),
            indexid2vec=vecs,
            node_scores={1: 2.5, 2: 0.4},
            alert_node_ids={1},
            indexid2msg=indexid2msg,
            index_to_key={"1": "hashA", "2": "file:hashB"},
        )
        col.add_window(
            window=self._window("w1", [(1, 2, "EVENT_WRITE")]),
            indexid2vec=vecs,
            node_scores={1: 1.0, 2: 3.0},
            alert_node_ids={2},
            indexid2msg=indexid2msg,
            index_to_key={},
        )

        # One embedding snapshot per node per window; node_meta keyed by int id.
        assert len(col.embeddings) == 4
        assert col.node_meta[1] == {
            "path": "/usr/bin/curl",
            "type": "subject",
            "cmd": "/usr/bin/curl",
        }
        # Edges carry the window index and the operation label.
        assert (1, 2, 0, "EVENT_READ") in col.edges
        assert (1, 2, 1, "EVENT_WRITE") in col.edges
        # The alerting node's snapshot is flagged; a below-threshold node is not.
        w0 = {e.node_id: e for e in col.embeddings if e.time_window_idx == 0}
        assert w0[1].detection_status == 1 and w0[1].anomaly_score == 2.5
        assert w0[2].detection_status == 0

    def test_scores_jsonl_has_every_node_every_window(self, tmp_path):
        col = self._collector(tmp_path)
        col.add_window(
            window=self._window("w0", [(1, 2, "EVENT_READ")]),
            indexid2vec={1: [0.1, 0.2], 2: [0.3, 0.4]},
            node_scores={1: 2.5, 2: 0.4},
            alert_node_ids={1},
            indexid2msg={"1": ["subject", "curl"], "2": ["file", "/etc/passwd"]},
            index_to_key={"1": "hashA"},
        )
        col._scores_fh.flush()

        rows = [json.loads(line) for line in open(col.scores_path)]
        assert len(rows) == 2  # both nodes, not just the alert
        alert_row = next(r for r in rows if r["node"] == 1)
        assert alert_row["is_alert"] is True
        assert alert_row["score"] == 2.5
        assert alert_row["source_key"] == "hashA"
        assert next(r for r in rows if r["node"] == 2)["is_alert"] is False

    def test_type_only_featurizer_still_records_scores(self, tmp_path):
        col = self._collector(tmp_path)
        col.add_window(
            window=self._window("w0", [(1, 2, "EVENT_READ")]),
            indexid2vec=None,  # type-only featurizer produces no embeddings
            node_scores={1: 2.5, 2: 0.4},
            alert_node_ids={1},
            indexid2msg={"1": ["subject", "curl"], "2": ["file", "/etc/passwd"]},
            index_to_key={},
        )
        col._scores_fh.flush()

        assert col.embeddings == []  # nothing to lay out in 3D
        assert len(list(open(col.scores_path))) == 2  # but scores are still recorded
        assert col.finalize() is None  # no viewer produced, no crash


class TestStreamedDatasetNames:
    """A streamed dataset can have any name; its dataset.yml carries the definition."""

    def _write(self, tmp_path, **extra):
        import yaml

        cfg = {
            "name": "TUTORIAL",
            "template": "SPADE_AUDIT",
            "database": "tutorial",
            "database_all_file": "tutorial",
            "start_date": "2026-09-10",
            "end_date": "2026-09-11",
            "unused_dates": [],
            "train_dates": ["2026-09-10"],
            "val_dates": ["2026-09-10"],
            "test_dates": ["2026-09-10"],
        }
        cfg.update(extra)
        path = tmp_path / "dataset.yml"
        path.write_text(yaml.safe_dump(cfg))
        return str(path)

    def test_unknown_name_inherits_its_template(self, tmp_path):
        from yacs.config import CfgNode as CN

        from pidsmaker.config.pipeline import set_dataset_cfg
        from pidsmaker.utils.dataset_utils import get_rel2id, rel2id_spade

        cfg = CN()
        set_dataset_cfg(cfg, "TUTORIAL", self._write(tmp_path))
        assert cfg.dataset.name == "TUTORIAL"
        assert cfg.dataset.template == "SPADE_AUDIT"
        assert cfg.dataset.database == "tutorial"
        assert cfg.dataset.num_edge_types == 28
        assert cfg.dataset.train_dates == ["2026-09-10"]
        assert get_rel2id(cfg) == rel2id_spade

    def test_unknown_name_without_config_is_explained(self):
        import pytest
        from yacs.config import CfgNode as CN

        from pidsmaker.config.pipeline import set_dataset_cfg

        with pytest.raises(ValueError, match="stream_ingest.py TUTORIAL"):
            set_dataset_cfg(CN(), "TUTORIAL")

    def test_builtin_name_still_works_and_is_its_own_template(self):
        from yacs.config import CfgNode as CN

        from pidsmaker.config.pipeline import set_dataset_cfg

        cfg = CN()
        set_dataset_cfg(cfg, "SPADE_AUDIT")
        assert cfg.dataset.template == "SPADE_AUDIT"
        assert cfg.dataset.database == "spade_audit"

    def test_database_name_is_identifier_safe(self):
        from pidsmaker.config.pipeline import streamed_database_name

        assert streamed_database_name("My-Host.2") == "my_host_2"
