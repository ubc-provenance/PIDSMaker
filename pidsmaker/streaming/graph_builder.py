"""Assembles a stream of canonical records into time-window provenance graphs.

This is the streaming counterpart of `preprocessing/build_graph_methods/build_default_graphs.py`:
it produces the very same `networkx.MultiDiGraph` objects, with the same node
attributes (`node_type`, `label`), the same edge attributes (`time`, `label`,
`event_uuid`, `y`) and the same edge fusion - except it builds them incrementally
as events arrive rather than by querying a database over a fixed date range.

Because the graphs are identical, everything downstream of construction
(featurization, batching, the model) works on a live stream unchanged.
"""

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import networkx as nx

from pidsmaker.config import get_darpa_tc_node_feats_from_cfg
from pidsmaker.preprocessing.build_graph_methods.build_default_graphs import fuse_edges
from pidsmaker.streaming.records import StreamEvent, StreamNode
from pidsmaker.utils.utils import build_node_label, log, ns_time_to_datetime_US

MINUTE_IN_NS = 60_000_000_000


@dataclass
class TimeWindow:
    """One closed time window, ready to be scored or persisted."""

    graph: nx.MultiDiGraph
    start_ns: int
    end_ns: int
    num_events: int

    @property
    def interval(self) -> str:
        """The window's name, in the `start~end` form used for offline graph files."""
        return f"{ns_time_to_datetime_US(self.start_ns)}~{ns_time_to_datetime_US(self.end_ns)}"

    def __repr__(self):
        return (
            f"TimeWindow({self.interval}, nodes={self.graph.number_of_nodes()}, "
            f"edges={self.graph.number_of_edges()}, events={self.num_events})"
        )


class StreamingGraphBuilder:
    """Turns `StreamNode`/`StreamEvent` records into time-window graphs.

    Nodes are assigned incremental integer indices the first time they are seen,
    mirroring the `index_id` column of the offline postgres schema, and their
    textual label is computed with the same `construction.node_label_features`
    the offline pipeline uses.

    A window closes as soon as an event arrives more than `time_window_size`
    minutes after the window started. On a quiet host no such event may come for
    a long while, so `flush_timeout` additionally closes a pending window after
    that many seconds of wall clock - without it, a detector could sit on a
    half-filled window indefinitely and never alert.

    Args:
        cfg: Full pipeline config (uses `construction.*`).
        window_size_minutes: Overrides `construction.time_window_size`.
        flush_timeout: Seconds of wall clock after which a pending window is
            closed anyway (0 disables).
        max_late_events: How many out-of-order events to report before staying
            quiet about them.
    """

    def __init__(
        self,
        cfg,
        window_size_minutes: Optional[float] = None,
        flush_timeout: float = 0.0,
        max_late_events: int = 10,
    ):
        self.cfg = cfg
        window_size = (
            window_size_minutes
            if window_size_minutes is not None
            else cfg.construction.time_window_size
        )
        self.window_size_ns = int(float(window_size) * MINUTE_IN_NS)
        self.flush_timeout = flush_timeout
        self.fuse_edge = cfg.construction.fuse_edge
        self.use_hashed_label = cfg.construction.use_hashed_label
        self.node_label_features = get_darpa_tc_node_feats_from_cfg(cfg)
        self.max_late_events = max_late_events

        # Node bookkeeping, kept for the whole run: an event may reference a node
        # first seen hours ago.
        self.key_to_index: Dict[str, str] = {}
        self.index_to_key: Dict[str, str] = {}
        self.indexid2msg: Dict[str, List[str]] = {}
        self.index_to_attrs: Dict[str, dict] = {}
        self.next_index = 0

        # Current window.
        self._buffer: List[Tuple[str, str, str, int, str]] = []
        self._window_start_ns: Optional[int] = None
        self._window_started_at: Optional[float] = None

        self.num_events = 0
        self.num_late_events = 0
        self.num_windows = 0

    def add_node(self, node: StreamNode) -> str:
        """Registers a node (or refreshes a known one) and returns its index id.

        Args:
            node: The node to register.

        Returns:
            str: The node's index id, as used in the graphs.
        """
        index_id = self.key_to_index.get(node.key)
        if index_id is None:
            index_id = str(self.next_index)
            self.next_index += 1
            self.key_to_index[node.key] = index_id
            self.index_to_key[index_id] = node.key

        attrs = {"type": node.node_type, **node.attrs}
        label = build_node_label(
            attrs, node.node_type, self.node_label_features, self.use_hashed_label
        )
        self.indexid2msg[index_id] = [node.node_type, label]
        self.index_to_attrs[index_id] = attrs
        return index_id

    def add_event(self, event: StreamEvent) -> Optional[TimeWindow]:
        """Adds an event, closing and returning the current window if it ends it.

        Args:
            event: The event to add.

        Returns:
            TimeWindow: The window this event just closed, or None.
        """
        src = self.key_to_index.get(event.src_key)
        dst = self.key_to_index.get(event.dst_key)
        if src is None or dst is None:
            # The adapter drops events with unknown endpoints, so this only happens
            # if a caller feeds events without their nodes.
            return None

        closed = None
        if self._window_start_ns is None:
            self._start_window(event.timestamp)
        elif event.timestamp >= self._window_start_ns + self.window_size_ns:
            closed = self._close_window()
            self._start_window(event.timestamp)
        elif event.timestamp < self._window_start_ns:
            # Slightly out-of-order events are normal (SPADE reorders audit records
            # within a bounded window); they simply join the window in progress.
            self.num_late_events += 1
            if self.num_late_events <= self.max_late_events:
                log(f"Warning: event {event.key} is older than its time window, keeping it anyway.")

        self._buffer.append((src, dst, event.operation, event.timestamp, event.key))
        self.num_events += 1
        return closed

    def flush(self) -> Optional[TimeWindow]:
        """Closes the pending window, whatever its age.

        Returns:
            TimeWindow: The closed window, or None if nothing is pending.
        """
        if not self._buffer:
            return None
        return self._close_window()

    def flush_if_idle(self) -> Optional[TimeWindow]:
        """Closes the pending window if it has been open longer than `flush_timeout`.

        Returns:
            TimeWindow: The closed window, or None.
        """
        if not self.flush_timeout or not self._buffer or self._window_started_at is None:
            return None
        if (time.time() - self._window_started_at) < self.flush_timeout:
            return None
        log(f"Flushing time window after {self.flush_timeout:.0f}s without a window boundary.")
        return self._close_window()

    def node_state(self) -> Dict[str, Tuple[str, dict]]:
        """Returns `{node key: (node type, raw attributes)}` for every known node.

        Handed to `pidsmaker.streaming.state` so a later run can resolve edges whose
        vertices were published before it started.
        """
        state = {}
        for key, index_id in self.key_to_index.items():
            attrs = self.index_to_attrs.get(index_id)
            if attrs is None:
                continue
            node_type = self.indexid2msg[index_id][0]
            state[key] = (node_type, {k: v for k, v in attrs.items() if k != "type"})
        return state

    @property
    def num_pending_events(self) -> int:
        """Events buffered in the window currently being built."""
        return len(self._buffer)

    def _start_window(self, timestamp: int):
        self._window_start_ns = timestamp
        self._window_started_at = time.time()

    def _close_window(self) -> Optional[TimeWindow]:
        events = sorted(self._buffer, key=lambda e: e[3])
        self._buffer = []
        self._window_start_ns = None
        self._window_started_at = None

        if not events:
            return None

        graph = self._build_graph(events)
        self.num_windows += 1
        return TimeWindow(
            graph=graph,
            start_ns=events[0][3],
            end_ns=events[-1][3],
            num_events=len(events),
        )

    def _build_graph(self, events) -> nx.MultiDiGraph:
        """Builds the window's graph, applying the same edge fusion as offline."""
        node_info = {}
        for src, dst, _, _, _ in events:
            for index_id in (src, dst):
                if index_id not in node_info:
                    node_type, label = self.indexid2msg[index_id]
                    node_info[index_id] = {"label": label, "node_type": node_type}

        if self.fuse_edge:
            edge_info = {}
            for src, dst, operation, timestamp, key in events:
                edge_info.setdefault((src, dst), []).append((timestamp, operation, key))
            edge_list = fuse_edges(edge_info)
        else:
            edge_list = [
                {"src": src, "dst": dst, "time": timestamp, "label": operation, "event_uuid": key}
                for src, dst, operation, timestamp, key in events
            ]

        graph = nx.MultiDiGraph()
        for index_id, info in node_info.items():
            graph.add_node(index_id, node_type=info["node_type"], label=info["label"])
        for edge in edge_list:
            graph.add_edge(
                edge["src"],
                edge["dst"],
                event_uuid=edge["event_uuid"],
                time=edge["time"],
                label=edge["label"],
                y=0,
            )
        return graph
