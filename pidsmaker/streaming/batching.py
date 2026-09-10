"""Turns a live time window into the exact batches the trained model expects.

The offline `batching` task walks the whole dataset three times: it groups events
globally, cuts them into TGN mini-batches while maintaining a last-neighbor
graph, reindexes node ids, then collates. The batching state that spans the whole
dataset offline - the neighbor loader, the node caches, the growing set of past
events - is precisely the state a detector must keep alive between windows.

`StreamingBatcher` is that state. It applies the same steps in the same order as
`load_all_datasets()`, one window at a time, reusing the same functions so a live
batch is indistinguishable from a pre-computed one.
"""

from typing import List, Optional

import torch

from pidsmaker.tasks.feat_inference import build_temporal_data
from pidsmaker.utils.data_utils import (
    CollatableTemporalData,
    GraphReindexer,
    TGNGraphBuilder,
    batch_temporal_data,
    custom_temporal_data_loader,
    extract_msg_from_data,
)
from pidsmaker.utils.dataset_utils import get_node_map, get_rel2id
from pidsmaker.utils.utils import gen_relation_onehot, log


class GrowingEventBuffer:
    """The events seen so far, indexed by the global event ids the TGN loader uses.

    Offline this is a single tensor over a dataset of known size; a stream has no
    known size, so the buffer grows geometrically. Events are appended in exactly
    the order they are inserted into the neighbor loader, which is what makes the
    loader's `e_id` a valid index into it.

    Args:
        max_events: Capacity ceiling. Reaching it resets the temporal context
            (see `_reset_hook`) rather than growing without bound, so a detector
            running for days keeps a flat memory profile.
    """

    FIELDS = ("msg", "t", "edge_type", "src", "dst")

    def __init__(self, max_events: int = 2_000_000):
        self.max_events = max_events
        self.size = 0
        self._buffers = {}
        self._reset_hook = None

    def _grow(self, field: str, sample: torch.Tensor, needed: int):
        buffer = self._buffers.get(field)
        if buffer is not None and buffer.shape[0] >= needed:
            return buffer

        capacity = max(1024, needed * 2)
        shape = (capacity,) + tuple(sample.shape[1:])
        grown = torch.zeros(shape, dtype=sample.dtype)
        if buffer is not None:
            grown[: self.size] = buffer[: self.size]
        self._buffers[field] = grown
        return grown

    def append(self, data: CollatableTemporalData) -> None:
        """Appends every event of a batch."""
        num_events = data.src.numel()
        if num_events == 0:
            return

        if self.size + num_events > self.max_events:
            log(
                f"Event buffer reached {self.max_events} events: resetting the temporal context. "
                "Raise `--stream_max_events` to keep a longer history."
            )
            self.reset()
            if self._reset_hook is not None:
                self._reset_hook()

        for field in self.FIELDS:
            values = getattr(data, field).detach().cpu()
            buffer = self._grow(field, values, self.size + num_events)
            buffer[self.size : self.size + num_events] = values
        self.size += num_events

    def reset(self) -> None:
        """Forgets every event."""
        self.size = 0

    def __getattr__(self, name):
        # `full_data.msg[e_id]` and friends: only the filled part is ever valid.
        if name in GrowingEventBuffer.FIELDS:
            buffer = self.__dict__["_buffers"].get(name)
            if buffer is None:
                raise AttributeError(f"No {name} recorded yet.")
            return buffer[: self.__dict__["size"]]
        raise AttributeError(name)


class StreamingBatcher:
    """Builds the model-ready batches of one time window, keeping state across windows.

    Args:
        cfg: Full pipeline config; every `batching.*` setting is honoured.
        device: Device the batching state lives on.
        max_nodes: Upper bound on the number of distinct nodes the stream will
            produce, used to size the reindexing and neighbor tables.
        max_events: Capacity of the event buffer backing the TGN neighbor loader.
    """

    def __init__(self, cfg, device, max_nodes: int = 2_000_000, max_events: int = 2_000_000):
        self.cfg = cfg
        self.device = device
        self.max_nodes = max_nodes

        self.etype2oh = gen_relation_onehot(rel2id=get_rel2id(cfg))
        self.ntype2oh = gen_relation_onehot(rel2id=get_node_map())

        self.graph_reindexer = GraphReindexer(
            device=device,
            num_nodes=max_nodes,
            fix_buggy_graph_reindexer=cfg.batching.fix_buggy_graph_reindexer,
        )

        self.intra_methods = [
            m.strip() for m in cfg.batching.intra_graph_batching.used_methods.split(",")
        ]
        self.use_tgn = "tgn_last_neighbor" in self.intra_methods
        self.use_unique_edge_types = "unique_edge_types" in cfg.batching.global_batching.used_method

        self.event_buffer = GrowingEventBuffer(max_events=max_events) if self.use_tgn else None
        self.tgn_builder = None

    def _init_tgn(self, sample: CollatableTemporalData):
        """Creates the TGN state, once the first window has revealed the feature dims."""
        self.tgn_builder = TGNGraphBuilder(
            full_data=self.event_buffer,
            graph_reindexer=self.graph_reindexer,
            device=self.device,
            max_node=self.max_nodes,
            tgn_loader_cfg=self.cfg.batching.intra_graph_batching.tgn_last_neighbor,
            node_feat_dim=sample.x_src.shape[1],
            node_type_dim=sample.node_type_src.shape[1],
        )
        # An event-buffer overflow invalidates every id the loader holds, so the
        # loader has to start over with it.
        self.event_buffer._reset_hook = self._reset_tgn_state

    def _reset_tgn_state(self):
        self.tgn_builder.neighbor_loader.reset_state()

    def build(self, graph, indexid2vec: Optional[dict]) -> List[CollatableTemporalData]:
        """Builds the batches of one time window.

        Args:
            graph: The window's `networkx.MultiDiGraph`.
            indexid2vec: Node embeddings from the online featurizer, or None.

        Returns:
            list: `CollatableTemporalData` batches, ready to be given to the model.
        """
        data = build_temporal_data(graph, indexid2vec, self.etype2oh, self.ntype2oh)
        extract_msg_from_data([data], self.cfg)

        batches = self._global_batching(data)
        batches = self._intra_graph_batching(batches)

        if not self.use_unique_edge_types:
            for batch in batches:
                batch.to(self.device)
                self.graph_reindexer.reindex_graph(
                    batch, use_tgn=self.use_tgn, x_is_tuple=self.cfg.training.encoder.x_is_tuple
                )

        return self._inter_graph_batching(batches)

    def _global_batching(self, data: CollatableTemporalData) -> List[CollatableTemporalData]:
        global_cfg = self.cfg.batching.global_batching
        mode = global_cfg.used_method
        if mode == "none":
            return [data]

        # A detector always runs inference, so the inference batch size wins when set.
        batch_size = global_cfg.global_batching_batch_size_inference or (
            global_cfg.global_batching_batch_size
        )
        if batch_size in (None, 0) and mode != "unique_edge_types":
            return [data]

        return batch_temporal_data(data, batch_size, mode, self.cfg, self.device)

    def _intra_graph_batching(self, batches):
        for method in self.intra_methods:
            if method == "none":
                continue

            if method == "edges":
                batch_size = self.cfg.batching.intra_graph_batching.edges.intra_graph_batch_size
                batches = [
                    small
                    for batch in batches
                    for small in custom_temporal_data_loader(batch, batch_size=batch_size)
                ]

            elif method == "tgn_last_neighbor":
                if self.tgn_builder is None:
                    if not batches:
                        return batches
                    self._init_tgn(batches[0])
                processed = []
                for batch in batches:
                    # The buffer must hold the batch before the loader is told about
                    # it, so that the ids it returns index the same events.
                    self.event_buffer.append(batch)
                    processed.append(self.tgn_builder.process(batch))
                batches = processed

            elif method == "neighbor_sampling":
                raise NotImplementedError(
                    "`neighbor_sampling` intra-graph batching is not implemented offline either."
                )
            else:
                raise ValueError(f"Invalid sampling method {method}")

        return batches

    def _inter_graph_batching(self, batches):
        method = self.cfg.batching.inter_graph_batching.used_method
        if method == "none":
            return batches
        if method != "graph_batching":
            raise ValueError(f"Invalid inter-graph batching method {method}")

        from torch_geometric.data.collate import collate

        batch_size = self.cfg.batching.inter_graph_batching.inter_graph_batch_size
        return [
            collate(CollatableTemporalData, data_list=batches[i : i + batch_size])[0]
            for i in range(0, len(batches), batch_size)
        ]
