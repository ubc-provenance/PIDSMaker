"""Data utilities for graph batching and temporal data handling.

Provides custom PyTorch Geometric data structures and utilities for:
- Temporal graph data collation and batching
- Graph reindexing for mini-batch training
- TGN memory and neighbor sampling integration
- Multi-dataset handling and preprocessing
"""

import copy
import math
import os
import pickle
from collections import defaultdict
from functools import cached_property

import torch
import torch.nn.functional as F
from torch_geometric.data import Data, TemporalData
from torch_geometric.data.collate import collate
from torch_geometric.data.data import size_repr
from torch_geometric.data.temporal import prepare_idx
from torch_geometric.loader import TemporalDataLoader
from torch_scatter import scatter

from pidsmaker.config import update_cfg_for_multi_dataset
from pidsmaker.debug_tests import debug_test_batching
from pidsmaker.tgn import LastNeighborLoader
from pidsmaker.utils.dataset_utils import (
    get_node_map,
    get_num_edge_type,
    get_rel2id,
    get_possible_events,
)
from pidsmaker.utils.utils import get_multi_datasets, log_dataset_stats, log_tqdm
from pidsmaker.hetero import compute_hetero_features


class CollatableTemporalData(TemporalData):
    """
    We use this class instead of TemporalData in order to easily concatenate data
    objects together without any batching behavior.
    Normal TemporalData doesn't support edge_index so we define it here.
    """

    def __init__(
        self,
        src=None,
        dst=None,
        t=None,
        msg=None,
        **kwargs,
    ):
        super().__init__(src=src, dst=dst, t=t, msg=msg, **kwargs)
        self.tgn_mode = False

    def __inc__(self, key: str, value, *args, **kwargs):
        if key == "original_edge_index":  # used to retrieve original node IDs during evaluation
            return 0
        if "edge_index" in key or key in ["src", "dst", "reindexed_original_n_id_tgn"]:
            if self.tgn_mode:
                return torch.unique(self.n_id_tgn).numel()
            return self.num_nodes
        return 0

    def __cat_dim__(self, key: str, value, *args, **kwargs):
        return 1 if "edge_index" in key else 0

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        info = ", ".join([size_repr(k, v) for k, v in self._store.items()])
        info += ", " + size_repr("edge_index", self.edge_index)
        return f"{cls}({info})"

    def index_select(self, idx):
        """ "Indexing to handle (2, E) index attributes"""
        idx = prepare_idx(idx)
        data = copy.copy(self)
        for key, value in data._store.items():
            if not isinstance(value, torch.Tensor):
                continue
            if value.size(0) == self.num_events:
                data[key] = value[idx]
            elif value.ndim == 2 and value.size(1) == self.num_events and "index" in key:
                data[key] = value[:, idx]
        return data

    @property
    def num_nodes(self):
        if "original_n_id" in self:
            return self.original_n_id.numel()
        return self.edge_index.unique().numel()

    @cached_property
    def node_type_argmax(self):
        return self.node_type.max(dim=1).indices

    @cached_property
    def node_type_src_argmax(self):
        return self.node_type_src.max(dim=1).indices

    @cached_property
    def edge_type_argmax(self):
        return self.edge_type.max(dim=1).indices

    @cached_property
    def node_type_dst_argmax(self):
        return self.node_type_dst.max(dim=1).indices


def batch_to_device(data, device):
    """Returns a copy of `data` on `device`, leaving `data` untouched.

    `Data.to()` is in-place: moving a CPU graph to the GPU and back to the CPU costs
    two transfers per batch. Moving a shallow copy only costs the host-to-device one;
    the GPU copy is released by the caching allocator once no longer referenced.
    Tensors already on `device` are not copied.
    """
    return copy.copy(data).to(device, non_blocking=True)


def load_all_datasets(cfg, device, only_keep=None):
    multi_dataset = cfg.batching.multi_dataset_training
    train_data = load_data_set(cfg, split="train", multi_dataset=multi_dataset)
    val_data = load_data_set(cfg, split="val", multi_dataset=multi_dataset)
    test_data = load_data_set(cfg, split="test", multi_dataset=False)

    if only_keep is not None:
        train_data = train_data[:only_keep]
        val_data = val_data[:only_keep]
        test_data = test_data[:only_keep]

    use_tgn_intra = (
        "tgn_last_neighbor" in cfg.batching.intra_graph_batching.used_methods
    )

    max_node = max(
        max(data.src.max().item(), data.dst.max().item())
        for dataset_group in [train_data, val_data, test_data]
        for dataset in dataset_group
        for data in dataset
    ) + 1
    print(f"Max node in {cfg.dataset.name}: {max_node}")

    graph_reindexer = GraphReindexer(
        device=device,
        num_nodes=max_node,
        fix_buggy_graph_reindexer=cfg.batching.fix_buggy_graph_reindexer,
    )

    # Global batching (unique edge type batches, fixed-size edge length)
    datasets = run_global_batching(train_data, val_data, test_data, cfg, device)

    # full_data must be built AFTER global batching so its row count matches the
    # edges actually fed into the TGN neighbor loader. The "minutes" batch_mode
    # duplicates edges across overlapping sliding windows; building full_data
    # from the pre-batched data would leave it shorter than the loader's e_id range.
    full_data = None
    if use_tgn_intra:
        # Only the attributes gathered by `compute_tgn_graphs` are concatenated
        edge_features = _edge_features_list(cfg)
        keys = ["src", "dst", "edge_type"]
        if "time_encoding" in edge_features:
            keys.append("t")
        if "msg" in edge_features:
            keys.append("msg")
        full_data = get_full_data(datasets, keys=keys)

    # Intra graph batching (TGN 1024 batches, last neighbor loader)
    datasets = run_intra_graph_batching(datasets, full_data, device, max_node, cfg, graph_reindexer)
    del full_data

    # Reindexing stuff (create node-level attributes)
    datasets = run_reindexing_preprocessing(datasets, graph_reindexer, device, cfg)

    # Inter graph batching (actual mini-batching of very small graphs)
    datasets = run_inter_graph_batching(datasets, cfg)

    train_data, val_data, test_data = datasets
    return train_data, val_data, test_data, max_node


# Compact storage of the edge messages written by feat_inference (see `expand_compact_msg`)
COMPACT_MSG_KEYS = (
    "msg_node_feats",  # (N, F) features of each node of the graph: [node_type | node_emb]
    "msg_src_row",  # (E,) row of each edge's source in msg_node_feats
    "msg_dst_row",  # (E,) row of each edge's destination in msg_node_feats
    "msg_edge_type_table",  # (K + 1, T) one-hot of each edge type (last row: no type)
    "msg_edge_type_row",  # (E,) row of each edge's type in msg_edge_type_table
)


def expand_compact_msg(data):
    """Rebuilds the (E, D) edge messages `msg` = [src_feats | edge_type | dst_feats] from
    their compact storage, in place. Data saved with a materialized `msg` is left as is."""
    if COMPACT_MSG_KEYS[0] not in data:
        return data
    node_feats, src_row, dst_row, etype_table, etype_row = (data[k] for k in COMPACT_MSG_KEYS)
    data.msg = torch.cat(
        [node_feats[src_row.long()], etype_table[etype_row.long()], node_feats[dst_row.long()]],
        dim=1,
    )
    for k in COMPACT_MSG_KEYS:
        del data[k]
    return data


def load_data_list(path, split, cfg):
    data_list = []
    for f in sorted(os.listdir(os.path.join(path, split))):
        filepath = os.path.join(path, split, f)
        data = expand_compact_msg(torch.load(filepath).to("cpu"))
        data_list.append(data)

    data_list = extract_msg_from_data(data_list, cfg)
    return data_list


def load_data_set(cfg, split: str, multi_dataset=False) -> list[CollatableTemporalData]:
    """
    Returns a list of time window graphs for a given `split` (train/val/test set).
    """
    if multi_dataset:
        multi_datasets = get_multi_datasets(cfg)
        all_data_lists = []
        for dataset in multi_datasets:
            updated_cfg, _ = update_cfg_for_multi_dataset(cfg, dataset)
            path = updated_cfg.feat_inference._edge_embeds_dir
            all_data_lists.append(load_data_list(path, split, cfg))
        return all_data_lists

    else:
        path = cfg.feat_inference._edge_embeds_dir
        return [load_data_list(path, split, cfg)]


def extract_msg_from_data(
    data_set: list[CollatableTemporalData], cfg
) -> list[CollatableTemporalData]:
    """
    Initializes the attributes of a `Data` object based on the `msg`
    computed in previous tasks.
    """
    emb_dim = cfg.featurization.emb_dim
    method = cfg.featurization.used_method.strip()
    only_type = method == "only_type"
    only_ones = method == "only_ones"
    if only_type or only_ones or emb_dim is None:
        emb_dim = 0
    node_type_dim = cfg.dataset.num_node_types
    # msg holds the plain (possibly OpTC-consistent) edge types; triplet types are derived later
    edge_type_dim = (
        cfg.dataset.num_edge_types
        if "edge_type_triplet" in cfg.batching.edge_features
        else get_num_edge_type(cfg)
    )

    if method == "ocrapt_features":
        # in-histogram + out-histogram (L2-normalized together) + idle min/max/avg,
        # plus one extra dim each for the optional lifespan/cumulative-active-time
        oc = cfg.featurization.ocrapt_features
        emb_dim = 2 * edge_type_dim + 3 + int(oc.use_lifespan) + int(oc.use_cumulative_active_time)

    selected_node_feats = cfg.batching.node_features

    msg_len = data_set[0].msg.shape[1]
    expected_msg_len = (emb_dim * 2) + (node_type_dim * 2) + edge_type_dim
    if msg_len != expected_msg_len:
        raise ValueError(
            f"The msg has an invalid shape, found {msg_len} instead of {expected_msg_len} (Is your num_edge_types correct?)"
        )

    field_to_size = {
        "src_type": node_type_dim,
        "src_emb": emb_dim,
        "edge_type": edge_type_dim,
        "dst_type": node_type_dim,
        "dst_emb": emb_dim,
    }

    if "edges_distribution" in selected_node_feats:
        max_num_nodes = max([torch.cat([g.src, g.dst]).max().item() for g in data_set]) + 1
        x_distrib = torch.zeros(max_num_nodes, edge_type_dim * 2, dtype=torch.float)

    if only_type:
        selected_node_feats = ["node_type"]
    elif only_ones:
        selected_node_feats = ["only_ones"]
    else:
        selected_node_feats = list(
            map(lambda x: x.strip(), selected_node_feats.replace("-", ",").split(","))
        )

    edge_features = list(
        map(lambda x: x.strip(), cfg.batching.edge_features.split(","))
    )
    if "edge_type_triplet" in edge_features:
        possible_triplets = get_possible_triplets(cfg)
    else:
        possible_triplets = None

    for g in data_set:
        fields = {}
        idx = 0
        for field, size in field_to_size.items():
            fields[field] = g.msg[:, idx : idx + size]
            idx += size

        # Selects only the node features we want
        x_src, x_dst = [], []
        for feat in selected_node_feats:
            if feat == "node_emb":
                x_src.append(fields["src_emb"])
                x_dst.append(fields["dst_emb"])

            elif feat == "node_type":
                x_src.append(fields["src_type"])
                x_dst.append(fields["dst_type"])

            elif feat == "edges_distribution":  # as in ThreaTrace
                x_distrib.scatter_add_(
                    0, g.src.unsqueeze(1).expand(-1, edge_type_dim), fields["edge_type"]
                )
                x_distrib[:, edge_type_dim:].scatter_add_(
                    0, g.dst.unsqueeze(1).expand(-1, edge_type_dim), fields["edge_type"]
                )

                # In ThreaTrace they don't standardize, here we do standardize by max value in TW
                x_distrib = x_distrib / (x_distrib.max() + 1e-12)

                x_src.append(x_distrib[g.src])
                x_dst.append(x_distrib[g.dst])

                x_distrib.fill_(0)

            elif feat == "only_ones":
                x_src.append(fields["src_type"].clone().fill_(1))
                x_dst.append(fields["dst_type"].clone().fill_(1))

            else:
                raise ValueError(f"Node feature {feat} is invalid.")

        x_src = torch.cat(x_src, dim=-1)
        x_dst = torch.cat(x_dst, dim=-1)

        # If we want to predict the edge type, we remove the edge type from the message
        if "predict_edge_type" in cfg.training.decoder.used_methods:
            msg = torch.cat([x_src, x_dst], dim=-1)
        else:
            msg = torch.cat([x_src, x_dst, fields["edge_type"]], dim=-1)

        num_edge_types = get_num_edge_type(cfg)
        edge_feats = build_edge_feats(fields, msg, edge_features, possible_triplets, num_edge_types)

        if "edge_type_triplet" in edge_features:
            edge_type = get_triplet_edge_types(
                fields["src_type"],
                fields["dst_type"],
                fields["edge_type"],
                possible_triplets,
                num_edge_types,
            )
        else:
            edge_type = fields["edge_type"]

        g.x_src = x_src
        g.x_dst = x_dst
        g.edge_feats = edge_feats
        # fields are views (slices) of g.msg. Clone them so that deleting
        # g.msg actually frees the underlying storage.
        g.edge_type = edge_type.clone()
        g.node_type_src = fields["src_type"].clone()
        g.node_type_dst = fields["dst_type"].clone()

        if _needs_raw_msg(cfg):
            g.msg = msg
        else:
            del g.msg

        # NOTE: do not add edge_index as it is already within `CollatableTemporalData`
        # g.edge_index = ...

    return data_set

def _edge_features_list(cfg):
    return [x.strip() for x in cfg.batching.edge_features.split(",")]


def _needs_raw_msg(cfg):
    """Whether the edge-level `msg` is consumed downstream.

    Only the TGN encoder reads it: as raw message of the TGN memory, or as edge
    feature (`msg_tgn`). It also sizes the TGN modules when `msg` is an edge feature.
    """
    use_tgn = "tgn" in cfg.training.encoder.used_methods
    if not use_tgn:
        return False
    return cfg.training.encoder.tgn.use_memory or "msg" in _edge_features_list(cfg)


def get_possible_triplets(cfg):
    entity_map = get_node_map(from_zero=True)
    event_map = get_rel2id(cfg, from_zero=True)
    possible_events = get_possible_events(cfg)

    possible_triplets = [
        [entity_map[src_type], entity_map[dst_type], event_map[event]]
        for (src_type, dst_type), events in possible_events.items()
        for event in events
    ]
    return torch.tensor(possible_triplets, dtype=torch.long)


def get_triplet_edge_types(src_type, dst_type, edge_type, possible_triplets, num_edge_types):
    triplets = torch.stack(
        (src_type.max(dim=1).indices, dst_type.max(dim=1).indices, edge_type.max(dim=1).indices),
        dim=1,
    )
    matches = (triplets.unsqueeze(1) == possible_triplets.unsqueeze(0)).all(dim=2)
    return F.one_hot(matches.long().argmax(dim=1), num_classes=num_edge_types).to(torch.float)


def build_edge_feats(fields, msg, edge_features, possible_triplets, num_edge_types):
    edge_feats = []
    if "edge_type" in edge_features:
        edge_feats.append(fields["edge_type"])
    if "edge_type_triplet" in edge_features:
        triplets = get_triplet_edge_types(
            fields["src_type"],
            fields["dst_type"],
            fields["edge_type"],
            possible_triplets,
            num_edge_types,
        )
        edge_feats.append(triplets)
    if "msg" in edge_features:
        edge_feats.append(msg)
    if len(edge_feats) == 0:
        return None
    if len(edge_feats) == 1 and edge_feats[0] is msg:
        # `msg` is a standalone tensor: share it rather than storing a copy. Other single
        # features are views of the raw msg and are copied so that it can be freed.
        return msg
    return torch.cat(edge_feats, dim=-1)


def get_full_data(datasets, keys=("t", "edge_type", "src", "dst", "msg")):
    """Concatenates the given edge-level attributes of all graphs (missing ones are skipped)."""
    sample = datasets[0][0][0]
    keys = [k for k in keys if hasattr(sample, k)]
    all_data = {k: [] for k in keys}
    for dataset_group in datasets:
        for dataset in dataset_group:
            for data in dataset:
                for k in all_data:
                    all_data[k].append(getattr(data, k))

    full_data = Data(**{k: torch.cat(v) for k, v in all_data.items()})
    return full_data


def custom_temporal_data_loader(data: CollatableTemporalData, batch_size: int, *args, **kwargs):
    """
    A simple `TemporalDataLoader` which also update the edge_index with the
    sampled edges of size `batch_size`. By default, only attributes of shape (E, d)
    are updated, `edge_index` is thus not updated automatically.
    """
    loader = TemporalDataLoader(data, batch_size=batch_size, *args, **kwargs)
    for batch in loader:
        yield batch


def temporal_data_to_data(data: CollatableTemporalData) -> Data:
    """
    NeighborLoader requires a `Data` object.
    We need to convert `CollatableTemporalData` to `Data` before using it.
    """
    data = Data(num_nodes=data.x_src.shape[0], **{k: v for k, v in data._store.items()})
    del data.num_nodes
    return data


def collate_temporal_data(data_list: list[CollatableTemporalData]) -> CollatableTemporalData:
    """
    Concatenates attributes from data ojects into a single data object.
    Do not use with `Data` directly because it will use batching when collating.
    """
    assert all([not isinstance(data, Data) for data in data_list]), (
        "Concatenating Data objects result in batching."
    )

    data = collate(CollatableTemporalData, data_list, increment=False)[0]
    del data.ptr
    del data.batch
    # `tgn_mode` is a flag, not a per-event attribute: collate turns it into a tensor
    data.tgn_mode = bool(data_list[0].tgn_mode) if len(data_list) > 0 else False

    return data


def batch_temporal_data(
    data: CollatableTemporalData, batch_size: int, batch_mode: str, cfg, device
) -> list[CollatableTemporalData]:
    if batch_mode == "edges":
        num_batches = math.ceil(
            len(data.src) / batch_size
        )  # NOTE: the last batch won't have the same number of edges as the batch

        data_list = [
            data[int(i * batch_size) : int((i + 1) * batch_size)] for i in range(num_batches)
        ]
        return data_list

    elif batch_mode == "minutes":
        window_length_ns = int(cfg.construction.time_window_size * 60_000_000_000)
        sliding_ns = int(batch_size * 60_000_000_000)  # min to ns

        t0 = data.t.min()
        t0_aligned = (t0 // sliding_ns) * sliding_ns

        # Window indices [i0, i1] of each event: an event belongs to all sliding
        # windows of length `window_length_ns` that contain its timestamp
        t = data.t
        i0 = (((t - window_length_ns - t0_aligned) + sliding_ns - 1) // sliding_ns).clamp(min=0)
        i1 = (t - t0_aligned) // sliding_ns
        counts = (i1 - i0 + 1).clamp(min=0)

        # One (window, event) pair per membership, grouped by window with events in order
        event_idx = torch.arange(t.numel(), device=t.device).repeat_interleave(counts)
        offsets = torch.cumsum(counts, 0) - counts
        window_idx = i0.repeat_interleave(counts) + (
            torch.arange(event_idx.numel(), device=t.device) - offsets.repeat_interleave(counts)
        )
        window_idx, perm = torch.sort(window_idx, stable=True)
        event_idx = event_idx[perm]

        windows = []
        _, sizes = torch.unique_consecutive(window_idx, return_counts=True)
        for idx in event_idx.split(sizes.tolist()):
            windows.append(data.index_select(idx))
        return windows

    elif batch_mode == "unique_edge_types":
        partitions = []
        seen_edges = defaultdict(set)
        data.to(device)

        partitions = []
        src_list = data.src.tolist()
        dst_list = data.dst.tolist()
        type_list = data.edge_type.max(dim=1).indices.tolist()
        start, end = 0, 0

        for i, (src, dst, edge_type) in log_tqdm(
            enumerate(zip(src_list, dst_list, type_list)),
            desc="Generating unique edge type batches",
        ):
            if (src, dst) in seen_edges:
                # Conflict: (src, dst) already exists with a different edge_type
                partitions.append(data[start:end])
                start = end
                end += 1
                seen_edges = defaultdict(set)
            else:
                end += 1

            seen_edges[(src, dst)].add(edge_type)

        # Add the last partition if not empty
        if end > start:
            partitions.append(data[start:end])

        data.to("cpu")
        return partitions

    raise ValueError(f"Invalid or missing batch mode {batch_mode}")


def run_global_batching(train_data, val_data, test_data, cfg, device):
    # Concatenates all data into a single data so that iterating over batches
    # of edges is more consistent with TGN
    global_batching_cfg = cfg.batching.global_batching
    batch_mode = global_batching_cfg.used_method
    bs = global_batching_cfg.global_batching_batch_size
    bs_inference = global_batching_cfg.global_batching_batch_size_inference
    if batch_mode != "none":
        if (bs not in [None, 0]) or batch_mode == "unique_edge_types":
            train_data = [
                batch_temporal_data(collate_temporal_data(graphs), bs, batch_mode, cfg, device)
                for graphs in train_data
            ]
            val_data = [
                batch_temporal_data(collate_temporal_data(graphs), bs, batch_mode, cfg, device)
                for graphs in val_data
            ]
            test_data = [
                batch_temporal_data(collate_temporal_data(graphs), bs, batch_mode, cfg, device)
                for graphs in test_data
            ]

        elif bs_inference not in [None, 0]:
            test_data = [
                batch_temporal_data(
                    collate_temporal_data(graphs), bs_inference, batch_mode, cfg, device
                )
                for graphs in test_data
            ]

    return train_data, val_data, test_data


def run_reindexing_preprocessing(datasets, graph_reindexer, device, cfg):
    use_unique_edge_types = "unique_edge_types" in cfg.batching.global_batching.used_method
    if not use_unique_edge_types:
        log_dataset_stats(datasets)
        # By default we only have x_src and x_dst of shape (E, d), here we create x of shape (N, d)
        use_tgn = "tgn" in cfg.training.encoder.used_methods
        # x_src/x_dst (E, d) are only needed after reindexing by specific encoders/decoders.
        # When not needed, reindex_graph deletes them after folding into x (N, d).
        encoder_methods = cfg.training.encoder.used_methods
        decoder_methods = cfg.training.decoder.used_methods
        keep_edge_level_features = (
            "tgn" in encoder_methods
            or "predict_edge_supervised" in decoder_methods
        )
        reindex_graphs(
            datasets,
            graph_reindexer,
            device,
            use_tgn,
            keep_edge_level_features=keep_edge_level_features,
            x_is_tuple=cfg.training.encoder.x_is_tuple,
        )

    use_tgn_loader = (
        "tgn_last_neighbor" in cfg.batching.intra_graph_batching.used_methods
    )
    if cfg._is_hetero and not use_tgn_loader:  # If TGN, we compute hetero feats in the encoder
        compute_hetero_graphs(datasets, device, cfg)

    return datasets


def run_intra_graph_batching(datasets, full_data, device, max_node, cfg, graph_reindexer):
    def standard_intra_batching(dataset, method):
        result = []
        for data_list in dataset:
            result.append([])
            for batch in log_tqdm(data_list, desc="Creating TGN batches"):
                # Use temporal batch loader used in TGN
                if method == "edges":
                    batch_size = cfg.batching.intra_graph_batching.edges.intra_graph_batch_size
                    batch_loader = custom_temporal_data_loader(batch, batch_size=batch_size)
                elif method == "neighbor_sampling":
                    raise NotImplementedError
                else:
                    raise ValueError(f"Invalid sampling method {method}")

                for small_batch in batch_loader:
                    result[-1].append(small_batch)
        return result

    methods = map(
        lambda x: x.strip(),
        cfg.batching.intra_graph_batching.used_methods.split(","),
    )
    for method in methods:
        if method == "none":
            continue

        elif method in ["edges", "neighbor_sampling"]:
            datasets = [standard_intra_batching(dataset, method) for dataset in datasets]

        elif method == "tgn_last_neighbor":
            tgn_loader_cfg = cfg.batching.intra_graph_batching.tgn_last_neighbor
            sample = datasets[0][0][0]
            datasets = compute_tgn_graphs(
                datasets=datasets,
                full_data=full_data,
                graph_reindexer=graph_reindexer,
                device=device,
                max_node=max_node,
                tgn_loader_cfg=tgn_loader_cfg,
                node_feat_dim=sample.x_src.shape[1],
                node_type_dim=sample.node_type_src.shape[1],
            )

        else:
            raise ValueError(f"Invalid sampling method {method}")

    return datasets


def compute_tgn_graphs(
    datasets,
    full_data,
    graph_reindexer,
    device,
    max_node,
    tgn_loader_cfg,
    node_feat_dim,
    node_type_dim,
):
    tgn_neighbor_n_hop = tgn_loader_cfg.tgn_neighbor_n_hop
    fix_tgn_neighbor_loader = tgn_loader_cfg.fix_tgn_neighbor_loader
    fix_buggy_orthrus_TGN = tgn_loader_cfg.fix_buggy_orthrus_TGN
    insert_neighbors_before = tgn_loader_cfg.insert_neighbors_before
    neighbor_size = tgn_loader_cfg.tgn_neighbor_size
    directed = tgn_loader_cfg.directed

    neighbor_loader = LastNeighborLoader(
        max_node, size=neighbor_size, directed=directed, device=device
    )

    node_feat_cache = torch.zeros((max_node, node_feat_dim), device=device)
    node_type_cache = torch.zeros((max_node, node_type_dim), device=device)
    assoc = torch.empty(max_node, dtype=torch.long, device=device)

    for dataset in datasets:
        for data_list in dataset:
            for batch in log_tqdm(data_list, desc="Computing TGN last neighbor graphs"):
                # Only the inputs are moved to the device and only the outputs are moved
                # back: transferring the whole batch both ways dominated this step.
                src = batch.src.to(device)
                dst = batch.dst.to(device)
                batch_edge_index = torch.stack([src, dst], dim=0)

                if insert_neighbors_before:
                    neighbor_loader.insert(src, dst)

                original_n_id = batch_edge_index.unique()
                n_id = original_n_id
                for _ in range(tgn_neighbor_n_hop):
                    n_id, edge_index, e_id = neighbor_loader(n_id)

                if fix_tgn_neighbor_loader:
                    # NOTE: TGN's loader wrongly index edges (less than 1% in the returned e_id and edge_index)
                    # https://github.com/pyg-team/pytorch_geometric/issues/10100
                    # Should be replaced by an actual fix when available
                    real_src = full_data.src[e_id.cpu()].to(device)
                    real_dst = full_data.dst[e_id.cpu()].to(device)

                    loader_src = n_id[edge_index[0]]
                    loader_dst = n_id[edge_index[1]]

                    match_dir1 = real_src.eq(loader_src) & real_dst.eq(loader_dst)
                    match_dir2 = real_src.eq(loader_dst) & real_dst.eq(loader_src)

                    valid_edges = match_dir1 | match_dir2
                    edge_index = edge_index[:, valid_edges]
                    e_id = e_id[valid_edges]

                num_nodes = n_id.size(
                    0
                )  # Important, this one is used as __inc__ when batching graphs
                assoc[n_id] = torch.arange(num_nodes, device=device)
                src_dst = torch.cat([src, dst])
                node_type_cache[src_dst] = torch.cat(
                    [batch.node_type_src, batch.node_type_dst]
                ).to(device)

                x_src_dev = batch.x_src.to(device)
                x_dst_dev = batch.x_dst.to(device)
                if fix_buggy_orthrus_TGN:
                    node_feat_cache[src_dst] = torch.cat([x_src_dev, x_dst_dev])
                    x_src = torch.zeros((num_nodes, node_feat_dim), device=device)
                    x_dst = x_src.clone()
                    src_id, dst_id = edge_index[0].unique(), edge_index[1].unique()
                    x_src[src_id] = node_feat_cache[n_id[src_id]]
                    x_dst[dst_id] = node_feat_cache[n_id[dst_id]]
                    new_x = node_feat_cache[n_id]
                    batch.x_from_tgn = x_src.cpu()  # (N, d)
                    batch.x_to_tgn = x_dst.cpu()  # (N, d)
                    batch.x_tgn = new_x.cpu()  # (N, d)

                else:
                    (x_src, x_dst), *_ = graph_reindexer._reindex_graph(
                        batch_edge_index,
                        x_src_dev,
                        x_dst_dev,
                        max_num_node=num_nodes,
                        x_is_tuple=True,
                    )
                    batch.x_from_tgn = x_src.cpu()
                    batch.x_to_tgn = x_dst.cpu()
                    batch.x_tgn = batch.x_from_tgn  # same tensor, stored once
                del x_src_dev, x_dst_dev

                batch.tgn_mode = True
                batch.original_edge_index = batch_edge_index.cpu()
                batch.original_n_id = original_n_id.cpu()
                batch.reindexed_original_n_id_tgn = assoc[original_n_id].cpu()
                batch.n_id_tgn = n_id.cpu()
                batch.edge_index_tgn = edge_index.cpu()
                batch.reindexed_edge_index_tgn = assoc[batch_edge_index].cpu()
                e_id_cpu = e_id.cpu()
                # msg_tgn and t_tgn are only read by the TGN encoder with the
                # `msg` and `time_encoding` edge features (see `run_global_batching`)
                if "msg" in full_data:
                    batch.msg_tgn = full_data.msg[e_id_cpu]
                if "t" in full_data:
                    batch.t_tgn = full_data.t[e_id_cpu]
                batch.node_type_tgn = node_type_cache[n_id].cpu()
                batch.edge_type_tgn = full_data.edge_type[e_id_cpu]

                if not insert_neighbors_before:
                    neighbor_loader.insert(src, dst)

    return datasets


def run_inter_graph_batching(datasets, cfg):
    def inter_batching(dataset, method):
        if method == "none":
            return dataset

        elif method == "graph_batching":
            bs = cfg.batching.inter_graph_batching.inter_graph_batch_size
            result = []
            for data_list in dataset:
                result.append([])
                for i in log_tqdm(
                    range(0, len(data_list), bs),
                    total=math.ceil(len(data_list) / bs),
                    desc="Mini-batching",
                ):
                    batch = data_list[i : i + bs]
                    data = collate(CollatableTemporalData, data_list=batch)[0]

                    use_tgn = "tgn" in cfg.training.encoder.used_methods
                    if cfg._debug and use_tgn:
                        debug_test_batching(batch, data, cfg)
                    result[-1].append(data)
            return result

        raise ValueError(f"Invalid inter-graph batching method {method}")

    method = cfg.batching.inter_graph_batching.used_method
    datasets = [inter_batching(dataset, method) for dataset in datasets]
    return datasets


class _Cache:
    def __init__(self, shape, device):
        self._cache = torch.zeros(shape, device=device)

    @property
    def cache(self):
        return self._cache

    def detach(self):
        self._cache = self._cache.detach()

    def to(self, device):
        self._cache = self._cache.to(device)
        return self


class GraphReindexer:
    """
    Simply transforms an edge_index and its src/dst node features of shape (E, d)
    to a reindexed edge_index with node IDs starting from 0 and src/dst node features of shape
    (max_num_node + 1, d).
    This reindexing is essential for the graph to be computed by a standard GNN model with PyG.
    """

    def __init__(self, device, num_nodes=None, fix_buggy_graph_reindexer=True):
        self.num_nodes = num_nodes
        self.device = device
        self.fix_buggy_graph_reindexer = fix_buggy_graph_reindexer

        self.assoc = None
        self.cache = {}
        self.is_warning = False

    def node_features_reshape(self, edge_index, x_src, x_dst, max_num_node=None, x_is_tuple=False):
        """
        Converts node features in shape (E, d) to a shape (N, d).
        Returns x as a tuple (x_src, x_dst).
        """
        # Only checked at the first call: `.min()` forces a GPU sync
        if not self.is_warning:
            self.is_warning = True
            if edge_index.min() != 0:
                print(
                    "Warning: reshaping features with non-reindexed edge index leads to large cache stored in GPU memory."
                )

        max_num_node = max_num_node + 1 if max_num_node else edge_index.max() + 1
        feature_dim = x_src.size(1)

        if feature_dim not in self.cache or self.cache[feature_dim].cache.shape[0] <= max_num_node:
            self.cache[feature_dim] = _Cache((max_num_node, feature_dim), self.device)
        self.cache[feature_dim].detach()

        # To avoid storing gradients from all nodes, we detach() BEFORE caching. If we detach()
        # after storing, we loose the gradient for all operations happening before the reindexing.
        output = self.cache[feature_dim].cache
        output.detach()
        output.zero_()

        # The cache may be larger than needed (it only grows): only its first `max_num_node`
        # rows are used (all indices are lower), and the returned tensors are compact copies.
        # Returning views of the cache would keep its whole storage alive and alias the
        # results of consecutive calls (the cache is zeroed at each call).
        output = output[:max_num_node]

        if x_is_tuple:
            scatter(x_src, edge_index[0], out=output, dim=0, reduce="mean")
            x_src_result = output.clone()
            output.zero_()

            scatter(x_dst, edge_index[1], out=output, dim=0, reduce="mean")
            x_dst_result = output.clone()
            return (x_src_result, x_dst_result)
        else:
            if self.fix_buggy_graph_reindexer:
                output = output.clone()
                scatter(
                    torch.cat([x_src, x_dst]),
                    torch.cat([edge_index[0], edge_index[1]]),
                    out=output,
                    dim=0,
                    reduce="mean",
                )
                return output
            else:
                # NOTE: this one, used in orthrus and velox is buggy because it does the mean twice, which can double
                # the value of features if duplicates exist
                scatter(x_src, edge_index[0], out=output, dim=0, reduce="mean")
                scatter(x_dst, edge_index[1], out=output, dim=0, reduce="mean")
                return output.clone()

    def reindex_graph(self, data, x_is_tuple=False, use_tgn=False, keep_edge_level_features=True):
        """
        Reindexes edge_index from 0 + reshapes node features.
        The original edge_index and node IDs are also kept.

        When keep_edge_level_features=False, x_src/x_dst and node_type_src/node_type_dst
        are deleted after being folded into x and node_type, saving ~4*E*d bytes per graph.
        """
        data.original_edge_index = data.edge_index
        x, edge_index, n_id = self._reindex_graph(
            data.edge_index, data.x_src, data.x_dst, x_is_tuple=x_is_tuple
        )
        data.original_n_id = n_id
        data.x = x

        if not use_tgn:
            data.src, data.dst = edge_index[0], edge_index[1]

        if x_is_tuple:
            data.x_src, data.x_dst = x
        else:
            data.x = x
            if not keep_edge_level_features:
                del data.x_src, data.x_dst


        data.node_type, *_ = self._reindex_graph(
            data.edge_index, data.node_type_src, data.node_type_dst, x_is_tuple=False
        )
        if not keep_edge_level_features:
            del data.node_type_src, data.node_type_dst

        return data

    def _reindex_graph(
        self, edge_index, x_src=None, x_dst=None, x_is_tuple=False, max_num_node=None
    ):
        """
        Reindexes edge_index with indices starting from 0.
        Also reshapes the node features.
        """
        if self.num_nodes is None:
            raise ValueError(f"Graph reindexing requires `num_nodes`.")

        if self.assoc is None:
            self.assoc = torch.empty((self.num_nodes,), dtype=torch.long, device=self.device)

        n_id = edge_index.unique()
        self.assoc[n_id] = torch.arange(n_id.size(0), device=self.assoc.device)
        edge_index = self.assoc[edge_index]

        if None not in [x_src, x_dst]:
            # Associates each feature vector to each reindexed node ID
            x = self.node_features_reshape(
                edge_index, x_src, x_dst, x_is_tuple=x_is_tuple, max_num_node=max_num_node
            )
        else:
            x = None

        return x, edge_index, n_id

    def clear_cache(self):
        """Free cached tensors to reclaim memory after reindexing is done."""
        self.cache.clear()
        self.assoc = None

    def to(self, device):
        self.device = device
        if self.assoc is not None:
            self.assoc = self.assoc.to(device)

        for k, v in self.cache.items():
            self.cache[k] = v.to(device)
        return self


def save_model(model, path: str):
    """
    Saves weights on disk.
    """
    os.makedirs(path, exist_ok=True)

    torch.save(
        model.state_dict(),
        os.path.join(path, "state_dict.pkl"),
        pickle_protocol=pickle.HIGHEST_PROTOCOL,
    )


def load_model(model, path: str):
    """
    Loads weights and tensors from disk into a model.
    """
    model.load_state_dict(torch.load(os.path.join(path, "state_dict.pkl")))

    return model


_REINDEX_INPUTS = ("src", "dst", "x_src", "x_dst", "node_type_src", "node_type_dst")


def _to_cpu_shared(value, copies):
    """Moves tensors (possibly nested in tuples/lists/dicts) to CPU, copying each tensor once."""
    if isinstance(value, torch.Tensor):
        key = id(value)
        if key not in copies:
            copies[key] = (value, value.cpu())  # keeps `value` alive so that its id stays unique
        return copies[key][1]
    if isinstance(value, (tuple, list)):
        return type(value)(_to_cpu_shared(v, copies) for v in value)
    if isinstance(value, dict):
        return {k: _to_cpu_shared(v, copies) for k, v in value.items()}
    return value


def reindex_graphs(datasets, graph_reindexer, device, use_tgn, keep_edge_level_features=True, x_is_tuple=False):
    for dataset in datasets:
        for data_list in dataset:
            for batch in log_tqdm(data_list, desc="Reindexing graphs"):
                # Reindexing runs on a device copy holding only its inputs; only the
                # new/updated attributes are copied back (and the deleted ones removed).
                tmp = copy.copy(batch)
                for key in list(tmp.keys()):
                    if isinstance(tmp[key], torch.Tensor) and key not in _REINDEX_INPUTS:
                        del tmp[key]
                tmp = tmp.to(device)
                inputs = dict(tmp._store.items())

                graph_reindexer.reindex_graph(
                    tmp,
                    use_tgn=use_tgn,
                    keep_edge_level_features=keep_edge_level_features,
                    x_is_tuple=x_is_tuple,
                )

                copies = {}  # id(device tensor) -> CPU copy, to keep aliased outputs aliased
                for key, value in tmp._store.items():
                    if key in inputs and value is inputs[key]:
                        continue
                    batch[key] = _to_cpu_shared(value, copies)
                for key in inputs:
                    if key not in tmp._store:
                        del batch[key]
    graph_reindexer.clear_cache()

def compute_hetero_graphs(datasets, device, cfg):
    node_map = get_node_map(from_zero=True)
    edge_map = get_rel2id(cfg, from_zero=True)

    for dataset in datasets:
        for data_list in log_tqdm(dataset, desc="Computing heterogeneous graphs"):
            for g in data_list:
                g.to(device)
                x_dict, edge_index_dict = compute_hetero_features(
                    g, node_map=node_map, edge_map=edge_map
                )
                g.x_dict = x_dict
                g.edge_index_dict = edge_index_dict
                g.to("cpu")
