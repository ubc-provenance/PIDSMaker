import os

import numpy as np
import torch

from pidsmaker.config import update_cfg_for_multi_dataset
from pidsmaker.featurization.feat_inference_methods import (
    feat_inference_alacarte,
    feat_inference_pretrained,
    feat_inference_doc2vec,
    feat_inference_fasttext,
    feat_inference_flash,
    feat_inference_HFH,
    feat_inference_ocrapt_features,
    feat_inference_TRW,
    feat_inference_word2vec,
)
from pidsmaker.featurization.feat_inference_methods.feat_inference_pretrained import GraphBasedEncoder
from pidsmaker.utils.data_utils import COMPACT_MSG_KEYS, CollatableTemporalData
from pidsmaker.utils.dataset_utils import (
    aggregated_edge_attribute_columns,
    encode_aggregated_edge_attributes,
    encode_edge_attributes,
    get_input_edge_attributes,
    uses_aggregated_edge_attributes,
    get_node_map,
    get_rel2id,
)
from pidsmaker.utils.graph_store import load_graph_columns
from pidsmaker.utils.utils import (
    gen_relation_onehot,
    get_multi_datasets,
    get_split_to_files,
    log_tqdm,
)


def _onehot_table(type2oh):
    """Stacks the one-hot vectors of a `gen_relation_onehot` mapping into a matrix.

    Returns the (num_types + 1, d) matrix, whose last row is all zeros (missing type),
    and the mapping {type_name: row}.
    """
    names = [k for k in type2oh if isinstance(k, str)]
    table = torch.stack([type2oh[k] for k in names] + [torch.zeros_like(type2oh[names[0]])])
    return table, {name: i for i, name in enumerate(names)}


def _edge_columns_from_graph(graph):
    """(nodes, node types, src rows, dst rows, times, labels, edge labels) in `graph.edges` order."""
    src, dst, t, y, labels = [], [], [], [], []
    for u, v, attr in graph.edges(data=True):
        src.append(u)
        dst.append(v)
        t.append(int(attr["time"]))
        y.append(int(attr.get("y", 0)))
        labels.append(attr.get("label"))

    # Map each node to a local row index, so that lookups are done once per node
    nodes = list(graph.nodes)
    node2row = {n: i for i, n in enumerate(nodes)}
    src_rows = torch.tensor([node2row[u] for u in src], dtype=torch.long)
    dst_rows = torch.tensor([node2row[v] for v in dst], dtype=torch.long)
    node_attrs = graph.nodes
    node_types = [node_attrs[n]["node_type"] for n in nodes]
    return nodes, node_types, src_rows, dst_rows, t, y, labels


def _edge_columns_from_store(cols):
    """Same as `_edge_columns_from_graph`, from the stored columns (no networkx graph)."""
    m = len(cols.src)
    t = [int(x) for x in cols.edge_attrs["time"]]
    y = [int(x) for x in cols.edge_attrs["y"]] if "y" in cols.edge_attrs else [0] * m
    labels = cols.edge_attrs["label"] if "label" in cols.edge_attrs else [None] * m
    src_rows = torch.from_numpy(cols.src.astype(np.int64))
    dst_rows = torch.from_numpy(cols.dst.astype(np.int64))
    return cols.node_ids, cols.node_attrs["node_type"], src_rows, dst_rows, t, y, labels


def _edge_attr_features(cfg, path, graph=None, cols=None):
    """Edge attribute features of the graph at `path`, in `graph.edges` order (None without
    attributes). `graph` or its stored columns `cols` are used when already loaded."""
    names = [a.name for a in get_input_edge_attributes(cfg)]
    if not names:
        return None
    if uses_aggregated_edge_attributes(cfg):
        # Attributes aggregated over the events of merged edges (see `merge_edges`)
        agg = [c for cols in aggregated_edge_attribute_columns(cfg).values() for c, _ in cols]
        if cols is not None and all(c in cols.edge_attrs for c in agg):
            columns = {c: list(cols.edge_attrs[c]) for c in agg}
        else:
            if graph is None:
                graph = torch.load(path)
            columns = {c: [] for c in agg}
            for _, _, attr in graph.edges(data=True):
                for c in agg:
                    columns[c].append(attr.get(c))
        return encode_aggregated_edge_attributes(cfg, columns)
    # Edges added by transformations (e.g. R-Caid's pseudo-graph) may have no attributes:
    # their missing attributes are null (OTHER for categorical, 0 for numeric attributes)
    if cols is not None and all(name in cols.edge_attrs for name in names):
        columns = {name: list(cols.edge_attrs[name]) for name in names}
    else:
        if graph is None:
            graph = torch.load(path)
        columns = {name: [] for name in names}
        for _, _, attr in graph.edges(data=True):
            for name in names:
                columns[name].append(attr.get(name))
    return encode_edge_attributes(cfg, columns)


def _edge_weights(path, graph=None, cols=None):
    """(E,) float tensor of the `weight` attribute of the edges, None if edges have no weight."""
    if cols is not None:
        if "weight" not in cols.edge_attrs:
            return None
        return torch.tensor([float(w) for w in cols.edge_attrs["weight"]], dtype=torch.float)
    if graph is None:
        graph = torch.load(path)
    weights = [attr.get("weight") for _, _, attr in graph.edges(data=True)]
    if not weights or any(w is None for w in weights):
        return None
    return torch.tensor(weights, dtype=torch.float)


def _vectorized_edge_msg(edge_columns, indexid2vec, etype2oh, ntype2oh):
    """Builds the edge messages of a graph with per-node (not per-edge) lookups.

    Equivalent to concatenating, for each edge in `graph.edges` order,
    [src_type_onehot, src_emb, edge_type_onehot, dst_type_onehot, dst_emb].
    """
    nodes, node_type_list, src_rows, dst_rows, t, y, labels = edge_columns

    # Edge endpoints in the original node order, before restricting the rows below
    node_ids = [int(n) for n in nodes]
    src = [node_ids[r] for r in src_rows.tolist()]
    dst = [node_ids[r] for r in dst_rows.tolist()]

    # Only nodes with edges get features: like the former per-edge lookups, isolated nodes
    # are never looked up (they may have no embedding, e.g. with edge-based featurizations)
    used = torch.unique(torch.cat([src_rows, dst_rows]))
    remap = torch.full((len(nodes),), -1, dtype=torch.long)
    remap[used] = torch.arange(used.numel())
    src_rows, dst_rows = remap[src_rows], remap[dst_rows]
    used = used.tolist()
    nodes = [nodes[i] for i in used]
    node_type_list = [node_type_list[i] for i in used]

    ntype_table, ntype2row = _onehot_table(ntype2oh)
    node_type_rows = torch.tensor([ntype2row[nt] for nt in node_type_list], dtype=torch.long)
    node_types = ntype_table[node_type_rows]

    etype_table, etype2row = _onehot_table(etype2oh)
    missing = etype_table.shape[0] - 1
    edge_type_rows = torch.tensor(
        [missing if l is None else etype2row[l] for l in labels], dtype=torch.long
    )

    if indexid2vec is None:
        node_feats = node_types
    else:
        # indexid2vec may use string keys (word2vec) or int keys;
        # graph node IDs are cast to int, so try both.
        def _lookup(nid):
            nid = int(nid)
            try:
                return indexid2vec[nid]
            except KeyError:
                return indexid2vec[str(nid)]

        node_embs = torch.from_numpy(np.stack([_lookup(n) for n in nodes])).float()
        node_feats = torch.cat([node_types, node_embs], dim=1)

    # The message of an edge is [src_feats | edge_type | dst_feats], where a node's feats
    # are [node_type | node_emb]. Instead of materializing it (E x D, with each node's
    # embedding repeated for each of its edges), each node's feats are stored once with
    # the row of each edge's src/dst/type: see `expand_compact_msg`.
    compact = {
        COMPACT_MSG_KEYS[0]: node_feats.to(torch.float),
        COMPACT_MSG_KEYS[1]: src_rows.to(torch.int32),
        COMPACT_MSG_KEYS[2]: dst_rows.to(torch.int32),
        COMPACT_MSG_KEYS[3]: etype_table.to(torch.float),
        COMPACT_MSG_KEYS[4]: edge_type_rows.to(torch.int32),
    }

    return compact, src, dst, t, y


def feat_inference(indexid2vec, etype2oh, ntype2oh, sorted_paths, out_dir, cfg, bert_generator=None, indexid2msg=None):
    """Generate edge embeddings from node embeddings or a SPIDER encoder.

    Args:
        indexid2vec: Pre-computed node embeddings (None for spider or only_type methods)
        etype2oh: Edge type one-hot encodings
        ntype2oh: Node type one-hot encodings
        sorted_paths: Paths to graph files
        out_dir: Output directory for edge embeddings
        cfg: Configuration object
        bert_generator: Optional SpiderEmbeddingGenerator for temporal embeddings
        indexid2msg: Optional node label mapping for SPIDER
    """
    for i, path in enumerate(log_tqdm(sorted_paths, desc="Computing edge embeddings")):
        # The standard path only needs the graph's columns, not a networkx graph
        cols = load_graph_columns(path) if bert_generator is None else None
        graph = torch.load(path) if cols is None else None

        src, dst, msg, t, y = [], [], [], [], []

        # For SPIDER: batch process edges to avoid memory issues with millions of edges
        if bert_generator is not None:
            # Collect all edge data first
            all_edge_data = []
            for u, v, k, attr in graph.edges(data=True, keys=True):
                src.append(int(u))
                dst.append(int(v))
                timestamp = int(attr["time"])
                t.append(timestamp)
                y.append(int(attr.get("y", 0)))

                if "label" in attr:
                    edge_label = etype2oh[attr["label"]]
                else:
                    edge_label = torch.zeros_like(etype2oh[list(etype2oh.keys())[0]])

                all_edge_data.append({
                    'src': u,
                    'dst': v,
                    'timestamp': timestamp,
                    'src_type': ntype2oh[graph.nodes[u]["node_type"]],
                    'dst_type': ntype2oh[graph.nodes[v]["node_type"]],
                    'edge_label': edge_label,
                })

            # Process edges in batches to avoid memory issues
            edge_batch_size = cfg.feat_inference.edge_batch_size

            for batch_start in range(0, len(all_edge_data), edge_batch_size):
                batch_edges = all_edge_data[batch_start : batch_start + edge_batch_size]

                # Prepare nodes and timestamps for this batch
                batch_nodes = []
                batch_timestamps = []
                for e in batch_edges:
                    batch_nodes.extend([e['src'], e['dst']])
                    batch_timestamps.extend([e['timestamp'], e['timestamp']])

                # Generate embeddings for this batch (handles src+dst together)
                batch_embeddings_list = bert_generator.get_batch_embeddings(
                    nodes=batch_nodes, graph=graph, indexid2msg=indexid2msg, timestamps=batch_timestamps
                )

                # Build messages for this batch vectorized
                # Embeddings are ordered as: [src0, dst0, src1, dst1, ...]
                src_embs = torch.from_numpy(
                    np.stack([batch_embeddings_list[idx * 2] for idx in range(len(batch_edges))])
                ).float()
                dst_embs = torch.from_numpy(
                    np.stack([batch_embeddings_list[idx * 2 + 1] for idx in range(len(batch_edges))])
                ).float()
                src_types = torch.stack([e['src_type'] for e in batch_edges])
                dst_types = torch.stack([e['dst_type'] for e in batch_edges])
                edge_labels = torch.stack([e['edge_label'] for e in batch_edges])
                msg.append(torch.cat([src_types, src_embs, edge_labels, dst_types, dst_embs], dim=1))

        # Standard processing (pre-computed embeddings or types only)
        else:
            num_edges = graph.number_of_edges() if cols is None else len(cols.src)
            if num_edges == 0:
                continue  # skip empty graphs (no edges in this time window)

            edge_columns = (
                _edge_columns_from_graph(graph) if cols is None else _edge_columns_from_store(cols)
            )
            compact, src, dst, t, y = _vectorized_edge_msg(edge_columns, indexid2vec, etype2oh, ntype2oh)

        data = CollatableTemporalData(
            src=torch.tensor(src).to(torch.long),
            dst=torch.tensor(dst).to(torch.long),
            t=torch.tensor(t).to(torch.long),
            msg=torch.vstack(msg).to(torch.float) if bert_generator is not None else None,
            y=torch.tensor(y).to(torch.long),
        )
        if bert_generator is None:
            for key, value in compact.items():
                data[key] = value
        edge_attrs = _edge_attr_features(cfg, path, graph=graph, cols=cols)
        if edge_attrs is not None:
            data.edge_attr_feats = edge_attrs
        # Weights of the edges merged by the `merge_edges` transformation
        edge_weight = _edge_weights(path, graph=graph, cols=cols)
        if edge_weight is not None:
            data.edge_weight = edge_weight

        os.makedirs(out_dir, exist_ok=True)
        file = path.split("/")[-1]
        torch.save(data, os.path.join(out_dir, f"{file}.TemporalData.simple"))


def get_indexid2vec(cfg):
    method = cfg.featurization.used_method.strip()
    if method in ["only_type", "only_ones"]:
        return None
    if method == "alacarte":
        return feat_inference_alacarte.main(cfg)
    if method == "doc2vec":
        return feat_inference_doc2vec.main(cfg)
    if method == "hierarchical_hashing":
        return feat_inference_HFH.main(cfg)
    if method == "word2vec":
        return feat_inference_word2vec.main(cfg)
    if method == "ocrapt_features":
        return feat_inference_ocrapt_features.main(cfg)
    if method == "temporal_rw":
        return feat_inference_TRW.main(cfg)
    if method == "flash":
        return feat_inference_flash.main(cfg)
    if method == "fasttext":
        return feat_inference_fasttext.main(cfg)
    if method == "pretrained":
        return feat_inference_pretrained.main(cfg)

    raise ValueError(f"Invalid node embedding method {method}")


def main_from_config(cfg):
    # When gnn_training uses spider, detection reads raw graphs directly
    # and never consumes the .TemporalData.simple files produced here.
    if cfg.training.used_method.strip() == "pretrained":
        from pidsmaker.utils.utils import log
        log("feat_inference: skipping edge embedding computation — "
            "gnn_training.used_method=spider reads raw graphs directly.")
        return

    rel2id = get_rel2id(cfg)
    ntype2id = get_node_map(cfg)
    etype2onehot = gen_relation_onehot(rel2id=rel2id)
    ntype2onehot = gen_relation_onehot(rel2id=ntype2id)

    base_dir = cfg.transformation._graphs_dir
    split_to_files = get_split_to_files(cfg, base_dir)

    # Here we get either:
    #   - a mapping {node_id => embedding vector} for non-graph methods
    #   - a GraphBasedEncoder for graph-based methods (GraphMAE/GAE/DGI),
    #     which computes per-graph embeddings to avoid cross-graph data snooping
    indexid2vec_or_encoder = get_indexid2vec(cfg)

    graph_encoder = None
    if isinstance(indexid2vec_or_encoder, GraphBasedEncoder):
        graph_encoder = indexid2vec_or_encoder
        indexid2vec = None  # will be computed per-graph
    else:
        indexid2vec = indexid2vec_or_encoder

    bert_generator = None
    indexid2msg = None

    # Create edges for Train, Val, Test sets
    for split, sorted_paths in split_to_files.items():
        if graph_encoder is not None:
            # Per-graph encoding: each graph gets embeddings from its own structure only
            for path in log_tqdm(sorted_paths, desc=f"Computing edge embeddings ({split})"):
                per_graph_indexid2vec = graph_encoder.encode_graph(path)
                feat_inference(
                    indexid2vec=per_graph_indexid2vec,
                    etype2oh=etype2onehot,
                    ntype2oh=ntype2onehot,
                    sorted_paths=[path],
                    out_dir=os.path.join(cfg.feat_inference._edge_embeds_dir, f"{split}/"),
                    cfg=cfg,
                    bert_generator=bert_generator,
                    indexid2msg=indexid2msg,
                )
        else:
            feat_inference(
                indexid2vec=indexid2vec,
                etype2oh=etype2onehot,
                ntype2oh=ntype2onehot,
                sorted_paths=sorted_paths,
                out_dir=os.path.join(cfg.feat_inference._edge_embeds_dir, f"{split}/"),
                cfg=cfg,
                bert_generator=bert_generator,
                indexid2msg=indexid2msg,
            )

    # Release GPU memory if using graph encoder
    if graph_encoder is not None:
        graph_encoder.cleanup()


def main(cfg):
    multi_dataset_training = cfg.batching.multi_dataset_training
    if not multi_dataset_training:
        main_from_config(cfg)

    # Multi-dataset mode
    else:
        trained_model_dir = cfg.featurization._model_dir
        multi_datasets = get_multi_datasets(cfg)
        for dataset in multi_datasets:
            updated_cfg, should_restart = update_cfg_for_multi_dataset(cfg, dataset)
            updated_cfg.featurization._model_dir = trained_model_dir

            if should_restart["feat_inference"]:
                main_from_config(updated_cfg)
