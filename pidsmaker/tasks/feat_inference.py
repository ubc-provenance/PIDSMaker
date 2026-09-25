import os

import numpy as np
import torch

from pidsmaker.config import update_cfg_for_multi_dataset
from pidsmaker.featurization.feat_inference_methods import (
    feat_inference_alacarte,
    feat_inference_spider,
    feat_inference_doc2vec,
    feat_inference_fasttext,
    feat_inference_flash,
    feat_inference_HFH,
    feat_inference_ocrapt_features,
    feat_inference_TRW,
    feat_inference_word2vec,
)
from pidsmaker.featurization.feat_inference_methods.feat_inference_spider import GraphBasedEncoder
from pidsmaker.utils.data_utils import CollatableTemporalData
from pidsmaker.utils.dataset_utils import get_node_map, get_rel2id
from pidsmaker.utils.utils import (
    gen_relation_onehot,
    get_multi_datasets,
    get_split_to_files,
    log_tqdm,
)


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
        graph = torch.load(path)
        sorted_edges = graph.edges(data=True, keys=True)

        src, dst, msg, t, y = [], [], [], [], []

        # For SPIDER: batch process edges to avoid memory issues with millions of edges
        if bert_generator is not None:
            # Collect all edge data first
            all_edge_data = []
            for u, v, k, attr in sorted_edges:
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
            edges_data = list(sorted_edges)

            if not edges_data:
                continue  # skip empty graphs (no edges in this time window)

            src = [int(u) for u, v, k, attr in edges_data]
            dst = [int(v) for u, v, k, attr in edges_data]
            t = [int(attr["time"]) for u, v, k, attr in edges_data]
            y = [int(attr.get("y", 0)) for u, v, k, attr in edges_data]

            # Batch all type and label lookups — avoids N individual torch.cat calls
            zero_edge_label = torch.zeros_like(etype2oh[next(iter(etype2oh))])
            ntype_u = torch.stack([ntype2oh[graph.nodes[u]["node_type"]] for u, v, k, attr in edges_data])
            ntype_v = torch.stack([ntype2oh[graph.nodes[v]["node_type"]] for u, v, k, attr in edges_data])
            edge_label_t = torch.stack([
                etype2oh[attr["label"]] if "label" in attr else zero_edge_label
                for u, v, k, attr in edges_data
            ])

            # Only types
            if indexid2vec is None:
                msg = [torch.cat([ntype_u, edge_label_t, ntype_v], dim=1)]

            # Types + node embeddings
            else:
                # indexid2vec may use string keys (word2vec) or int keys;
                # graph node IDs are cast to int above, so try both.
                def _lookup(nid):
                    try:
                        return indexid2vec[nid]
                    except KeyError:
                        return indexid2vec[str(nid)]
                src_embs = torch.from_numpy(np.stack([_lookup(u) for u in src])).float()
                dst_embs = torch.from_numpy(np.stack([_lookup(v) for v in dst])).float()
                msg = [torch.cat([ntype_u, src_embs, edge_label_t, ntype_v, dst_embs], dim=1)]

        data = CollatableTemporalData(
            src=torch.tensor(src).to(torch.long),
            dst=torch.tensor(dst).to(torch.long),
            t=torch.tensor(t).to(torch.long),
            msg=torch.vstack(msg).to(torch.float),
            y=torch.tensor(y).to(torch.long),
        )

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
    if method == "spider":
        return feat_inference_spider.main(cfg)

    raise ValueError(f"Invalid node embedding method {method}")


def main_from_config(cfg):
    # When gnn_training uses spider, detection reads raw graphs directly
    # and never consumes the .TemporalData.simple files produced here.
    if cfg.training.used_method.strip() == "spider":
        from pidsmaker.utils.utils import log
        log("feat_inference: skipping edge embedding computation — "
            "gnn_training.used_method=spider reads raw graphs directly.")
        return

    rel2id = get_rel2id(cfg)
    ntype2id = get_node_map()
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
