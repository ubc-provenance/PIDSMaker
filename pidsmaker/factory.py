import os

import torch
import torch.nn as nn

from pidsmaker.config import decoder_matches_objective
from pidsmaker.decoders import *
from pidsmaker.encoders import *
from pidsmaker.experiments.uncertainty import add_dropout_to_model
from pidsmaker.losses import *
from pidsmaker.model import Model
from pidsmaker.objectives import *
from pidsmaker.tgn import IdentityMessage, LastAggregator, TGNMemory, TimeEncodingMemory
from pidsmaker.utils.data_utils import GraphReindexer
from pidsmaker.utils.dataset_utils import (
    get_node_map,
    get_num_edge_type,
    get_rel2id,
    possible_events,
    OPTC_DATASETS,
)
from pidsmaker.hetero import get_metadata


def build_model(data_sample, device, cfg, max_node_num, all_data=None):
    """
    Builds and loads the initial model into memory.
    The `data_sample` is required to infer the shape of the layers.
    `all_data` is an optional tuple (val_data, test_data) used by the
    predict_edge_supervised objective to extract attack edge features.
    """
    msg_dim, edge_dim, in_dim = get_dimensions_from_data_sample(data_sample)

    graph_reindexer = GraphReindexer(
        device=device,
        num_nodes=max_node_num,
        fix_buggy_graph_reindexer=cfg.detection.graph_preprocessing.fix_buggy_graph_reindexer,
    )

    encoder = encoder_factory(
        cfg,
        msg_dim=msg_dim,
        in_dim=in_dim,
        edge_dim=edge_dim,
        device=device,
        max_node_num=max_node_num,
        graph_reindexer=graph_reindexer,
    )
    objectives = objective_factory(
        cfg, in_dim=in_dim, graph_reindexer=graph_reindexer, device=device,
        all_data=all_data,
    )
    objective_few_shot = few_shot_decoder_factory(
        cfg, device=device, graph_reindexer=graph_reindexer
    )
    model = model_factory(encoder, objectives, objective_few_shot, cfg, device=device)

    if cfg._is_running_mc_dropout:
        dropout = cfg.experiment.uncertainty.mc_dropout.dropout
        add_dropout_to_model(model, p=dropout)

    return model


def model_factory(encoder, objectives, objective_few_shot, cfg, device):
    return Model(
        encoder=encoder,
        objectives=objectives,
        objective_few_shot=objective_few_shot,
        device=device,
        is_running_mc_dropout=cfg._is_running_mc_dropout,
        use_few_shot=cfg.detection.gnn_training.decoder.use_few_shot,
        freeze_encoder=cfg.detection.gnn_training.decoder.few_shot.freeze_encoder,
        fuse_duplicate_edges_training=cfg.detection.gnn_training.fuse_duplicate_edges_training,
        is_hybrid_loss=cfg._is_hybrid_loss,
    ).to(device)


def encoder_block_factory(cfg, msg_dim, in_dim, edge_dim, device, max_node_num):
    node_hid_dim = cfg.detection.gnn_training.node_hid_dim
    node_out_dim = cfg.detection.gnn_training.node_out_dim
    dropout = cfg.detection.gnn_training.encoder.dropout
    tgn_memory_dim = cfg.detection.gnn_training.encoder.tgn.tgn_memory_dim
    tgn_time_dim = cfg.detection.gnn_training.encoder.tgn.tgn_time_dim
    use_tgn = "tgn" in cfg.detection.gnn_training.encoder.used_methods
    use_event_type_encoding = (
        "event_type_encoding" in cfg.detection.gnn_training.encoder.used_methods
    )

    node_map = get_node_map(from_zero=True)

    # If edge features are used, we set them here
    # edge_dim = 0
    # edge_features = list(
    #     map(lambda x: x.strip(), cfg.detection.graph_preprocessing.edge_features.split(","))
    # )
    # for edge_feat in edge_features:
    #     if edge_feat in ["edge_type", "edge_type_triplet"]:
    #         edge_dim += get_num_edge_type(cfg)
    #     elif edge_feat == "msg":
    #         edge_dim += msg_dim
    #     elif edge_feat == "time_encoding":
    #         if not use_tgn:
    #             raise TypeError("Edge feature `time_encoding` is only available if TGN is used.")
    #         edge_dim += tgn_memory_dim
    #     elif edge_feat == "none":
    #         pass
    #     else:
    #         raise ValueError(f"Invalid edge feature {edge_feat}")

    if use_tgn:
        in_dim = tgn_memory_dim
    
    if use_event_type_encoding:
        edge_dim = in_dim
        
    edge_features = cfg.detection.graph_preprocessing.edge_features
    if "time_encoding" in edge_features:
        edge_dim += tgn_time_dim
    
    for method in map(
        lambda x: x.strip(),
        cfg.detection.gnn_training.encoder.used_methods.replace("-", ",").split(","),
    ):
        if method in ["tgn", "ancestor_encoding", "entity_type_encoding", "event_type_encoding"]:
            pass

        # Basic GNN encoders
        elif method == "graph_attention":
            encoder = GraphAttentionEmbedding(
                in_dim=in_dim,
                hid_dim=node_hid_dim,
                out_dim=node_out_dim,
                edge_dim=edge_dim or None,
                activation=activation_fn_factory(
                    cfg.detection.gnn_training.encoder.graph_attention.activation
                ),
                dropout=dropout,
                num_heads=cfg.detection.gnn_training.encoder.graph_attention.num_heads,
                concat=cfg.detection.gnn_training.encoder.graph_attention.concat,
                flow=cfg.detection.gnn_training.encoder.graph_attention.flow,
                num_layers=cfg.detection.gnn_training.encoder.graph_attention.num_layers,
            )
        elif method == "hetero_graph_transformer":
            if cfg.dataset.name in OPTC_DATASETS:
                raise NotImplementedError(
                    "Hetero OPTC not implemented (need to compute possible_events)"
                )

            node_map = get_node_map(from_zero=True)
            metadata = get_metadata(possible_events, node_map)

            encoder = HeteroGraphTransformer(
                in_dim=in_dim,
                out_dim=node_out_dim,
                num_heads=cfg.detection.gnn_training.encoder.hetero_graph_transformer.num_heads,
                num_layers=cfg.detection.gnn_training.encoder.hetero_graph_transformer.num_layers,
                metadata=metadata,
                device=device,
                node_map=node_map,
            )
        elif method == "sage":
            encoder = SAGE(
                in_dim=in_dim,
                hid_dim=node_hid_dim,
                out_dim=node_out_dim,
                activation=activation_fn_factory(
                    cfg.detection.gnn_training.encoder.sage.activation
                ),
                dropout=dropout,
                num_layers=cfg.detection.gnn_training.encoder.sage.num_layers,
            )
        elif method == "gat":
            encoder = GAT(
                in_dim=in_dim,
                hid_dim=node_hid_dim,
                out_dim=node_out_dim,
                activation=activation_fn_factory(cfg.detection.gnn_training.encoder.gat.activation),
                dropout=dropout,
                num_heads=cfg.detection.gnn_training.encoder.gat.num_heads,
                concat=cfg.detection.gnn_training.encoder.gat.concat,
                num_layers=cfg.detection.gnn_training.encoder.gat.num_layers,
            )
        elif method == "gin":
            encoder = GIN(
                in_dim=in_dim,
                hid_dim=node_hid_dim,
                out_dim=node_out_dim,
                edge_dim=edge_dim or None,
                dropout=dropout,
                activation=activation_fn_factory(cfg.detection.gnn_training.encoder.gin.activation),
                num_layers=cfg.detection.gnn_training.encoder.gin.num_layers,
            )
        elif method == "sum_aggregation":
            encoder = SumAggregation(
                in_dim=in_dim,
                hid_dim=node_hid_dim,
                out_dim=node_out_dim,
            )

        # System-specific encoders
        elif method == "glstm":
            encoder = GLSTM(
                in_features=in_dim,
                out_features=node_out_dim,
                cell_clip=None,
                type_specific_decoding=False,
                exclude_file=True,
                exclude_ip=True,
                typed_hidden_rep=False,
                edge_dim=None,
                full_param=False,
                num_edge_type=15,  # TODO: we should use 10 here
            ).to(device)
        elif method == "rcaid_gat":
            encoder = RCaidGAT(
                in_dim=in_dim,
                hid_dim=node_hid_dim,
                out_dim=node_out_dim,
                dropout=dropout,
            )
        elif method == "sum_aggregation":
            encoder = SumAggregation(
                in_dim=in_dim,
                hid_dim=node_hid_dim,
                out_dim=node_out_dim,
            )
        elif method == "magic_gat":
            n_layers = cfg.detection.gnn_training.encoder.magic_gat.num_layers
            n_heads = cfg.detection.gnn_training.encoder.magic_gat.num_heads
            negative_slope = cfg.detection.gnn_training.encoder.magic_gat.negative_slope
            assert node_hid_dim % n_heads == 0, "Invalid shape dim for number of heads"

            encoder = MagicGAT(
                in_dim=in_dim,
                hid_dim=node_hid_dim,
                out_dim=node_out_dim,
                n_layers=n_layers,
                n_heads=n_heads,
                feat_drop=0.1,
                attn_drop=0.0,
                negative_slope=negative_slope,
                concat_out=True,
                residual=True,
                activation=activation_fn_factory(
                    cfg.detection.gnn_training.encoder.magic_gat.activation
                ),
                is_decoder=False,
            )

        # MLP encoders
        elif method == "none":
            encoder = LinearEncoder(in_dim, node_out_dim)
        elif method == "custom_mlp":
            encoder = CustomMLPEncoder(
                in_dim=in_dim,
                out_dim=node_out_dim,
                architecture=cfg.detection.gnn_training.encoder.custom_mlp.architecture_str,
                dropout=dropout,
            )
        else:
            raise ValueError(f"Invalid encoder {method}")
    
    return encoder

def inner_encoder_factory(cfg, in_dim, encoder, original_edge_dim, node_map, edge_map, max_node_num, device):
    use_ancestor_encoding = "ancestor_encoding" in cfg.detection.gnn_training.encoder.used_methods
    use_entity_type_encoding = (
        "entity_type_encoding" in cfg.detection.gnn_training.encoder.used_methods
    )
    use_event_type_encoding = (
        "event_type_encoding" in cfg.detection.gnn_training.encoder.used_methods
    )
    if use_entity_type_encoding:
        encoder = EntityLinearEncoder(
            in_dim=in_dim,
            out_dim=in_dim,
            encoder=encoder,
            activation=True,
        )

    if use_event_type_encoding:
        encoder = EventLinearEncoder(
            in_dim=original_edge_dim,
            out_dim=in_dim,
            possible_events=possible_events,
            node_map=node_map,
            edge_map=edge_map,
            encoder=encoder,
            activation=True,
        )

    if use_ancestor_encoding:
        encoder = AncestorEncoder(
            in_dim=in_dim,
            out_dim=in_dim,  # try in_dim*2 ou out_dim
            edge_dim=edge_dim,
            encoder=encoder,
            num_nodes=max_node_num,
            device=device,
        )
        
    return encoder

def encoder_factory(cfg, msg_dim, in_dim, edge_dim, device, max_node_num, graph_reindexer):
    node_out_dim = cfg.detection.gnn_training.node_out_dim
    dropout = cfg.detection.gnn_training.encoder.dropout
    use_tgn = "tgn" in cfg.detection.gnn_training.encoder.used_methods
    node_map = get_node_map(from_zero=True)
    edge_map = get_rel2id(cfg, from_zero=True)
        
    encoder = encoder_block_factory(cfg, msg_dim, in_dim, edge_dim, device, max_node_num)
    tgn_pre_encoder = None
            
    encoder = inner_encoder_factory(cfg, in_dim, encoder, edge_dim, node_map, edge_map, max_node_num, device)
    
    tgn_cfg = cfg.detection.gnn_training.encoder.tgn
    time_dim = tgn_cfg.tgn_time_dim
    use_node_feats_in_gnn = tgn_cfg.use_node_feats_in_gnn
    use_memory = tgn_cfg.use_memory
    use_time_order_encoding = tgn_cfg.use_time_order_encoding
    project_src_dst = tgn_cfg.project_src_dst
    tgn_memory_dim = tgn_cfg.tgn_memory_dim
    edge_features = cfg.detection.graph_preprocessing.edge_features
    use_time_enc = "time_encoding" in edge_features

    tgn_in_dim = in_dim

    memory = None
    if use_tgn:
        if use_memory:
            memory = TGNMemory(
                max_node_num,
                msg_dim,
                tgn_memory_dim,
                time_dim,
                message_module=IdentityMessage(msg_dim, tgn_memory_dim, time_dim),
                aggregator_module=LastAggregator(),
                device=device,
            )
        elif use_time_enc:
            memory = TimeEncodingMemory(
                max_node_num,
                time_dim,
                device=device,
            )
    
    # Standard TGN
    if use_tgn:
        encoder = TGNEncoder(
            encoder=encoder,
            memory=memory,
            time_encoder=memory.time_enc if memory else None,
            in_dim=tgn_in_dim,
            memory_dim=tgn_memory_dim,
            out_dim=node_out_dim,
            use_node_feats_in_gnn=use_node_feats_in_gnn,
            edge_features=edge_features,
            device=device,
            use_memory=use_memory,
            use_time_enc=use_time_enc,
            edge_dim=edge_dim,
            use_time_order_encoding=use_time_order_encoding,
            project_src_dst=project_src_dst,
            is_hetero=cfg._is_hetero,
            node_map=node_map,
            edge_map=edge_map,
            tgn_pre_encoder=tgn_pre_encoder,
            dropout=dropout,
            use_residual_norm=tgn_cfg.use_residual_norm,
        )

    return encoder


def decoder_factory(method, objective, cfg, in_dim, out_dim, device, objective_cfg=None):
    if objective_cfg is None:
        objective_cfg = cfg.detection.gnn_training.decoder
    decoder_cfg = getattr(getattr(objective_cfg, objective), method, None)

    if method == "edge_mlp":
        return CustomEdgeMLP(
            in_dim=in_dim,
            out_dim=out_dim,
            architecture=decoder_cfg.architecture_str,
            dropout=cfg.detection.gnn_training.encoder.dropout,
            src_dst_projection_coef=decoder_cfg.src_dst_projection_coef,
        )
    elif method == "node_mlp":
        return CustomMLPDecoder(
            in_dim=in_dim,
            out_dim=out_dim,
            architecture=decoder_cfg.architecture_str,
            dropout=cfg.detection.gnn_training.encoder.dropout,
        )
    elif method == "nodlink":
        return NodLinkDecoder(
            in_dim=in_dim,
            out_dim=out_dim,
            device=device,
        )
    elif method == "magic_gat":
        n_layers = decoder_cfg.num_layers
        n_heads = decoder_cfg.num_heads
        negative_slope = decoder_cfg.negative_slope

        return MagicGAT(
            in_dim=in_dim,
            hid_dim=in_dim,
            out_dim=out_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            feat_drop=0.1,
            attn_drop=0.0,
            negative_slope=negative_slope,
            concat_out=True,
            residual=True,
            activation=activation_fn_factory(
                cfg.detection.gnn_training.encoder.magic_gat.activation
            ),
            is_decoder=True,
        )
    elif method == "none":
        return lambda x: x
    else:
        raise ValueError(f"Invalid decoder {method}")


def _load_attack_features(cfg, edge_scores_path, top_n, all_data):
    """Extract x_src and x_dst tensors for the top-N highest-loss attack edges.

    Matches edges from the edge_scores pickle (srcnode, dstnode, time) against
    the already-preprocessed val+test graph data using original node IDs.
    all_data is a tuple (val_data, test_data) where each element is a list of
    dataset lists (list[list[CollatableTemporalData]]).
    """
    import pandas as pd
    from pidsmaker.utils.utils import log

    edge_scores = torch.load(edge_scores_path, map_location="cpu")
    top_attacks = edge_scores.nlargest(top_n, "loss")[["srcnode", "dstnode", "time"]]
    attack_keys = set(
        zip(top_attacks["srcnode"].tolist(),
            top_attacks["dstnode"].tolist(),
            top_attacks["time"].tolist())
    )

    atk_x_src_list, atk_x_dst_list, atk_edge_type_list = [], [], []
    found_keys = set()
    n_graphs_scanned = 0

    for split_data in all_data:          # (val_data, test_data)
        for dataset in split_data:       # list of datasets (multi-dataset)
            for g in dataset:            # individual time-window batches
                # original_edge_index holds global IDs (before reindexing);
                # g.src/g.dst have been overwritten with local 0..N-1 indices.
                orig_ei = g.original_edge_index.cpu()
                src_np = orig_ei[0].numpy()
                dst_np = orig_ei[1].numpy()
                t_np = g.t.cpu().numpy()
                n_graphs_scanned += 1

                for i, (s, d, t) in enumerate(zip(src_np, dst_np, t_np)):
                    key = (int(s), int(d), int(t))
                    if key in attack_keys and key not in found_keys:
                        atk_x_src_list.append(g.x_src[i].cpu())
                        atk_x_dst_list.append(g.x_dst[i].cpu())
                        atk_edge_type_list.append(g.edge_type[i].cpu())
                        found_keys.add(key)

    if not atk_x_src_list:
        # Diagnostic: show sample values from both sides to help identify the mismatch
        sample_attack = list(attack_keys)[:3]
        sample_data = []
        for split_data in all_data:
            for dataset in split_data:
                for g in dataset:
                    s0 = int(g.original_edge_index[0, 0].item())
                    d0 = int(g.original_edge_index[1, 0].item())
                    t0 = int(g.t[0].item())
                    sample_data.append((s0, d0, t0))
                    if len(sample_data) >= 3:
                        break
                if len(sample_data) >= 3:
                    break
            if len(sample_data) >= 3:
                break
        raise ValueError(
            f"predict_edge_supervised: no attack edges found after scanning {n_graphs_scanned} graph batches.\n"
            f"  Attack keys (srcnode, dstnode, time) sample: {sample_attack}\n"
            f"  Graph data (src, dst, t) sample:             {sample_data}\n"
            "Check that the edge_scores_path matches the dataset being trained on."
        )

    log(
        f"predict_edge_supervised: loaded {len(atk_x_src_list)}/{len(attack_keys)} "
        f"attack edge features from '{edge_scores_path}'"
    )
    return torch.stack(atk_x_src_list), torch.stack(atk_x_dst_list), torch.stack(atk_edge_type_list)


def _load_attack_features_from_patterns(cfg, attack_patterns, max_edges_per_pattern, all_data):
    """Collect attack edge feature tensors by matching hand-crafted TTP patterns.

    Each pattern is a dict with any subset of:
      src_type           : "file" | "subject" | "netflow"
      src_label_contains : substring that must appear in the src node path/cmd
      dst_type           : "file" | "subject" | "netflow"
      dst_label_contains : substring that must appear in the dst node path/cmd
      edge_type          : event name string, e.g. "EVENT_EXECUTE"

    Omitting a field means "match any". At most `max_edges_per_pattern` edges
    are collected per pattern. Duplicates across patterns are de-duplicated.

    all_data is (val_data, test_data) — same structure as in _load_attack_features.
    """
    from pidsmaker.utils.utils import get_indexid2msg, log

    # node_id (int or str) -> [node_type_str, label_str]
    indexid2msg = get_indexid2msg(cfg)
    # Normalise keys to int for fast lookup
    indexid2msg_int = {int(k): v for k, v in indexid2msg.items()}

    rel2id_zero = get_rel2id(cfg, from_zero=True)  # event_name -> 0-based index

    atk_x_src_list, atk_x_dst_list, atk_edge_type_list = [], [], []
    seen = set()

    for pat_idx, pattern in enumerate(attack_patterns):
        src_type      = pattern.get("src_type")
        src_contains  = pattern.get("src_label_contains")
        dst_type      = pattern.get("dst_type")
        dst_contains  = pattern.get("dst_label_contains")
        etype_name    = pattern.get("edge_type")
        etype_idx     = rel2id_zero.get(etype_name) if etype_name else None

        collected = 0
        for split_data in all_data:
            for dataset in split_data:
                for g in dataset:
                    if collected >= max_edges_per_pattern:
                        break

                    orig_ei  = g.original_edge_index.cpu()
                    src_ids  = orig_ei[0].tolist()
                    dst_ids  = orig_ei[1].tolist()
                    etypes   = g.edge_type  # (E, num_edge_types)

                    for i, (s_id, d_id) in enumerate(zip(src_ids, dst_ids)):
                        if collected >= max_edges_per_pattern:
                            break
                        uid = (int(s_id), int(d_id), int(g.t[i].item()))
                        if uid in seen:
                            continue

                        s_info = indexid2msg_int.get(int(s_id))
                        d_info = indexid2msg_int.get(int(d_id))
                        if s_info is None or d_info is None:
                            continue

                        s_ntype, s_label = s_info[0], s_info[1]
                        d_ntype, d_label = d_info[0], d_info[1]

                        if src_type     and s_ntype != src_type:             continue
                        if src_contains and src_contains not in s_label:     continue
                        if dst_type     and d_ntype != dst_type:             continue
                        if dst_contains and dst_contains not in d_label:     continue
                        if etype_idx is not None:
                            if etypes[i].argmax().item() != etype_idx:       continue

                        atk_x_src_list.append(g.x_src[i].cpu())
                        atk_x_dst_list.append(g.x_dst[i].cpu())
                        atk_edge_type_list.append(etypes[i].cpu())
                        seen.add(uid)
                        collected += 1

        log(f"  pattern[{pat_idx}] ({pattern}): collected {collected} edges")

    if not atk_x_src_list:
        raise ValueError(
            "predict_edge_supervised (patterns mode): no edges matched any attack_patterns. "
            "Check that pattern fields match the node types and labels in this dataset."
        )

    log(f"predict_edge_supervised: {len(atk_x_src_list)} attack edges collected from {len(attack_patterns)} patterns")
    return torch.stack(atk_x_src_list), torch.stack(atk_x_dst_list), torch.stack(atk_edge_type_list)


def _load_attack_features_from_synthetic(cfg, attack_edges_path):
    """Build attack edge feature tensors from pre-computed synthetic node embeddings.

    During feat_inference, feat_inference_spider embeds the synthetic attack node
    labels (which may not exist in the real dataset) and writes them to
    {model_dir}/synthetic_attack_node_embeddings.pt as {(ntype, label): np.ndarray}.

    This function loads that file and constructs x_src, x_dst, edge_type tensors
    with the same format as batch.x_src / batch.x_dst / batch.edge_type, so that
    PredictEdgeSupervised can use them directly for 1:1 oversampling training.
    """
    import yaml
    from pidsmaker.config.pipeline import ROOT_PROJECT_PATH
    from pidsmaker.utils.utils import gen_relation_onehot, log

    if not os.path.isabs(attack_edges_path):
        attack_edges_path = os.path.join(ROOT_PROJECT_PATH, attack_edges_path)

    with open(attack_edges_path) as f:
        data = yaml.safe_load(f)
    attack_edges = data.get("attack_edges", [])

    synth_emb_path = os.path.join(
        cfg.featurization.feat_training._model_dir, "synthetic_attack_node_embeddings.pt"
    )
    if not os.path.exists(synth_emb_path):
        raise FileNotFoundError(
            f"predict_edge_supervised (synthetic mode): embeddings file not found: {synth_emb_path}\n"
            "Re-run feat_inference so that synthetic attack node embeddings are computed first."
        )
    synth_embs = torch.load(synth_emb_path, map_location="cpu")  # {(ntype, label): np.ndarray}

    ntype2oh = gen_relation_onehot(get_node_map())    # {type_str -> 1-D LongTensor}
    etype2oh = gen_relation_onehot(get_rel2id(cfg))   # {event_str -> 1-D LongTensor}

    node_feats_str = cfg.detection.graph_preprocessing.node_features
    selected_node_feats = [f.strip() for f in node_feats_str.replace("-", ",").split(",")]

    atk_x_src_list, atk_x_dst_list, atk_edge_type_list = [], [], []

    for edge in attack_edges:
        src_type  = edge["src_type"]
        src_label = edge["src_label"]
        etype_str = edge["edge_type"]
        dst_type  = edge["dst_type"]
        dst_label = edge["dst_label"]

        src_key = (src_type, src_label)
        dst_key = (dst_type, dst_label)

        if src_key not in synth_embs:
            raise KeyError(
                f"predict_edge_supervised (synthetic mode): no embedding for src node {src_key}. "
                "Re-run feat_inference."
            )
        if dst_key not in synth_embs:
            raise KeyError(
                f"predict_edge_supervised (synthetic mode): no embedding for dst node {dst_key}. "
                "Re-run feat_inference."
            )

        src_emb = torch.from_numpy(synth_embs[src_key]).float()
        dst_emb = torch.from_numpy(synth_embs[dst_key]).float()

        x_src_parts, x_dst_parts = [], []
        for feat in selected_node_feats:
            if feat == "node_emb":
                x_src_parts.append(src_emb)
                x_dst_parts.append(dst_emb)
            elif feat == "node_type":
                x_src_parts.append(ntype2oh[src_type].float())
                x_dst_parts.append(ntype2oh[dst_type].float())
            elif feat in ("only_ones", "edges_distribution"):
                raise ValueError(
                    f"predict_edge_supervised (synthetic mode): node feature '{feat}' "
                    "is not supported for synthetic edges."
                )

        atk_x_src_list.append(torch.cat(x_src_parts))
        atk_x_dst_list.append(torch.cat(x_dst_parts))
        atk_edge_type_list.append(etype2oh[etype_str].float())

    log(f"predict_edge_supervised: {len(atk_x_src_list)} synthetic attack edges loaded from '{attack_edges_path}'")
    return torch.stack(atk_x_src_list), torch.stack(atk_x_dst_list), torch.stack(atk_edge_type_list)


def objective_factory(cfg, in_dim, graph_reindexer, device, objective_cfg=None, all_data=None):
    if objective_cfg is None:
        objective_cfg = cfg.detection.gnn_training.decoder
    node_out_dim = cfg.detection.gnn_training.node_out_dim
    node_hid_dim = cfg.detection.gnn_training.node_hid_dim

    entity_map = get_node_map(from_zero=True)
    event_map = get_rel2id(cfg, from_zero=True)

    objectives = []
    for objective in map(lambda x: x.strip(), objective_cfg.used_methods.split(",")):
        method = getattr(getattr(objective_cfg, objective.strip()), "decoder")

        if not decoder_matches_objective(decoder=method, objective=objective):
            raise ValueError(f"Decoder {method} doesn't match with objective {objective}")

        if objective == "reconstruct_node_features":
            loss_fn = recon_loss_fn_factory(objective_cfg.reconstruct_node_features.loss)

            decoder = decoder_factory(
                method, objective, cfg, in_dim=node_out_dim, out_dim=in_dim, device=device
            )
            objectives.append(NodeFeatReconstruction(decoder=decoder, loss_fn=loss_fn))

        elif objective == "reconstruct_node_embeddings":
            loss_fn = recon_loss_fn_factory(objective_cfg.reconstruct_node_embeddings.loss)

            decoder = decoder_factory(
                method, objective, cfg, in_dim=node_out_dim, out_dim=node_out_dim, device=device
            )
            objectives.append(NodeEmbReconstruction(decoder=decoder, loss_fn=loss_fn))

        elif objective == "reconstruct_edge_embeddings":
            loss_fn = recon_loss_fn_factory(objective_cfg.reconstruct_edge_embeddings.loss)

            decoder = decoder_factory(
                method, objective, cfg, in_dim=node_out_dim, out_dim=node_hid_dim * 2, device=device
            )
            objectives.append(
                EdgeEmbReconstruction(
                    decoder=decoder,
                    loss_fn=loss_fn,
                )
            )

        elif objective == "predict_edge_type":
            loss = objective_cfg.predict_edge_type.loss
            version = objective_cfg.predict_edge_type.AMS.version
            margin = objective_cfg.predict_edge_type.AMS.margin
            scale = objective_cfg.predict_edge_type.AMS.scale
            
            num_edge_types = get_num_edge_type(cfg)
            decoder_out_dim = node_out_dim if loss =="AMS" else num_edge_types
            
            multi_edge = False
            loss_fn = categorical_loss_fn_factory("BCE") if multi_edge else \
                special_categorical_loss_fn_factory(
                    loss, out_dim=decoder_out_dim, num_classes=num_edge_types, version=version, margin=margin, scale=scale)
            
            balanced_loss = objective_cfg.predict_edge_type.balanced_loss
            decoder = decoder_factory(
                method, objective, cfg, in_dim=node_out_dim, out_dim=decoder_out_dim, device=device
            )
            objectives.append(
                EdgeTypePrediction(
                    decoder=decoder,
                    loss_fn=loss_fn,
                    balanced_loss=balanced_loss,
                    edge_type_dim=num_edge_types,
                    multi_edge=multi_edge,
                )
            )

        elif objective == "predict_node_type":
            loss_fn = categorical_loss_fn_factory("cross_entropy")
            balanced_loss = objective_cfg.predict_node_type.balanced_loss

            decoder = decoder_factory(
                method, objective, cfg, in_dim=node_out_dim, out_dim=node_out_dim, device=device
            )
            objectives.append(
                NodeTypePrediction(
                    decoder=decoder,
                    loss_fn=loss_fn,
                    balanced_loss=balanced_loss,
                    node_type_dim=cfg.dataset.num_node_types,
                )
            )

        elif objective == "reconstruct_masked_features":
            mask_rate = objective_cfg.reconstruct_masked_features.mask_rate

            loss_fn = recon_loss_fn_factory(objective_cfg.reconstruct_masked_features.loss)

            decoder = decoder_factory(
                method, objective, cfg, in_dim=node_out_dim, out_dim=in_dim, device=device
            )
            objectives.append(
                GMAEFeatReconstruction(
                    decoder=decoder,
                    loss_fn=loss_fn,
                    mask_rate=mask_rate,
                )
            )

        elif objective == "predict_masked_struct":
            loss_fn = categorical_loss_fn_factory(objective_cfg.predict_masked_struct.loss)

            decoder = decoder_factory(
                method, objective, cfg, in_dim=node_out_dim * 2, out_dim=1, device=device
            )
            objectives.append(
                GMAEStructPrediction(
                    decoder=decoder,
                    loss_fn=loss_fn,
                )
            )

        elif objective == "detect_edge_few_shot":
            classes = 2
            decoder = decoder_factory(
                method,
                objective,
                cfg,
                in_dim=node_out_dim,
                out_dim=classes,
                device=device,
                objective_cfg=objective_cfg,
            )

            objectives.append(
                FewShotEdgeDetection(
                    decoder=decoder,
                    loss_fn=categorical_loss_fn_factory("cross_entropy"),
                )
            )

        elif objective == "predict_edge_contrastive":
            predict_edge_method = objective_cfg.predict_edge_contrastive.decoder.strip()

            if predict_edge_method == "inner_product":
                edge_decoder = EdgeInnerProductDecoder(
                    dropout=objective_cfg.predict_edge_contrastive.inner_product.dropout,
                )

            else:
                edge_decoder = decoder_factory(
                    method,
                    objective,
                    cfg,
                    in_dim=node_out_dim,
                    out_dim=1,
                    device=device,
                    objective_cfg=objective_cfg,
                )

            loss_fn = bce_contrastive

            objectives.append(
                EdgeContrastivePrediction(
                    decoder=edge_decoder,
                    loss_fn=loss_fn,
                    graph_reindexer=graph_reindexer,
                )
            )

        elif objective == "predict_edge_supervised":
            obj_cfg = objective_cfg.predict_edge_supervised

            mode = obj_cfg.mode.strip()
            if mode == "synthetic":
                atk_x_src, atk_x_dst, atk_edge_type = _load_attack_features_from_synthetic(
                    cfg,
                    attack_edges_path=obj_cfg.attack_edges_path,
                )
            else:
                if all_data is None:
                    raise ValueError(
                        f"predict_edge_supervised (mode='{mode}') requires val/test data. "
                        "Pass all_data=(val_data, test_data) to build_model()."
                    )
                if mode == "scores":
                    atk_x_src, atk_x_dst, atk_edge_type = _load_attack_features(
                        cfg,
                        edge_scores_path=obj_cfg.edge_scores_path,
                        top_n=obj_cfg.top_n_attacks,
                        all_data=all_data,
                    )
                elif mode == "patterns":
                    atk_x_src, atk_x_dst, atk_edge_type = _load_attack_features_from_patterns(
                        cfg,
                        attack_patterns=obj_cfg.attack_patterns,
                        max_edges_per_pattern=obj_cfg.max_edges_per_pattern,
                        all_data=all_data,
                    )
                else:
                    raise ValueError(
                        f"predict_edge_supervised: unknown mode '{mode}'. Use 'scores', 'patterns', or 'synthetic'."
                    )

            # Decoder input = [x_src | edge_type] and [x_dst | edge_type]
            edge_type_dim = get_num_edge_type(cfg)
            supervised_in_dim = in_dim + edge_type_dim
            decoder = decoder_factory(
                method, objective, cfg, in_dim=supervised_in_dim, out_dim=1, device=device
            )
            objectives.append(
                PredictEdgeSupervised(
                    decoder=decoder,
                    atk_x_src=atk_x_src.to(device),
                    atk_x_dst=atk_x_dst.to(device),
                    atk_edge_type=atk_edge_type.to(device),
                    pos_weight=obj_cfg.pos_weight,
                )
            )

        else:
            raise ValueError(f"Invalid objective {objective}")

    # We wrap objectives into this class to calculate some metrics on validation set easily
    is_edge_type_prediction = objective_cfg.used_methods.strip() == "predict_edge_type"
    objectives = [
        ValidationWrapper(
            objective,
            graph_reindexer,
            is_edge_type_prediction,
            use_few_shot=cfg.detection.gnn_training.decoder.use_few_shot,
        )
        for objective in objectives
    ]

    return objectives


def few_shot_decoder_factory(cfg, graph_reindexer, device, objective_cfg=None):
    if not cfg.detection.gnn_training.decoder.use_few_shot:
        return None

    node_out_dim = cfg.detection.gnn_training.node_out_dim
    objective_cfg = cfg.detection.gnn_training.decoder.few_shot.decoder

    objective = objective_factory(
        cfg,
        in_dim=node_out_dim,
        graph_reindexer=graph_reindexer,
        device=device,
        objective_cfg=objective_cfg,
    )
    return nn.ModuleList(objective)


def edge_decoder_factory(edge_decoder, in_dim):
    if edge_decoder == "MLP":
        return nn.Sequential(
            nn.Linear(in_dim, in_dim * 2),
            nn.ReLU(),
            nn.Linear(in_dim * 2, in_dim),
        )
    elif edge_decoder == "none":
        return None

    raise ValueError(f"Invalid edge decoder {edge_decoder}")


def recon_loss_fn_factory(loss: str):
    if loss == "SCE":
        return sce_loss
    if loss == "MSE":
        return mse_loss
    if loss == "MSE_sum":
        return mse_loss_sum
    if loss == "MAE":
        return mae_loss
    if loss == "none":
        return nn.Identity()
    raise ValueError(f"Invalid loss function {loss}")


def categorical_loss_fn_factory(loss: str):
    if loss == "cross_entropy":
        return cross_entropy
    if loss == "BCE":
        return binary_cross_entropy
    raise ValueError(f"Invalid loss function {loss}")

def special_categorical_loss_fn_factory(loss: str, out_dim: int, num_classes: int, version, margin, scale):
    if loss == "AMS":
        if version == 1:
            return AdMSoftmaxLoss(emb_dim=out_dim, num_classes=num_classes, margin=margin, scale=scale)
        elif version == 2:
            return AMSoftmax(emb_dim=out_dim, num_classes=num_classes, margin=margin, scale=scale)
        raise ValueError(f"Invalid version {version}")
    
    return categorical_loss_fn_factory(loss)

def activation_fn_factory(activation: str):
    if activation == "sigmoid":
        return nn.Sigmoid()
    if activation == "relu":
        return nn.ReLU()
    if activation == "tanh":
        return nn.Tanh()
    if activation == "prelu":
        return nn.PReLU()
    if activation == "none":
        return nn.Identity()
    raise ValueError(f"Invalid activation function {activation}")


def optimizer_factory(cfg, parameters):
    lr = cfg.detection.gnn_training.lr
    weight_decay = cfg.detection.gnn_training.weight_decay
    stable = cfg.detection.gnn_training.stable_optim

    if stable:
        return torch.optim.AdamW(
            parameters, lr=lr, betas=(0.9, 0.99), eps=1e-10,
            weight_decay=0.02,
        )
    return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)


def optimizer_few_shot_factory(cfg, parameters):
    lr = cfg.detection.gnn_training.decoder.few_shot.lr_few_shot
    weight_decay = cfg.detection.gnn_training.decoder.few_shot.weight_decay_few_shot

    return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)


def get_dimensions_from_data_sample(data):
    edge_dim = data.edge_feats.shape[1] if hasattr(data, "edge_feats") else None
    msg_dim = data.msg.shape[1] if hasattr(data, "msg") else edge_dim
    in_dim = data.x_src.shape[1] if hasattr(data, "x_src") else data.x.shape[1]

    return msg_dim, edge_dim, in_dim


def get_edge_dim(cfg, msg_dim):
    edge_dim = 0
    edge_features = list(
        map(lambda x: x.strip(), cfg.detection.graph_preprocessing.edge_features.split(","))
    )
    use_tgn = "tgn" in cfg.detection.gnn_training.encoder.used_methods
    tgn_memory_dim = cfg.detection.gnn_training.encoder.tgn.tgn_memory_dim

    for edge_feat in edge_features:
        if edge_feat in ["edge_type", "edge_type_triplet"]:
            edge_dim += get_num_edge_type(cfg)
        elif edge_feat == "msg":
            edge_dim += msg_dim
        elif edge_feat == "time_encoding":
            if not use_tgn:
                raise TypeError("Edge feature `time_encoding` is only available if TGN is used.")
            edge_dim += tgn_memory_dim
        elif edge_feat == "none":
            pass
        else:
            raise ValueError(f"Invalid edge feature {edge_feat}")

    return edge_dim
