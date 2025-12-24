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
)


def build_model(data_sample, device, cfg, max_node_num):
    """
    Builds and loads the initial model into memory.
    The `data_sample` is required to infer the shape of the layers.
    """
    msg_dim, edge_dim, in_dim = get_dimensions_from_data_sample(data_sample)

    graph_reindexer = GraphReindexer(
        device=device,
        num_nodes=max_node_num,
        fix_buggy_graph_reindexer=cfg.graph_preprocessing.fix_buggy_graph_reindexer,
    )

    encoder = encoder_factory(
        cfg,
        msg_dim=msg_dim,
        in_dim=in_dim,
        device=device,
        max_node_num=max_node_num,
        graph_reindexer=graph_reindexer,
    )
    objectives = objective_factory(
        cfg, in_dim=in_dim, graph_reindexer=graph_reindexer, device=device
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
        use_few_shot=cfg.gnn_training.decoder.use_few_shot,
        freeze_encoder=cfg.gnn_training.decoder.few_shot.freeze_encoder,
    ).to(device)


def encoder_factory(cfg, msg_dim, in_dim, device, max_node_num, graph_reindexer):
    node_hid_dim = cfg.gnn_training.node_hid_dim
    node_out_dim = cfg.gnn_training.node_out_dim
    tgn_memory_dim = cfg.gnn_training.encoder.tgn.tgn_memory_dim
    use_tgn = "tgn" in cfg.gnn_training.encoder.used_methods
    dropout = cfg.gnn_training.encoder.dropout

    node_map = get_node_map(from_zero=True)
    edge_map = get_rel2id(cfg, from_zero=True)

    edge_dim = get_edge_dim(cfg, msg_dim)

    original_in_dim = in_dim
    if use_tgn:
        in_dim = tgn_memory_dim

    for method in map(
        lambda x: x.strip(),
        cfg.gnn_training.encoder.used_methods.replace("-", ",").split(","),
    ):
        if method in ["tgn"]:
            pass

        # Basic GNN encoders
        elif method == "graph_attention":
            encoder = GraphAttentionEmbedding(
                in_dim=in_dim,
                hid_dim=node_hid_dim,
                out_dim=node_out_dim,
                edge_dim=edge_dim or None,
                activation=activation_fn_factory(
                    cfg.gnn_training.encoder.graph_attention.activation
                ),
                dropout=dropout,
                num_heads=cfg.gnn_training.encoder.graph_attention.num_heads,
                concat=cfg.gnn_training.encoder.graph_attention.concat,
                flow=cfg.gnn_training.encoder.graph_attention.flow,
                num_layers=cfg.gnn_training.encoder.graph_attention.num_layers,
            )
        elif method == "sage":
            encoder = SAGE(
                in_dim=in_dim,
                hid_dim=node_hid_dim,
                out_dim=node_out_dim,
                activation=activation_fn_factory(
                    cfg.gnn_training.encoder.sage.activation
                ),
                dropout=dropout,
                num_layers=cfg.gnn_training.encoder.sage.num_layers,
            )
        elif method == "gat":
            encoder = GAT(
                in_dim=in_dim,
                hid_dim=node_hid_dim,
                out_dim=node_out_dim,
                activation=activation_fn_factory(cfg.gnn_training.encoder.gat.activation),
                dropout=dropout,
                num_heads=cfg.gnn_training.encoder.gat.num_heads,
                concat=cfg.gnn_training.encoder.gat.concat,
                num_layers=cfg.gnn_training.encoder.gat.num_layers,
            )
        elif method == "gin":
            encoder = GIN(
                in_dim=in_dim,
                hid_dim=node_hid_dim,
                out_dim=node_out_dim,
                edge_dim=edge_dim or None,
                dropout=dropout,
                activation=activation_fn_factory(cfg.gnn_training.encoder.gin.activation),
                num_layers=cfg.gnn_training.encoder.gin.num_layers,
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
        elif method == "magic_gat":
            n_layers = cfg.gnn_training.encoder.magic_gat.num_layers
            n_heads = cfg.gnn_training.encoder.magic_gat.num_heads
            negative_slope = cfg.gnn_training.encoder.magic_gat.negative_slope
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
                    cfg.gnn_training.encoder.magic_gat.activation
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
                architecture=cfg.gnn_training.encoder.custom_mlp.architecture_str,
                dropout=dropout,
            )
        else:
            raise ValueError(f"Invalid encoder {method}")

    if use_tgn:
        tgn_cfg = cfg.gnn_training.encoder.tgn
        time_dim = tgn_cfg.tgn_time_dim
        use_node_feats_in_gnn = tgn_cfg.use_node_feats_in_gnn
        use_memory = tgn_cfg.use_memory
        use_time_order_encoding = tgn_cfg.use_time_order_encoding
        project_src_dst = tgn_cfg.project_src_dst
        edge_features = list(
            map(lambda x: x.strip(), cfg.graph_preprocessing.edge_features.split(","))
        )

        use_time_enc = "time_encoding" in cfg.graph_preprocessing.edge_features

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
        else:
            memory = None

        encoder = TGNEncoder(
            encoder=encoder,
            memory=memory,
            time_encoder=memory.time_enc if memory else None,
            in_dim=original_in_dim,
            memory_dim=tgn_memory_dim,
            use_node_feats_in_gnn=use_node_feats_in_gnn,
            edge_features=edge_features,
            device=device,
            use_memory=use_memory,
            use_time_enc=use_time_enc,
            edge_dim=edge_dim,
            use_time_order_encoding=use_time_order_encoding,
            project_src_dst=project_src_dst,
            node_map=node_map,
            edge_map=edge_map,
        )

    return encoder


def decoder_factory(method, objective, cfg, in_dim, out_dim, device, objective_cfg=None):
    if objective_cfg is None:
        objective_cfg = cfg.gnn_training.decoder
    decoder_cfg = getattr(getattr(objective_cfg, objective), method, None)

    if method == "edge_mlp":
        return CustomEdgeMLP(
            in_dim=in_dim,
            out_dim=out_dim,
            architecture=decoder_cfg.architecture_str,
            dropout=cfg.gnn_training.encoder.dropout,
            src_dst_projection_coef=decoder_cfg.src_dst_projection_coef,
        )
    elif method == "node_mlp":
        return CustomMLPDecoder(
            in_dim=in_dim,
            out_dim=out_dim,
            architecture=decoder_cfg.architecture_str,
            dropout=cfg.gnn_training.encoder.dropout,
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
                cfg.gnn_training.encoder.magic_gat.activation
            ),
            is_decoder=True,
        )
    elif method == "none":
        return lambda x: x
    else:
        raise ValueError(f"Invalid decoder {method}")


def objective_factory(cfg, in_dim, graph_reindexer, device, objective_cfg=None):
    if objective_cfg is None:
        objective_cfg = cfg.gnn_training.decoder
    node_out_dim = cfg.gnn_training.node_out_dim

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
                method, objective, cfg, in_dim=node_out_dim, out_dim=node_out_dim * 2, device=device
            )
            objectives.append(
                EdgeEmbReconstruction(
                    decoder=decoder,
                    loss_fn=loss_fn,
                )
            )

        elif objective == "predict_edge_type":
            loss_fn = categorical_loss_fn_factory("cross_entropy")
            balanced_loss = objective_cfg.predict_edge_type.balanced_loss

            num_edge_types = get_num_edge_type(cfg)

            decoder = decoder_factory(
                method, objective, cfg, in_dim=node_out_dim, out_dim=num_edge_types, device=device
            )
            objectives.append(
                EdgeTypePrediction(
                    decoder=decoder,
                    loss_fn=loss_fn,
                    balanced_loss=balanced_loss,
                    edge_type_dim=num_edge_types,
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

        else:
            raise ValueError(f"Invalid objective {objective}")

    # We wrap objectives into this class to calculate some metrics on validation set easily
    # This is useful only if use_few_shot is True
    is_edge_type_prediction = objective_cfg.used_methods.strip() == "predict_edge_type"
    objectives = [
        ValidationWrapper(
            objective,
            graph_reindexer,
            is_edge_type_prediction,
            use_few_shot=cfg.gnn_training.decoder.use_few_shot,
        )
        for objective in objectives
    ]

    return objectives


def few_shot_decoder_factory(cfg, graph_reindexer, device, objective_cfg=None):
    if not cfg.gnn_training.decoder.use_few_shot:
        return None

    node_out_dim = cfg.gnn_training.node_out_dim
    objective_cfg = cfg.gnn_training.decoder.few_shot.decoder

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
    lr = cfg.gnn_training.lr
    weight_decay = cfg.gnn_training.weight_decay

    return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)


def optimizer_few_shot_factory(cfg, parameters):
    lr = cfg.gnn_training.decoder.few_shot.lr_few_shot
    weight_decay = cfg.gnn_training.decoder.few_shot.weight_decay_few_shot

    return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)


def get_dimensions_from_data_sample(data):
    edge_dim = data.edge_feats.shape[1] if hasattr(data, "edge_feats") else None
    msg_dim = data.msg.shape[1] if hasattr(data, "msg") else edge_dim
    in_dim = data.x_src.shape[1] if hasattr(data, "x_src") else data.x.shape[1]

    return msg_dim, edge_dim, in_dim


def get_edge_dim(cfg, msg_dim):
    edge_dim = 0
    edge_features = list(
        map(lambda x: x.strip(), cfg.graph_preprocessing.edge_features.split(","))
    )
    use_tgn = "tgn" in cfg.gnn_training.encoder.used_methods
    tgn_memory_dim = cfg.gnn_training.encoder.tgn.tgn_memory_dim

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
