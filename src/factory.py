import torch
import torch.nn as nn

from provnet_utils import *
from config import *
from config import ntype2id, get_rel2id, get_node_map, OPTC_DATASETS, possible_events, get_num_edge_type
from model import *
from losses import *
from encoders import *
from decoders import *
from data_utils import *
from tgn import TGNMemory, TimeEncodingMemory, LastAggregator, LastNeighborLoader, IdentityMessage
from experiments.uncertainty import add_dropout_to_model, IdentityWrapper
from hetero import get_metadata


def build_model(data_sample, device, cfg, max_node_num):
    """
    Builds and loads the initial model into memory.
    The `data_sample` is required to infer the shape of the layers.
    """
    msg_dim, edge_dim, in_dim = get_dimensions_from_data_sample(data_sample)

    graph_reindexer = GraphReindexer(
        num_nodes=max_node_num,
        device=device,
        fix_buggy_graph_reindexer=cfg.detection.graph_preprocessing.fix_buggy_graph_reindexer,
    )
    
    encoder = encoder_factory(cfg, msg_dim=msg_dim, in_dim=in_dim, edge_dim=edge_dim, device=device, max_node_num=max_node_num)
    decoder = objective_factory(cfg, in_dim=in_dim, graph_reindexer=graph_reindexer, device=device)
    decoder_few_shot = few_shot_decoder_factory(cfg, device=device, graph_reindexer=graph_reindexer)
    model = model_factory(encoder, decoder, decoder_few_shot, cfg, device=device)
    
    if cfg._is_running_mc_dropout:
        dropout = cfg.experiment.uncertainty.mc_dropout.dropout
        add_dropout_to_model(model, p=dropout)
    
    return model

def model_factory(encoder, decoders, decoder_few_shot, cfg, device):
    return Model(
        encoder=encoder,
        decoders=decoders,
        decoder_few_shot=decoder_few_shot,
        device=device,
        is_running_mc_dropout=cfg._is_running_mc_dropout,
        use_few_shot=cfg.detection.gnn_training.decoder.use_few_shot,
        freeze_encoder=cfg.detection.gnn_training.decoder.few_shot.freeze_encoder,
    ).to(device)

def encoder_factory(cfg, msg_dim, in_dim, edge_dim, device, max_node_num):
    node_hid_dim = cfg.detection.gnn_training.node_hid_dim
    node_out_dim = cfg.detection.gnn_training.node_out_dim
    tgn_memory_dim = cfg.detection.gnn_training.encoder.tgn.tgn_memory_dim
    use_tgn = "tgn" in cfg.detection.gnn_training.encoder.used_methods
    use_ancestor_encoding = "ancestor_encoding" in cfg.detection.gnn_training.encoder.used_methods
    use_entity_type_encoding = "entity_type_encoding" in cfg.detection.gnn_training.encoder.used_methods
    use_event_type_encoding = "event_type_encoding" in cfg.detection.gnn_training.encoder.used_methods
    
    node_map = get_node_map(from_zero=True)
    edge_map = get_rel2id(cfg, from_zero=True)
    
    # If edge features are used, we set them here
    edge_dim = 0
    edge_features = list(map(lambda x: x.strip(), cfg.detection.graph_preprocessing.edge_features.split(",")))
    for edge_feat in edge_features:
        if edge_feat in ["edge_type", "edge_type_triplet"]:
            edge_dim += get_num_edge_type(cfg)
        elif edge_feat == "msg":
            edge_dim += msg_dim
        elif edge_feat == "time_encoding":
            if not use_tgn:
                raise TypeError(f"Edge feature `time_encoding` is only available if TGN is used.")
            edge_dim += tgn_memory_dim
        elif edge_feat == "none":
            pass
        else:
            raise ValueError(f"Invalid edge feature {edge_feat}")

    original_in_dim = in_dim
    if use_tgn:
        in_dim = cfg.detection.gnn_training.encoder.tgn.tgn_memory_dim
        
    original_edge_dim = edge_dim
    if use_event_type_encoding:
        edge_dim = in_dim

    for method in map(lambda x: x.strip(), cfg.detection.gnn_training.encoder.used_methods.replace("-", ",").split(",")):
        if method in ["tgn", "ancestor_encoding", "entity_type_encoding", "event_type_encoding"]:
            pass
        elif method == "graph_attention":
            encoder = GraphAttentionEmbedding(
                in_dim=in_dim,
                hid_dim=node_hid_dim,
                out_dim=node_out_dim,
                edge_dim=edge_dim or None,
                activation=activation_fn_factory(cfg.detection.gnn_training.encoder.graph_attention.activation),
                dropout=cfg.detection.gnn_training.encoder.dropout,
                num_heads=cfg.detection.gnn_training.encoder.graph_attention.num_heads,
                concat=cfg.detection.gnn_training.encoder.graph_attention.concat,
                flow=cfg.detection.gnn_training.encoder.graph_attention.flow,
            )
        elif method == "hetero_graph_transformer":
            if cfg.dataset.name in OPTC_DATASETS:
                raise NotImplementedError(f"Hetero OPTC not implemented (need to compute possible_events)")

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
                activation=activation_fn_factory(cfg.detection.gnn_training.encoder.sage.activation),
                dropout=cfg.detection.gnn_training.encoder.dropout,
            )
        elif method == "LSTM":
            encoder = LSTM(
                 in_features=in_dim,
                 out_features=node_out_dim,
                 cell_clip=None,
                 type_specific_decoding=False,
                 exclude_file=True,
                 exclude_ip=True,
                 typed_hidden_rep=False,
                 edge_dim=None,
                 full_param=False,
                 num_edge_type = 15 # TODO: we should use 10 here
            
            ).to(device)
        elif method == "rcaid_gat":
            encoder = RCaidGAT(
                in_dim=in_dim,
                hid_dim=node_hid_dim,
                out_dim=node_out_dim,
                dropout=cfg.detection.gnn_training.encoder.dropout,
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
                concat_out=True,residual=True,
                activation=activation_fn_factory(cfg.detection.gnn_training.encoder.magic_gat.activation),
            )
        elif method == "GIN":
            encoder = GIN(
                in_dim=in_dim,
                hid_dim=node_hid_dim,
                out_dim=node_out_dim,
                edge_dim=edge_dim or None,
                # activation=activation_fn_factory("relu"),
            )
        elif method == "none":
            encoder = LinearEncoder(in_dim, node_out_dim)
        elif method == "custom_mlp":
            encoder = CustomMLPEncoder(
                in_dim=in_dim,
                out_dim=node_out_dim,
                architecture=cfg.detection.gnn_training.encoder.custom_mlp.architecture_str,
                dropout=cfg.detection.gnn_training.encoder.dropout,
            )
        else:
            raise ValueError(f"Invalid encoder {method}")
            
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
            out_dim=in_dim, # try in_dim*2 ou out_dim
            edge_dim=edge_dim,
            encoder=encoder,
            num_nodes=max_node_num,
            device=device,
        )
        
    if use_tgn:
        tgn_cfg = cfg.detection.gnn_training.encoder.tgn
        time_dim = tgn_cfg.tgn_time_dim
        use_node_feats_in_gnn = tgn_cfg.use_node_feats_in_gnn
        use_memory = tgn_cfg.use_memory
        use_time_order_encoding = tgn_cfg.use_time_order_encoding
        project_src_dst = tgn_cfg.project_src_dst
        
        use_time_enc = "time_encoding" in cfg.detection.graph_preprocessing.edge_features

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
            is_hetero=cfg._is_hetero,
            node_map=node_map,
            edge_map=edge_map,
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
        hid_dim = decoder_cfg.hid_dim

        return MagicGAT(
            n_dim=in_dim,
            hidden_dim=hid_dim,
            out_dim=out_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            feat_drop=0.1,
            attn_drop=0.0,
            negative_slope=negative_slope,
            concat_out=True,residual=True,
            activation=activation_fn_factory(cfg.detection.gnn_training.encoder.magic_gat.activation),
        )
    elif method == "none":
        return lambda x: x
    else:
        raise ValueError(f"Invalid decoder {method}")
        

def objective_factory(cfg, in_dim, graph_reindexer, device, objective_cfg=None):
    if objective_cfg is None:
        objective_cfg = cfg.detection.gnn_training.decoder
    node_out_dim = cfg.detection.gnn_training.node_out_dim
    
    entity_map = get_node_map(from_zero=True)
    event_map = get_rel2id(cfg, from_zero=True)

    objectives = []
    for objective in map(lambda x: x.strip(), objective_cfg.used_methods.split(",")):
        method = getattr(getattr(objective_cfg, objective.strip()), "decoder")

        if objective == "reconstruct_node_features":
            loss_fn = recon_loss_fn_factory(objective_cfg.reconstruct_node_features.loss)

            decoder = decoder_factory(method, objective, cfg, in_dim=node_out_dim, out_dim=in_dim, device=device)
            objectives.append(NodeFeatReconstruction(decoder=decoder, loss_fn=loss_fn))
            
        elif objective == "reconstruct_node_embeddings":
            loss_fn = recon_loss_fn_factory(objective_cfg.reconstruct_node_embeddings.loss)

            decoder = decoder_factory(method, objective, cfg, in_dim=node_out_dim, out_dim=node_out_dim, device=device)
            objectives.append(NodeEmbReconstruction(decoder=decoder, loss_fn=loss_fn))
        
        elif objective == "reconstruct_edge_embeddings":
            loss_fn = recon_loss_fn_factory(objective_cfg.reconstruct_edge_embeddings.loss)
            in_dim_edge = node_out_dim * 2  # concatenation of 2 nodes
            
            decoder = decoder_factory(method, objective, cfg, in_dim=in_dim_edge, out_dim=in_dim_edge, device=device)
            objectives.append(
                EdgeEmbReconstruction(
                    decoder=decoder,
                    loss_fn=loss_fn,
                ))
            
        elif objective == "predict_edge_type":
            loss_fn = categorical_loss_fn_factory("cross_entropy")
            balanced_loss = objective_cfg.predict_edge_type.balanced_loss
            
            num_edge_types = get_num_edge_type(cfg)
            
            decoder = decoder_factory(method, objective, cfg, in_dim=node_out_dim, out_dim=num_edge_types, device=device)
            objectives.append(
                EdgeTypePrediction(
                    decoder=decoder,
                    loss_fn=loss_fn,
                    balanced_loss=balanced_loss,
                    edge_type_dim=num_edge_types,
                ))
            
        elif objective == "predict_edge_type_hetero":
            loss_fn = categorical_loss_fn_factory("cross_entropy")
            decoder = decoder_factory(method, objective, cfg, in_dim=node_out_dim, out_dim=node_out_dim, device=device)
            
            decoder_hetero_head_method = getattr(getattr(objective_cfg, objective.strip()), "decoder_hetero_head")
            
            edge_type_predictors = nn.ModuleDict()
        
            ntype2edgemap = {}
            for (src_type, dst_type), events in possible_events.items():
                src_idx = entity_map[src_type]
                dst_idx = entity_map[dst_type]
                events = torch.tensor([event_map[e] for e in events])
                max_event = len(event_map) // 2 # event maps contain num=>label and label=>num entries so we /2
                
                reindexed_events = torch.zeros((max_event,), dtype=torch.long, device=device)
                reindexed_events[events] = torch.arange(events.size(0), device=device)
                ntype2edgemap[(src_idx, dst_idx)] = reindexed_events
                
                layer_name = f"{src_idx}_{dst_idx}"
                num_events = torch.unique(events).numel()
                
                decoder_hetero_head = decoder_factory(decoder_hetero_head_method, objective, cfg, in_dim=node_out_dim, out_dim=num_events, device=device)
                edge_type_predictors[layer_name] = decoder_hetero_head
            
            objectives.append(
                EdgeTypePredictionHetero(
                    decoder=decoder,
                    loss_fn=loss_fn,
                    edge_type_predictors=edge_type_predictors,
                    ntype2edgemap=ntype2edgemap,
                ))
            
        elif objective == "predict_node_type":
            loss_fn = categorical_loss_fn_factory("cross_entropy")
            balanced_loss = objective_cfg.predict_node_type.balanced_loss
            
            decoder = decoder_factory(method, objective, cfg, in_dim=node_out_dim, out_dim=node_out_dim, device=device)
            objectives.append(
                NodeTypePrediction(
                    decoder=decoder,
                    loss_fn=loss_fn,
                    balanced_loss=balanced_loss,
                    node_type_dim=cfg.dataset.num_node_types,
                ))
            
        elif objective == "reconstruct_masked_features":
            mask_rate = objective_cfg.reconstruct_masked_features.mask_rate

            loss_fn = recon_loss_fn_factory(objective_cfg.reconstruct_masked_features.loss)

            decoder = decoder_factory(method, objective, cfg, in_dim=node_out_dim, out_dim=in_dim, device=device)
            objectives.append(
                GMAEFeatReconstruction(
                    decoder=decoder,
                    loss_fn=loss_fn,
                    mask_rate=mask_rate,
                )
            )
        
        elif objective == "predict_masked_struct":
            loss_fn = categorical_loss_fn_factory(objective_cfg.predict_masked_struct.loss)

            decoder = decoder_factory(method, objective, cfg, in_dim=node_out_dim * 2, out_dim=1, device=device)
            objectives.append(
                GMAEStructPrediction(
                    decoder=decoder,
                    loss_fn=loss_fn,
                )
            )
            
        elif objective == "detect_edge_few_shot":
            classes = 2
            decoder = decoder_factory(method, objective, cfg, in_dim=node_out_dim, out_dim=classes, device=device, objective_cfg=objective_cfg)
            
            objectives.append(
                FewShotEdgeDetection(
                    decoder=decoder,
                    loss_fn=categorical_loss_fn_factory("cross_entropy"),
                )
            )
        
        elif objective == "predict_edge_contrastive":
            predict_edge_method = objective_cfg.predict_edge_contrastive.decoder.strip()
            if predict_edge_method == "linear":
                edge_decoder = EdgeLinearDecoder(
                    in_dim=node_out_dim,
                    dropout=objective_cfg.predict_edge_contrastive.linear.dropout,
                )
            elif predict_edge_method == "inner_product":
                edge_decoder = EdgeInnerProductDecoder(
                    dropout=objective_cfg.predict_edge_contrastive.inner_product.dropout,
                )
            else:
                raise ValueError(f"Invalid edge decoding method {predict_edge_method}")
            
            loss_fn = bce_contrastive
            
            objectives.append(EdgeContrastiveDecoder(
                decoder=edge_decoder,
                loss_fn=loss_fn,
                graph_reindexer=graph_reindexer,
            ))
        
        else:
            raise ValueError(f"Invalid objective {objective}")
        
    # We wrap objectives into this class to calculate some metrics on validation set easily
    is_edge_type_prediction = objective_cfg.used_methods.strip() == "predict_edge_type"
    objectives = [
        ValidationContrastiveStopper(
            objective,
            graph_reindexer,
            is_edge_type_prediction,
            use_few_shot=cfg.detection.gnn_training.decoder.use_few_shot,
        ) for objective in objectives]
    
    return objectives

def few_shot_decoder_factory(cfg, graph_reindexer, device, objective_cfg=None):
    if not cfg.detection.gnn_training.decoder.use_few_shot:
        return None

    node_out_dim = cfg.detection.gnn_training.node_out_dim
    objective_cfg = cfg.detection.gnn_training.decoder.few_shot.decoder
    
    objective = objective_factory(cfg, in_dim=node_out_dim, graph_reindexer=graph_reindexer, device=device, objective_cfg=objective_cfg)
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
    lr = cfg.detection.gnn_training.lr
    weight_decay = cfg.detection.gnn_training.weight_decay

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
