import torch
import torch.nn as nn
from torch_geometric.loader import NeighborLoader

from provnet_utils import *
from config import *
from model import *
from losses import *
from encoders import *
from decoders import *
from data_utils import *
from tgn import TGNMemory, TimeEncodingMemory, LastAggregator, LastNeighborLoader, IdentityMessage
from experiments.uncertainty import add_dropout_to_model


def build_model(data_sample, device, cfg, max_node_num):
    """
    Builds and loads the initial model into memory.
    The `data_sample` is required to infer the shape of the layers.
    """
    msg_dim, edge_dim, in_dim = get_dimensions_from_data_sample(data_sample)

    graph_reindexer = GraphReindexer(
        num_nodes=max_node_num,
        device=device,
    )
    
    encoder = encoder_factory(cfg, msg_dim=msg_dim, in_dim=in_dim, edge_dim=edge_dim, graph_reindexer=graph_reindexer, device=device, max_node_num=max_node_num)
    decoder = objective_factory(cfg, in_dim=in_dim, device=device, max_node_num=max_node_num)
    model = model_factory(encoder, decoder, cfg, in_dim=in_dim, graph_reindexer=graph_reindexer, device=device, max_node_num=max_node_num)
    
    if cfg._is_running_mc_dropout:
        dropout = cfg.experiment.uncertainty.mc_dropout.dropout
        add_dropout_to_model(model, p=dropout)
    
    return model

def model_factory(encoder, decoders, cfg, in_dim, graph_reindexer, device, max_node_num):
    return Model(
        encoder=encoder,
        decoders=decoders,
        num_nodes=max_node_num,
        device=device,
        in_dim=in_dim,
        out_dim=cfg.detection.gnn_training.node_out_dim,
        use_contrastive_learning="predict_edge_contrastive" in cfg.detection.gnn_training.decoder.used_methods,
        graph_reindexer=graph_reindexer,
        node_level=cfg._is_node_level,
        is_running_mc_dropout=cfg._is_running_mc_dropout,
    ).to(device)

def encoder_factory(cfg, msg_dim, in_dim, edge_dim, graph_reindexer, device, max_node_num):
    node_hid_dim = cfg.detection.gnn_training.node_hid_dim
    node_out_dim = cfg.detection.gnn_training.node_out_dim
    tgn_memory_dim = cfg.detection.gnn_training.encoder.tgn.tgn_memory_dim
    use_tgn = "tgn" in cfg.detection.gnn_training.encoder.used_methods
    
    # If edge features are used, we set them here
    edge_dim = 0
    edge_features = list(map(lambda x: x.strip(), cfg.detection.gnn_training.encoder.edge_features.split(",")))
    for edge_feat in edge_features:
        if edge_feat == "edge_type":
            edge_dim += cfg.dataset.num_edge_types
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

    if use_tgn:
        # Only for TGN, in_dim becomes memory dim
        original_in_dim = in_dim
        in_dim = cfg.detection.gnn_training.encoder.tgn.tgn_memory_dim

    for method in map(lambda x: x.strip(), cfg.detection.gnn_training.encoder.used_methods.split(",")):
        if method == "tgn":
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
                 in_features = in_dim,
                 out_features = node_out_dim,
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
            hid_dim = cfg.detection.gnn_training.encoder.magic_gat.hid_dim
            assert hid_dim % n_heads == 0, "Invalid shape dim for number of heads"

            return MagicGAT(
                n_dim=in_dim,
                hidden_dim=hid_dim,
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
            return lambda x: x
        else:
            raise ValueError(f"Invalid encoder {method}")
    
    if use_tgn:
        time_dim = cfg.detection.gnn_training.encoder.tgn.tgn_time_dim
        neighbor_size = cfg.detection.gnn_training.encoder.tgn.tgn_neighbor_size
        use_node_feats_in_gnn = cfg.detection.gnn_training.encoder.tgn.use_node_feats_in_gnn
        use_memory = cfg.detection.gnn_training.encoder.tgn.use_memory
        use_time_enc = "time_encoding" in cfg.detection.gnn_training.encoder.edge_features
        use_time_order_encoding = cfg.detection.gnn_training.encoder.tgn.use_time_order_encoding
        tgn_neighbor_n_hop = cfg.detection.gnn_training.encoder.tgn.tgn_neighbor_n_hop

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

        neighbor_loader = LastNeighborLoader(max_node_num, size=neighbor_size, device=device)

        encoder = TGNEncoder(
            encoder=encoder,
            memory=memory,
            neighbor_loader=neighbor_loader,
            time_encoder=memory.time_enc if memory else None,
            in_dim=original_in_dim,
            memory_dim=tgn_memory_dim,
            use_node_feats_in_gnn=use_node_feats_in_gnn,
            graph_reindexer=graph_reindexer,
            edge_features=edge_features,
            device=device,
            use_memory=use_memory,
            num_nodes=max_node_num,
            use_time_enc=use_time_enc,
            edge_dim=edge_dim,
            use_time_order_encoding=use_time_order_encoding,
            tgn_neighbor_n_hop=tgn_neighbor_n_hop,
        )

    return encoder

def decoder_factory(method, objective, cfg, in_dim, out_dim):
    decoder_cfg = getattr(getattr(cfg.detection.gnn_training.decoder, objective), method, None)
    
    if method == "edge_mlp":
        return EdgeMLPDecoder(
            in_dim=in_dim,
            out_dim=out_dim,
            architecture=decoder_cfg["architecture_str"],
        )
    elif method == "node_mlp":
        return NodeMLPDecoder(
            in_dim=in_dim,
            out_dim=out_dim,
            architecture=decoder_cfg["architecture_str"],
        )
    elif method == "nodlink":
        return NodLinkDecoder(
            in_dim=in_dim,
            out_dim=out_dim,
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
        

def objective_factory(cfg, in_dim, device, max_node_num):
    node_out_dim = cfg.detection.gnn_training.node_out_dim

    objectives = []
    for objective in map(lambda x: x.strip(), cfg.detection.gnn_training.decoder.used_methods.split(",")):
        method = getattr(getattr(cfg.detection.gnn_training.decoder, objective.strip()), "decoder")

        if objective == "reconstruct_node_features":
            loss_fn = recon_loss_fn_factory(cfg.detection.gnn_training.decoder.reconstruct_node_features.loss)

            decoder = decoder_factory(method, objective, cfg, in_dim=node_out_dim, out_dim=in_dim)
            objectives.append(NodeFeatReconstruction(decoder=decoder, loss_fn=loss_fn))
            
        elif objective == "reconstruct_node_embeddings":
            loss_fn = recon_loss_fn_factory(cfg.detection.gnn_training.decoder.reconstruct_node_embeddings.loss)

            decoder = decoder_factory(method, objective, cfg, in_dim=node_out_dim, out_dim=node_out_dim)
            objectives.append(NodeEmbDecoder(decoder=decoder, loss_fn=loss_fn))
        
        elif objective == "reconstruct_edge_embeddings":
            loss_fn = recon_loss_fn_factory(cfg.detection.gnn_training.decoder.reconstruct_edge_embeddings.loss)
            in_dim_edge = node_out_dim * 2  # concatenation of 2 nodes
            
            decoder = decoder_factory(method, objective, cfg, in_dim=in_dim_edge, out_dim=in_dim_edge)
            objectives.append(
                EdgeEmbReconstruction(
                    decoder=decoder,
                    loss_fn=loss_fn,
                ))
            
        elif objective == "predict_edge_type":
            loss_fn = categorical_loss_fn_factory("cross_entropy")
            balanced_loss = cfg.detection.gnn_training.decoder.predict_edge_type.balanced_loss
            
            decoder = decoder_factory(method, objective, cfg, in_dim=node_out_dim, out_dim=cfg.dataset.num_edge_types)
            objectives.append(
                EdgeTypePrediction(
                    decoder=decoder,
                    loss_fn=loss_fn,
                    balanced_loss=balanced_loss,
                    edge_type_dim=cfg.dataset.num_edge_types,
                ))
            
        elif objective == "predict_node_type":
            loss_fn = categorical_loss_fn_factory("cross_entropy")
            balanced_loss = cfg.detection.gnn_training.decoder.predict_node_type.balanced_loss
            
            decoder = decoder_factory(method, objective, cfg, in_dim=node_out_dim, out_dim=node_out_dim)
            objectives.append(
                NodeTypePrediction(
                    decoder=decoder,
                    loss_fn=loss_fn,
                    balanced_loss=balanced_loss,
                    node_type_dim=cfg.dataset.num_node_types,
                ))
            
        elif objective == "reconstruct_masked_features":
            mask_rate = cfg.detection.gnn_training.decoder.reconstruct_masked_features.mask_rate

            loss_fn = recon_loss_fn_factory(cfg.detection.gnn_training.decoder.reconstruct_masked_features.loss)

            decoder = decoder_factory(method, objective, cfg, in_dim=node_out_dim, out_dim=in_dim)
            objectives.append(
                GMAEFeatReconstruction(
                    decoder=decoder,
                    loss_fn=loss_fn,
                    mask_rate=mask_rate,
                )
            )
        
        elif objective == "predict_masked_struct":
            loss_fn = categorical_loss_fn_factory(cfg.detection.gnn_training.decoder.predict_masked_struct.loss)

            decoder = decoder_factory(method, objective, cfg, in_dim=node_out_dim * 2, out_dim=1)
            objectives.append(
                GMAEStructPrediction(
                    decoder=decoder,
                    loss_fn=loss_fn,
                )
            )
        
        # elif objective == "predict_edge_contrastive":
        #     predict_edge_method = cfg.detection.gnn_training.decoder.predict_edge_contrastive.used_method.strip()
        #     if predict_edge_method == "linear":
        #         edge_decoder = EdgeLinearDecoder(
        #             in_dim=node_out_dim,
        #             dropout=cfg.detection.gnn_training.decoder.predict_edge_contrastive.linear.dropout,
        #         )
        #     elif predict_edge_method == "inner_product":
        #         edge_decoder = EdgeInnerProductDecoder(
        #             dropout=cfg.detection.gnn_training.decoder.predict_edge_contrastive.inner_product.dropout,
        #         )
        #     else:
        #         raise ValueError(f"Invalid edge decoding method {predict_edge_method}")
            
        #     contrastive_graph_reindexer = GraphReindexer(
        #         num_nodes=max_node_num,
        #         device=device,
        #     )
        #     loss_fn = bce_contrastive
        #     neg_sampling_method = cfg.detection.gnn_training.decoder.predict_edge_contrastive.neg_sampling_method.strip()
        #     if neg_sampling_method not in ["nodes_in_current_batch", "previously_seen_nodes"]:
        #         raise ValueError(f"Invalid negative sampling method {neg_sampling_method}")
            
        #     objectives.append(EdgeContrastiveDecoder(
        #         decoder=edge_decoder,
        #         loss_fn=loss_fn,
        #         graph_reindexer=contrastive_graph_reindexer,
        #         neg_sampling_method=neg_sampling_method,
        #     ))
        
        else:
            raise ValueError(f"Invalid objective {objective}")
        
    # We wrap objectives into this class to calculate some metrics on validation set easily
    graph_reindexer = GraphReindexer(
        num_nodes=max_node_num,
        device=device,
    )
    is_edge_type_prediction = cfg.detection.gnn_training.decoder.used_methods.strip() == "predict_edge_type"
    objectives = [
        ValidationContrastiveStopper(
            objective,
            graph_reindexer,
            is_edge_type_prediction,
        ) for objective in objectives]
    
    return objectives

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

def batch_loader_factory(cfg, data, graph_reindexer, test_mode=False):
    use_tgn = "tgn" in cfg.detection.gnn_training.encoder.used_methods
    neigh_sampling = cfg.detection.gnn_training.encoder.neighbor_sampling
    
    try:
        if len(neigh_sampling) > 0 and isinstance(neigh_sampling[0], str):
            neigh_sampling = eval("".join(neigh_sampling))
        use_neigh_sampling = all([isinstance(num_hop, int) for num_hop in neigh_sampling])
        error = False
    except:
        error = True
    if error or not use_neigh_sampling:
        raise ValueError(f"Invalid neighbor sampling {neigh_sampling}. Expected 'None' or a list of integers.")
    
    # Use neigh sampling batch loader
    if use_neigh_sampling and len(neigh_sampling) > 0 and not all([n == -1 for n in neigh_sampling]):
        if use_tgn:
            raise ValueError(f"Cannot use both TGN and traditional neighbor sampling.")

        if graph_reindexer is not None:
            data = graph_reindexer.reindex_graph(data)
        data = temporal_data_to_data(data)
        return NeighborLoader(
            data,
            num_neighbors=neigh_sampling,
            batch_size=10_000_000, # no need for batching as a time window is already small
            shuffle=False,
        )
    # Use TGN batch loader
    if use_tgn:
        batch_size = cfg.detection.gnn_training.encoder.tgn.tgn_batch_size_inference if test_mode \
            else cfg.detection.gnn_training.encoder.tgn.tgn_batch_size
        return custom_temporal_data_loader(data, batch_size=batch_size)
    
    # Don't use any batching
    if graph_reindexer is not None:
        data = graph_reindexer.reindex_graph(data)
    return [data]

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

    return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay) # TODO: parametrize

def get_dimensions_from_data_sample(data):
    edge_dim = data.edge_feats.shape[1] if hasattr(data, "edge_feats") else None
    msg_dim = data.msg.shape[1] if hasattr(data, "msg") else edge_dim
    in_dim = data.x_src.shape[1] if hasattr(data, "x_src") else data.x.shape[1]
    
    return msg_dim, edge_dim, in_dim
