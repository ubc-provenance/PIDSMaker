import torch
import torch.nn as nn
from torch_geometric.loader import NeighborLoader

from provnet_utils import *
from config import *
from model import *
from losses import sce_loss, bce_contrastive
from encoders import *
from decoders import *
from data_utils import *


def model_factory(encoder, decoders, cfg, in_dim, device):
    return Model(
        encoder=encoder,
        decoders=decoders,
        num_nodes=cfg.dataset.max_node_num,
        device=device,
        in_dim=in_dim,
        out_dim=cfg.detection.gnn_training.node_out_dim,
        use_contrastive_learning="predict_edge_contrastive" in cfg.detection.gnn_training.decoder.used_methods,
    )

def encoder_factory(cfg, msg_dim, in_dim, edge_dim, device):
    node_hid_dim = cfg.detection.gnn_training.node_hid_dim
    node_out_dim = cfg.detection.gnn_training.node_out_dim
    max_node_num = cfg.dataset.max_node_num
    tgn_memory_dim = cfg.detection.gnn_training.encoder.tgn.tgn_memory_dim
    use_msg_as_edge_feature = cfg.detection.gnn_training.encoder.tgn.use_msg_as_edge_feature
    use_time_encoding = cfg.detection.gnn_training.encoder.tgn.use_time_encoding
    
    # If edge features are used in TGN, and the downstream encoder uses edge features, we set them here
    if "tgn" in cfg.detection.gnn_training.encoder.used_methods:
        edge_dim = 0
        if use_msg_as_edge_feature:
            edge_dim += msg_dim
        if use_time_encoding:
            edge_dim += tgn_memory_dim

        # Only for TGN, in_dim becomes memory dim
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
                dropout=cfg.detection.gnn_training.encoder.graph_attention.dropout,
                num_heads=cfg.detection.gnn_training.encoder.graph_attention.num_heads,
            ).to(device)
        elif method == "sage":
            encoder = SAGE(
                in_dim=in_dim,
                hid_dim=node_hid_dim,
                out_dim=node_out_dim,
                activation=activation_fn_factory(cfg.detection.gnn_training.encoder.sage.activation),
                dropout=cfg.detection.gnn_training.encoder.sage.dropout,
            ).to(device)
        else:
            raise ValueError(f"Invalid encoder {method}")
    
    if "tgn" in cfg.detection.gnn_training.encoder.used_methods:
        time_dim = cfg.detection.gnn_training.encoder.tgn.tgn_time_dim
        neighbor_size = cfg.detection.gnn_training.encoder.tgn.tgn_neighbor_size

        memory = TGNMemory(
            max_node_num,
            msg_dim,
            tgn_memory_dim,
            time_dim,
            message_module=IdentityMessage(msg_dim, tgn_memory_dim, time_dim),
            aggregator_module=LastAggregator(),
        ).to(device)
        neighbor_loader = LastNeighborLoader(max_node_num, size=neighbor_size, device=device)

        encoder = TGNEncoder(
            encoder=encoder,
            memory=memory,
            neighbor_loader=neighbor_loader,
            time_encoder=memory.time_enc,
            use_msg_as_edge_feature=use_msg_as_edge_feature,
            use_time_encoding=use_time_encoding,
        )

    return encoder

def decoder_factory(cfg, in_dim, device):
    node_out_dim = cfg.detection.gnn_training.node_out_dim

    decoders = []
    for method in map(lambda x: x.strip(), cfg.detection.gnn_training.decoder.used_methods.split(",")):
        if method == "reconstruct_node":
            recon_hid_dim = cfg.detection.gnn_training.decoder.reconstruct_node.recon_hid_dim
            recon_use_bias = cfg.detection.gnn_training.decoder.reconstruct_node.recon_use_bias
            loss_fn = recon_loss_fn_factory(cfg.detection.gnn_training.decoder.reconstruct_node.loss)
            out_activation = activation_fn_factory(cfg.detection.gnn_training.decoder.reconstruct_node.out_activation)

            src_recon = AutoEncoder(
                in_dim=node_out_dim,
                h_dim=recon_hid_dim,
                out_dim=in_dim,
                use_bias=recon_use_bias,
                out_activation=out_activation,
            ).to(device)
            dst_recon = AutoEncoder(
                in_dim=node_out_dim,
                h_dim=recon_hid_dim,
                out_dim=in_dim,
                use_bias=recon_use_bias,
                out_activation=out_activation,
            ).to(device)
            decoders.append(SrcDstNodeDecoder(src_decoder=src_recon, dst_decoder=dst_recon, loss_fn=loss_fn))
        
        elif method == "predict_edge_type":
            loss_fn = nn.CrossEntropyLoss()
            
            method = cfg.detection.gnn_training.decoder.predict_edge_type.used_method.strip()
            if method not in ["kairos", "custom"]:
                raise ValueError(f"Invalid edge type decoder method {method}")
            use_kairos_decoder = method == "kairos"
            activation = None if use_kairos_decoder else \
                activation_fn_factory(cfg.detection.gnn_training.decoder.predict_edge_type.custom.activation)
            
            decoder = EdgeTypeDecoder(
                in_dim=node_out_dim,
                num_edge_types=cfg.dataset.num_edge_types,
                loss_fn=loss_fn,
                use_kairos_decoder=use_kairos_decoder,
                dropout=cfg.detection.gnn_training.decoder.predict_edge_type.custom.dropout,
                num_layers=cfg.detection.gnn_training.decoder.predict_edge_type.custom.num_layers,
                activation=activation,
            )
            decoders.append(decoder)
        
        elif method == "predict_edge_contrastive":
            predict_edge_method = cfg.detection.gnn_training.decoder.predict_edge_contrastive.used_method.strip()
            if predict_edge_method == "linear":
                edge_decoder = EdgeLinearDecoder(
                    in_dim=node_out_dim,
                    dropout=cfg.detection.gnn_training.decoder.predict_edge_contrastive.linear.dropout,
                )
            elif predict_edge_method == "inner_product":
                edge_decoder = EdgeInnerProductDecoder(
                    dropout=cfg.detection.gnn_training.decoder.predict_edge_contrastive.inner_product.dropout,
                )
            else:
                raise ValueError(f"Invalid edge decoding method {predict_edge_method}")
            
            loss_fn = bce_contrastive
            decoders.append(EdgeContrastiveDecoder(decoder=edge_decoder, loss_fn=loss_fn))
        
        else:
            raise ValueError(f"Invalid decoder {method}")
        
    return decoders

def batch_loader_factory(cfg, data, graph_reindexer):
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
    if use_neigh_sampling and len(neigh_sampling) > 0:
        if use_tgn:
            raise ValueError(f"Cannot use both TGN and traditional neighbor sampling.")

        data = graph_reindexer(data)
        data = temporal_data_to_data(data)
        return NeighborLoader(
            data,
            num_neighbors=neigh_sampling,
            batch_size=10_000_000, # no need for batching as a time window is already small
        )
    # Use TGN batch loader
    if use_tgn:
        return custom_temporal_data_loader(data, batch_size=cfg.detection.gnn_training.encoder.tgn.tgn_batch_size)
    
    # Don't use any batching
    data = graph_reindexer(data)
    return [data]

def recon_loss_fn_factory(loss: str):
    if loss == "SCE":
        return sce_loss
    if loss == "MSE":
        return F.mse_loss
    raise ValueError(f"Invalid loss function {loss}")

def activation_fn_factory(activation: str):
    if activation == "sigmoid":
        return nn.Sigmoid()
    if activation == "relu":
        return nn.ReLU()
    if activation == "tanh":
        return nn.Tanh()
    raise ValueError(f"Invalid activation function {activation}")

def optimizer_factory(cfg, parameters):
    lr = cfg.detection.gnn_training.lr
    weight_decay = cfg.detection.gnn_training.weight_decay

    return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay) # TODO: parametrize

def get_dimensions_from_data_sample(data):
    msg_dim = data.msg.shape[1]
    edge_dim = data.edge_feats.shape[1] if hasattr(data, "edge_feats") else None
    in_dim = data.x_src.shape[1]
    
    return msg_dim, edge_dim, in_dim
