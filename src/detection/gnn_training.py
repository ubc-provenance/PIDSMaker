import argparse
import logging
from time import perf_counter as timer

import torch.nn as nn
import wandb
from torch_geometric.loader import NeighborLoader

from provnet_utils import *
from config import *
from model import *
from losses import sce_loss, bce_contrastive
from encoders import *
from decoders import *
from data_utils import (
    custom_temporal_data_loader,
    temporal_data_to_data,
    GraphReindexer,
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def encoder_factory(cfg, msg_dim, in_dim, edge_dim):
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

    if "graph_attention" in cfg.detection.gnn_training.encoder.used_methods:
        encoder = GraphAttentionEmbedding(
            in_dim=in_dim,
            hid_dim=node_hid_dim,
            out_dim=node_out_dim,
            edge_dim=edge_dim or None,
            node_dropout=cfg.detection.gnn_training.node_dropout,
            num_heads=cfg.detection.gnn_training.encoder.graph_attention.num_heads,
        ).to(device)
    
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

def recon_loss_fn_factory(loss: str):
    if loss == "SCE":
        return sce_loss
    if loss == "MSE":
        return F.mse_loss
    raise ValueError(f"Invalid loss function {loss}")

def activation_fn_factory(activation: str):
    if activation == "sigmoid":
        return torch.sigmoid
    if activation == "relu":
        return torch.relu
    if activation == "tanh":
        return torch.tanh
    raise ValueError(f"Invalid activation function {activation}")

def decoder_factory(cfg, in_dim):
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

def model_factory(encoder, decoders, cfg, in_dim):
    return Model(
        encoder=encoder,
        decoders=decoders,
        num_nodes=cfg.dataset.max_node_num,
        device=device,
        in_dim=in_dim,
        out_dim=cfg.detection.gnn_training.node_out_dim,
        use_contrastive_learning="predict_edge_contrastive" in cfg.detection.gnn_training.decoder.used_methods,
    )

def optimizer_factory(cfg, parameters):
    lr = cfg.detection.gnn_training.lr
    weight_decay = cfg.detection.gnn_training.weight_decay

    return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay) # TODO: parametrize

def batch_loader_factory(cfg, data, graph_reindexer):
    use_tgn = "tgn" in cfg.detection.gnn_training.encoder.used_methods
    neigh_sampling = cfg.detection.gnn_training.encoder.neighbor_sampling
    
    try:
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
    
def train(data,
          model,
          optimizer,
          graph_reindexer,
          cfg
          ):
    model.train()

    if isinstance(model.encoder, TGNEncoder):
        model.encoder.reset_state()

    losses = []
    batch_loader = batch_loader_factory(cfg, data, graph_reindexer)

    for batch in batch_loader:
        optimizer.zero_grad()

        loss = model(batch, data)

        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return np.mean(losses)

def load_train_data(cfg):
    edge_embeds_dir = cfg.featurization.embed_edges._edge_embeds_dir
    glist = []
    filelist = sorted(os.listdir(os.path.join(edge_embeds_dir, "train")))
    for file in filelist:
        filepath = os.path.join(edge_embeds_dir, "train", file)
        g = torch.load(filepath).to(device='cpu')
        glist.append(g)
    return glist

def extract_msg_from_data(train_data, cfg):
    emb_dim = cfg.featurization.embed_nodes.emb_dim
    node_type_dim = cfg.dataset.num_node_types
    edge_type_dim = cfg.dataset.num_edge_types
    
    msg_len = train_data[0].msg.shape[1]
    expected_msg_len = (emb_dim*2) + (node_type_dim*2) + edge_type_dim
    if msg_len != expected_msg_len:
        raise ValueError(f"The msg has an invalid shape, found {msg_len} instead of {expected_msg_len}")
    
    field_to_size = [
        ("src_type", node_type_dim),
        ("src_emb", emb_dim),
        ("edge_type", edge_type_dim),
        ("dst_type", node_type_dim),
        ("dst_emb", emb_dim),
    ]
    for g in train_data:
        fields = {}
        idx = 0
        for field, size in field_to_size:
            fields[field] = g.msg[:, idx: idx + size]
            idx += size
            
        x_src = fields["src_emb"]
        x_dst = fields["dst_emb"]
        
        if cfg.detection.gnn_training.encoder.use_node_type_in_node_feats:
            x_src = torch.cat([x_src, fields["src_type"]], dim=-1)
            x_dst = torch.cat([x_dst, fields["dst_type"]], dim=-1)
        
        # If we want to predict the edge type, we remove the edge type from the message
        if "predict_edge_type" in cfg.detection.gnn_training.decoder.used_methods:
            msg = torch.cat([x_src, x_dst], dim=-1)
            edge_feats = None
        else:
            msg = torch.cat([x_src, x_dst, fields["edge_type"]], dim=-1)
            edge_feats = fields["edge_type"] # For now, we only use the edge type as edge feature
            
        g.x_src = x_src
        g.x_dst = x_dst
        g.msg = msg
        g.edge_type = fields["edge_type"]
        g.edge_feats = edge_feats
        g.edge_index = torch.stack([g.src, g.dst])
    
    return train_data

def main(cfg):
    logger = get_logger(
        name="gnn_training",
        filename=os.path.join(cfg.detection.gnn_training._logs_dir, "gnn_training.log"))
    
    if cfg.detection.gnn_training.use_seed:
        seed = 0
        random.seed(seed)
        np.random.seed(seed)

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    gnn_models_dir = cfg.detection.gnn_training._trained_models_dir
    os.makedirs(gnn_models_dir, exist_ok=True)

    train_data = load_train_data(cfg)
    train_data = extract_msg_from_data(train_data, cfg)
    
    msg_dim = train_data[0].msg.shape[1]
    edge_dim = train_data[0].edge_feats.shape[1] \
        if hasattr(train_data[0], "edge_feats") else None
    in_dim = train_data[0].x_src.shape[1]

    encoder = encoder_factory(cfg, msg_dim=msg_dim, in_dim=in_dim, edge_dim=edge_dim)
    decoder = decoder_factory(cfg, in_dim=in_dim)
    model = model_factory(encoder, decoder, cfg, in_dim=in_dim)
    graph_reindexer = GraphReindexer(
        num_nodes=cfg.dataset.max_node_num,
        device=device,
    )
    optimizer = optimizer_factory(cfg, parameters=set(model.parameters()))
    
    num_epochs = cfg.detection.gnn_training.num_epochs
    tot_loss = 0.0
    for epoch in tqdm(range(1, num_epochs+1)):
        start = timer()
        for g in train_data:
            g.to(device=device)
            loss = train(
                data=g.clone(), # avoids alteration of the graph across epochs
                model=model,
                optimizer=optimizer,
                graph_reindexer=graph_reindexer,
                cfg=cfg,
            )
            tot_loss += loss
            print(f"Loss {loss:4f}")
        
        tot_loss /= len(train_data)
        logger.info(f'  Epoch: {epoch:02d}, Loss: {tot_loss:.4f}')
        wandb.log({
            "train_epoch": epoch,
            "train_loss": round(tot_loss, 4),
            "train_epoch_time": round(timer() - start, 2),
        })
        print(f'GNN training loss Epoch: {epoch:02d}, Loss: {tot_loss:.4f}')

        # Check points
        if epoch % 5 == 0:
            torch.save(model, f"{gnn_models_dir}/model_epoch{epoch}.pt")


if __name__ == "__main__":
    args = get_runtime_required_args()
    cfg = get_yml_cfg(args)

    main(cfg)
