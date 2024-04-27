##########################################################################################
# Some of the code is adapted from:
# https://github.com/pyg-team/pytorch_geometric/blob/master/examples/tgn.py
##########################################################################################
import argparse
import logging
from time import perf_counter as timer

import torch.nn as nn
import wandb

from provnet_utils import *
from config import *
from model import *
from losses import sce_loss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def encoder_factory(cfg, edge_feat_size):
    node_hid_dim = cfg.detection.gnn_training.node_hid_dim
    node_out_dim = cfg.detection.gnn_training.node_out_dim
    max_node_num = cfg.dataset.max_node_num
    tgn_memory_dim = cfg.detection.gnn_training.encoder.tgn.tgn_memory_dim
    use_msg_as_edge_feature = cfg.detection.gnn_training.encoder.tgn.use_msg_as_edge_feature
    use_time_encoding = cfg.detection.gnn_training.encoder.tgn.use_time_encoding
    
    # If edge features are used in TGN, and the downstream encoder uses edge features, we set them here
    edge_dim = 0
    if "tgn" in cfg.detection.gnn_training.encoder.used_methods:
        if use_msg_as_edge_feature:
            edge_dim += edge_feat_size
        if use_time_encoding:
            edge_dim += tgn_memory_dim

    in_dim = tgn_memory_dim \
        if "graph_attention" in cfg.detection.gnn_training.encoder.used_methods \
        else cfg.featurization.embed_nodes.emb_dim

    if "graph_attention" in cfg.detection.gnn_training.encoder.used_methods:
        encoder = GraphAttentionEmbedding(
            in_channels=in_dim,
            hid_channels=node_hid_dim,
            out_channels=node_out_dim,
            edge_dim=edge_dim or None,
            node_dropout=cfg.detection.gnn_training.node_dropout,
        ).to(device)
    
    if "tgn" in cfg.detection.gnn_training.encoder.used_methods:
        time_dim = cfg.detection.gnn_training.encoder.tgn.tgn_time_dim
        neighbor_size = cfg.detection.gnn_training.encoder.tgn.tgn_neighbor_size

        memory = TGNMemory(
            max_node_num,
            edge_feat_size,
            tgn_memory_dim,
            time_dim,
            message_module=IdentityMessage(edge_feat_size, tgn_memory_dim, time_dim),
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

def loss_fn_factory(loss: str):
    if loss == "SCE":
        return sce_loss
    if loss == "CE":
        return nn.CrossEntropyLoss()
    raise ValueError(f"Invalid loss function {loss}")

def decoder_factory(cfg):
    node_out_dim = cfg.detection.gnn_training.node_out_dim
    emb_dim = cfg.featurization.embed_nodes.emb_dim

    decoders = []
    for method in map(lambda x: x.strip(), cfg.detection.gnn_training.decoder.used_methods.split(",")):
        if method == "node_recon_MLP":
            recon_hid_dim = cfg.detection.gnn_training.decoder.node_recon_MLP.recon_hid_dim
            recon_use_bias = cfg.detection.gnn_training.decoder.node_recon_MLP.recon_use_bias
            loss_fn = loss_fn_factory(cfg.detection.gnn_training.decoder.node_recon_MLP.recon_use_bias.loss)

            src_recon = NodeRecon_MLP(
                in_dim=node_out_dim,
                h_dim=recon_hid_dim,
                out_dim=emb_dim,
                use_bias=recon_use_bias,
            ).to(device)
            dst_recon = NodeRecon_MLP(
                in_dim=node_out_dim,
                h_dim=recon_hid_dim,
                out_dim=emb_dim,
                use_bias=recon_use_bias,
            ).to(device)
            decoders.append(SrcDstNodeDecoder(src_decoder=src_recon, dst_decoder=dst_recon, loss_fn=loss_fn))
        
        elif method == "predict_edge_type":
            loss_fn = loss_fn_factory(cfg.detection.gnn_training.decoder.predict_edge_type.recon_use_bias.loss)
            
            decoder = EdgeTypeDecoder(
                in_dim=node_out_dim,
                loss_fn=loss_fn,
            )
            decoders.append(decoder)
        else:
            raise ValueError(f"Invalid decoder {method}")
        
    return decoders

def model_factory(encoder, decoders, cfg):
    return Model(
        encoder,
        decoders,
    )

def optimizer_factory(cfg, parameters):
    lr = cfg.detection.gnn_training.lr
    weight_decay = cfg.detection.gnn_training.weight_decay

    return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay) # TODO: parametrize

def train(train_data,
          model,
          optimizer,
          cfg
          ):
    model.train()

    if isinstance(model.encoder, TGNEncoder):
        model.encoder.reset_state()

    total_loss = 0
    word_embedding_dim = cfg.featurization.embed_nodes.emb_dim
    batch_size = cfg.detection.gnn_training.encoder.tgn.tgn_batch_size

    for batch in train_data.seq_batches(batch_size=batch_size):
        optimizer.zero_grad()

        src, dst, t, msg = batch.src, batch.dst, batch.t, batch.msg
        edge_index = torch.stack([src, dst])
        h_src = msg[:, :word_embedding_dim]
        h_dst = msg[:, -word_embedding_dim:]

        loss = model(edge_index, t, h_src, h_dst, msg)

        loss.backward()
        optimizer.step()
        total_loss += float(loss) * batch.num_events
    return total_loss / train_data.num_events

def load_train_data(cfg):
    edge_embeds_dir = cfg.featurization.embed_edges._edge_embeds_dir
    glist = []
    filelist = sorted(os.listdir(os.path.join(edge_embeds_dir, "train")))
    for file in filelist:
        filepath = os.path.join(edge_embeds_dir, "train", file)
        g = torch.load(filepath).to(device='cpu')
        glist.append(g)
    return glist

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
    
    # If we want to predict the edge type, we remove the edge type from the message
    word_embedding_dim = cfg.featurization.embed_nodes.emb_dim
    if "predict_edge_type" in cfg.detection.gnn_training.decoder.used_methods:
        for g in train_data:
            g.msg = torch.cat([g.msg[:, :word_embedding_dim],  g.msg[:, -word_embedding_dim:]], dim=-1)

    encoder = encoder_factory(cfg, edge_feat_size=train_data[0].msg.size(-1))
    decoder = decoder_factory(cfg)
    model = model_factory(encoder, decoder, cfg)
    optimizer = optimizer_factory(cfg, parameters=set(model.parameters()))
    
    num_epochs = cfg.detection.gnn_training.num_epochs
    for epoch in tqdm(range(1, num_epochs+1)):
        for g in train_data:
            g.to(device=device)
            start = timer()
            loss = train(
                train_data=g,
                model=model,
                optimizer=optimizer,
                cfg=cfg,
                assoc=assoc
            )
            logger.info(f'  Epoch: {epoch:02d}, Loss: {loss:.4f}')
            wandb.log({
                "train_epoch": epoch,
                "train_loss": round(loss, 4),
                "train_epoch_time": round(timer() - start, 2),
            })
            print(f'GNN training loss Epoch: {epoch:02d}, Loss: {loss:.4f}')

        # Check points
        if epoch % 5 == 0:
            torch.save(model, f"{gnn_models_dir}/model_epoch{epoch}.pt")

    # Save the trained model
    torch.save(model, f"{gnn_models_dir}/model_epoch{epoch}.pt")


if __name__ == "__main__":
    args = get_runtime_required_args()
    cfg = get_yml_cfg(args)

    main(cfg)
