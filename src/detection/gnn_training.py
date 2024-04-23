##########################################################################################
# Some of the code is adapted from:
# https://github.com/pyg-team/pytorch_geometric/blob/master/examples/tgn.py
##########################################################################################
import argparse
import logging

from provnet_utils import *
from config import *
from model import *


def train(train_data,
          memory,
          gnn,
          optimizer,
          neighbor_loader,
          src_recon,
          dst_recon,
          cfg
          ):
    memory.train()
    gnn.train()
    src_recon.train()
    dst_recon.train()

    memory.reset_state()  # Start with a fresh memory.
    neighbor_loader.reset_state()  # Start with an empty graph.

    total_loss = 0
    batch_size = cfg.detection.gnn_training.tgn_batch_size
    for batch in train_data.seq_batches(batch_size=batch_size):
        optimizer.zero_grad()

        src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg

        n_id = torch.cat([src, pos_dst]).unique()
        n_id, edge_index, e_id = neighbor_loader(n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        # Get updated memory of all nodes involved in the computation.
        z, last_update = memory(n_id)
        z = gnn(z, edge_index)

        y_pred_src = src_recon(z[assoc[src]])
        y_pred_dst = dst_recon(z[assoc[pos_dst]])

        word_embedding_dim = cfg.featurization.embed_nodes.emb_dim
        y_true_src = []
        y_true_dst = []
        for m in msg:
            y_true_src.append(m[:word_embedding_dim])
            y_true_dst.append(m[-word_embedding_dim:])
        y_true_src = torch.stack(y_true_src).to(device)
        y_true_dst = torch.stack(y_true_dst).to(device)

        loss_src = criterion(y_pred_src, y_true_src)
        loss_dst = criterion(y_pred_dst, y_true_dst)

        loss = loss_src + loss_dst

        # Update memory and neighbor loader with ground-truth state.
        memory.update_state(src, pos_dst, t, msg)
        neighbor_loader.insert(src, pos_dst)

        loss.backward()
        optimizer.step()
        memory.detach()
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

def init_models(edge_feat_size, cfg):
    node_hid_dim = cfg.detection.gnn_training.node_hid_dim
    node_out_dim = cfg.detection.gnn_training.node_out_dim
    tgn_node_state_dim = cfg.detection.gnn_training.tgn_node_state_dim
    time_dim = cfg.detection.gnn_training.tgn_time_dim
    max_node_num = cfg.dataset.max_node_num

    lr = cfg.detection.gnn_training.lr
    weight_decay = cfg.detection.gnn_training.weight_decay
    neighbor_size = cfg.detection.gnn_training.tgn_neighbor_size

    memory = TGNMemory(
        max_node_num,
        edge_feat_size,
        tgn_node_state_dim,
        time_dim,
        message_module=IdentityMessage(edge_feat_size, tgn_node_state_dim, time_dim),
        aggregator_module=LastAggregator(), # TODO: parametrize
    ).to(device)

    gnn = GraphAttentionEmbedding(
        in_channels=tgn_node_state_dim,
        hid_channels=node_hid_dim,
        out_channels=node_out_dim,
    ).to(device)

    src_recon = NodeRecon_MLP().to(device)
    dst_recon = NodeRecon_MLP().to(device)

    optimizer = torch.optim.Adam(
        set(memory.parameters()) | set(gnn.parameters()) | set(src_recon.parameters()) | set(dst_recon.parameters()),
        lr=lr, weight_decay=weight_decay)

    neighbor_loader = LastNeighborLoader(max_node_num, size=neighbor_size, device=device)

    return memory, gnn, src_recon, dst_recon, optimizer, neighbor_loader

def main(cfg):
    logger = get_logger(
        name="gnn_training",
        filename=os.path.join(cfg.detection.gnn_training._logs_dir, "gnn_training.log"))

    gnn_models_dir = cfg.detection.gnn_training._trained_models_dir
    os.makedirs(gnn_models_dir, exist_ok=True)

     # Load data for training
    train_data = load_train_data(cfg)

    # Initialize the models and the optimizer
    edge_feat_size = train_data[0].msg.size(-1)
    memory, gnn, src_recon, dst_recon, optimizer, neighbor_loader = init_models(edge_feat_size=edge_feat_size, cfg=cfg)
    # train the model
    num_epochs = cfg.detection.gnn_training.num_epochs
    for epoch in tqdm(range(1, num_epochs+1)):
        for g in train_data:
            g.to(device=device)
            loss = train(
                train_data=g,
                memory=memory,
                gnn=gnn,
                optimizer=optimizer,
                neighbor_loader=neighbor_loader,
                src_recon=src_recon,
                dst_recon=dst_recon,
                cfg=cfg,
            )
            logger.info(f'  Epoch: {epoch:02d}, Loss: {loss:.4f}')

        # Check points
        if epoch % 5 == 0:
            model = [memory, gnn, neighbor_loader, src_recon, dst_recon,]
            torch.save(model, f"{gnn_models_dir}/model_epoch{epoch}.pt")

    # Save the trained model
    model = [memory, gnn, neighbor_loader, src_recon, dst_recon,]
    torch.save(model, f"{gnn_models_dir}/model_epoch{epoch}.pt")


if __name__ == "__main__":
    args = get_runtime_required_args()
    cfg = get_yml_cfg(args)

    main(cfg)
