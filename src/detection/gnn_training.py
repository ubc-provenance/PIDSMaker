import logging
from time import perf_counter as timer

import torch.nn as nn
import wandb

from encoders import TGNEncoder
from config import *
from data_utils import *
from factory import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(data,
          model,
          optimizer,
          graph_reindexer,
          cfg
          ):
    model.train()

    losses = []
    batch_loader = batch_loader_factory(cfg, data, graph_reindexer)

    for batch in batch_loader:
        optimizer.zero_grad()

        loss = model(batch, data)

        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return np.mean(losses)

def main(cfg, save_model: bool=True):
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

    train_data = load_data_set(cfg, path=cfg.featurization.embed_edges._edge_embeds_dir, split="train")
    
    msg_dim, edge_dim, in_dim = get_dimensions_from_data_sample(train_data[0])

    encoder = encoder_factory(cfg, msg_dim=msg_dim, in_dim=in_dim, edge_dim=edge_dim, device=device)
    decoder = decoder_factory(cfg, in_dim=in_dim, device=device)
    model = model_factory(encoder, decoder, cfg, in_dim=in_dim, device=device)
    graph_reindexer = GraphReindexer(
        num_nodes=cfg.dataset.max_node_num,
        device=device,
    )
    optimizer = optimizer_factory(cfg, parameters=set(model.parameters()))
    
    num_epochs = cfg.detection.gnn_training.num_epochs
    tot_loss = 0.0
    for epoch in tqdm(range(1, num_epochs+1)):
        start = timer()
        
        # Before each epoch, we reset the memory
        if isinstance(model.encoder, TGNEncoder):
            model.encoder.reset_state()

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
        if cfg._test_mode or (save_model and epoch % 2 == 0):
            torch.save(model, f"{gnn_models_dir}/model_epoch{epoch}.pt")


if __name__ == "__main__":
    args = get_runtime_required_args()
    cfg = get_yml_cfg(args)

    main(cfg)
