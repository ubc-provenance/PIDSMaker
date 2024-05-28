import logging
from time import perf_counter as timer

import torch.nn as nn
import wandb

from encoders import TGNEncoder
from config import *
from data_utils import *
from factory import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device == torch.device("cpu"):
    log("Warning: the device is CPU instead of CUDA")

def train(data,
          model,
          optimizer,
          cfg
          ):
    model.train()

    losses = []
    batch_loader = batch_loader_factory(cfg, data, model.graph_reindexer)

    for batch in batch_loader:
        optimizer.zero_grad()

        loss = model(batch, data)

        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return np.mean(losses)

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

    train_data = load_data_set(cfg, path=cfg.featurization.embed_edges._edge_embeds_dir, split="train")
    
    model = build_model(data_sample=train_data[0], device=device, cfg=cfg)
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
                cfg=cfg,
            )
            tot_loss += loss
            log(f"Loss {loss:4f}")
        
        tot_loss /= len(train_data)
        logger.info(f'  Epoch: {epoch:02d}, Loss: {tot_loss:.4f}')
        wandb.log({
            "train_epoch": epoch,
            "train_loss": round(tot_loss, 4),
            "train_epoch_time": round(timer() - start, 2),
        })
        log(f'GNN training loss Epoch: {epoch:02d}, Loss: {tot_loss:.4f}')

        # Check points
        if cfg._test_mode or epoch % 2 == 0:
            model_path = os.path.join(gnn_models_dir, f"model_epoch_{epoch}")
            save_model(model, model_path)


if __name__ == "__main__":
    args = get_runtime_required_args()
    cfg = get_yml_cfg(args)

    main(cfg)
