from config import *
from provnet_utils import *
from yacs.config import CfgNode as CN

from .magic_utils.utils import set_random_seed, create_optimizer
from .magic_utils.loaddata import load_entity_level_dataset, load_metadata
from .magic_utils.autoencoder import build_model

from tqdm import tqdm
import torch
import os

def main(cfg):
    checkpoints_dir = cfg.featurization.embed_nodes.magic._magic_checkpoints_dir
    os.makedirs(checkpoints_dir, exist_ok=True)

    device = get_device(cfg)

    log("Get training args")
    train_args = CN()
    train_args.num_hidden = cfg.featurization.embed_nodes.magic.num_hidden
    train_args.num_layers = cfg.featurization.embed_nodes.magic.num_layers
    train_args.max_epoch = cfg.featurization.embed_nodes.magic.max_epoch
    train_args.negative_slope = cfg.featurization.embed_nodes.magic.negative_slope
    train_args.mask_rate = cfg.featurization.embed_nodes.magic.mask_rate
    train_args.alpha_l = cfg.featurization.embed_nodes.magic.alpha_l
    train_args.optimizer = cfg.featurization.embed_nodes.magic.optimizer
    train_args.lr = cfg.featurization.embed_nodes.magic.lr
    train_args.weight_decay = cfg.featurization.embed_nodes.magic.weight_decay

    set_random_seed(0)

    log("Get metadata")
    metadata = load_metadata(cfg=cfg)
    train_args.n_dim = metadata['node_feature_dim']
    train_args.e_dim = metadata['edge_feature_dim']

    log("Build model")
    model = build_model(train_args, device)
    model = model.to(device)
    model.train()

    optimizer = create_optimizer(train_args.optimizer, model, train_args.lr, train_args.weight_decay)
    epoch_iter = tqdm(range(train_args.max_epoch), desc='Epoch of MAGIC training')
    n_train = metadata['n_train']

    log("Start training")
    for epoch in epoch_iter:
        epoch_loss = 0.0
        for i in range(n_train):
            g, tw_name = load_entity_level_dataset(t='train', n=i, cfg=cfg)
            g.to(device)
            model.train()
            loss = model(g)
            loss /= n_train
            optimizer.zero_grad()
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            del g
        epoch_iter.set_description(f"Epoch {epoch} | train_loss: {epoch_loss:.4f}")

    torch.save(model.state_dict(), checkpoints_dir + "checkpoints.pt")
    log(f"state dict of trained model saved at {checkpoints_dir}")
    log("Training finished")


if __name__ == "__main__":
    args = get_runtime_required_args()
    cfg = get_yml_cfg(args)

    main(cfg)