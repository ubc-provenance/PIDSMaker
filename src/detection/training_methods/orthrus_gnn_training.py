import logging
from time import perf_counter as timer

import torch.nn as nn
import wandb

from encoders import TGNEncoder
from config import *
from data_utils import *
from factory import *
from . import orthrus_gnn_testing


def train(
    data,
    full_data,
    model,
    optimizer,
    cfg,
):
    model.train()

    losses = []
    batch_loader = batch_loader_factory(cfg, data, model.graph_reindexer)

    for batch in batch_loader:
        optimizer.zero_grad()

        results = model(batch, full_data)
        loss = results["loss"]

        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return np.mean(losses)


def main(cfg):
    log_start(__file__)
    device = get_device(cfg)
    use_cuda = device == torch.device("cuda")
    
    # Reset the peak memory usage counter
    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device=device)

    train_data, val_data, test_data, full_data, max_node_num = load_all_datasets(cfg)

    model = build_model(data_sample=train_data[0], device=device, cfg=cfg, max_node_num=max_node_num)
    optimizer = optimizer_factory(cfg, parameters=set(model.parameters()))
    
    run_evaluation = cfg.training_loop.run_evaluation
    assert run_evaluation in ["best_epoch", "each_epoch"], f"Invalid run evaluation {run_evaluation}"
    best_epoch_mode = run_evaluation == "best_epoch"

    num_epochs = cfg.detection.gnn_training.num_epochs
    tot_loss = 0.0
    epoch_times = []
    best_val_ap, best_model, best_epoch = -1.0, None, None
    
    for epoch in range(0, num_epochs):
        start = timer()

        # Before each epoch, we reset the memory
        if isinstance(model.encoder, TGNEncoder):
            model.encoder.reset_state()

        tot_loss = 0
        for g in log_tqdm(train_data, f"Training"):
            g.to(device=device)
            loss = train(
                data=g,
                full_data=full_data,  # full list of edge messages (do not store on CPU)
                model=model,
                optimizer=optimizer,
                cfg=cfg,
            )
            g.to("cpu")
            tot_loss += loss

        tot_loss /= len(train_data)    
        epoch_times.append(timer() - start)
        
        if use_cuda:
            peak_memory = torch.cuda.max_memory_allocated(device=device) / (1024 ** 3)  # Convert to GB
        else:
            peak_memory = 0
            
        log(f'[@epoch{epoch:02d}] Training finished - Mean Loss: {tot_loss:.4f}, Peak CUDA memory: {peak_memory:.2f} GB', return_line=True)
        
        # Check points
        if cfg._test_mode or epoch % 1 == 0:
            # model_path = os.path.join(gnn_models_dir, f"model_epoch_{epoch}")
            # save_model(model, model_path, cfg)
            
            split_to_run = "val" if best_epoch_mode else "all"
            val_ap = orthrus_gnn_testing.main(
                cfg=cfg,
                model=model,
                val_data=val_data,
                test_data=test_data,
                full_data=full_data,
                epoch=epoch,
                split=split_to_run,
            )
            if best_epoch_mode:
                if val_ap > best_val_ap:
                    best_val_ap = val_ap
                    best_model = copy.deepcopy(model)
                    best_epoch = epoch
            model.to_device(device)
            
        wandb.log({
            "train_epoch": epoch,
            "train_loss": round(tot_loss, 4),
            "peak_cuda_memory_GB": round(peak_memory, 2),
            "val_ap": round(val_ap, 5),
        })
        
    if best_epoch_mode:
        orthrus_gnn_testing.main(
            cfg=cfg,
            model=best_model,
            val_data=val_data,
            test_data=test_data,
            full_data=full_data,
            epoch=best_epoch,
            split="test",
        )

    wandb.log({
        "train_epoch_time": round(np.mean(epoch_times), 2),
        "val_ap": round(best_val_ap, 5),
    })
    
    return best_val_ap


if __name__ == "__main__":
    args = get_runtime_required_args()
    cfg = get_yml_cfg(args)

    main(cfg)
