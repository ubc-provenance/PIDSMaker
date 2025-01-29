import logging
from time import perf_counter as timer

import torch.nn as nn
import tracemalloc
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

    grad_acc = cfg.detection.gnn_training.grad_accumulation
    loss_acc = None
    
    for i, batch in enumerate(batch_loader):
        optimizer.zero_grad()

        results = model(batch, full_data)
        loss = results["loss"]

        if loss_acc is None:
            loss_acc = loss
        else:
            loss_acc += loss
        
        if (i+1) % grad_acc == 0:
            loss_acc.backward()
            optimizer.step()
            loss_acc = None
        
        losses.append(loss.item())
    
    # Last batch
    if loss_acc is not None:
        loss_acc.backward()
        optimizer.step()
    
    return np.mean(losses)


def main(cfg):
    log_start(__file__)
    device = get_device(cfg)
    use_cuda = device == torch.device("cuda")
    
    # Reset the peak memory usage counter
    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device=device)
    tracemalloc.start()

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
    peak_train_cpu_mem = 0
    peak_train_gpu_mem = 0
    test_stats = None
    patience = cfg.detection.gnn_training.patience
    patience_counter = 0
    all_test_stats = []
    
    for epoch in range(0, num_epochs):
        start = timer()
        tracemalloc.start()

        # Before each epoch, we reset the memory
        if isinstance(model.encoder, TGNEncoder):
            model.encoder.reset_state()
            
        model.to_fine_tuning(False)

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
            if use_cuda:
                torch.cuda.empty_cache()
            
            tot_loss += loss

        tot_loss /= len(train_data)
        epoch_times.append(timer() - start)
        
        _, peak_inference_cpu_memory = tracemalloc.get_traced_memory()
        peak_train_cpu_mem = max(peak_train_cpu_mem, peak_inference_cpu_memory / (1024 ** 3))
        tracemalloc.stop()
        
        if use_cuda:
            peak_inference_gpu_memory = torch.cuda.max_memory_allocated(device=device) / (1024 ** 3)
            peak_train_gpu_mem = max(peak_train_gpu_mem, peak_inference_gpu_memory)
            torch.cuda.reset_peak_memory_stats(device=device)
            
        log(f'[@epoch{epoch:02d}] Training finished - GPU memory: {peak_train_gpu_mem:.2f} GB | CPU memory: {peak_train_cpu_mem:.2f} GB | Mean Loss: {tot_loss:.4f}', return_line=True)
        
        # Few-shot learning fine tuning
        if cfg.detection.gnn_training.decoder.use_few_shot:
            model.to_fine_tuning(True)
            
            tot_loss = 0
            for g in log_tqdm(train_data, f"Fine-tuning"):
                g.to(device=device)
                loss = train( # NOTE: THERE IS NO MALICIOUS SAMPLES IN THE TRAIN SET ONLY TRAINED ON BENIGN NOW + WE SHOULD TRAIN SSL, THEN FEW_SHOT NOT AT THE SAME TIME
                    data=g,
                    full_data=full_data,  # full list of edge messages (do not store on CPU)
                    model=model,
                    optimizer=optimizer,
                    cfg=cfg,
                )
                g.to("cpu")
                if use_cuda:
                    torch.cuda.empty_cache()
                
                tot_loss += loss
            tot_loss /= len(train_data)
            log(f'[@epoch{epoch:02d}] Fine-tuning finished - Mean Loss: {tot_loss:.4f}', return_line=True)
        
        # model_path = os.path.join(gnn_models_dir, f"model_epoch_{epoch}")
        # save_model(model, model_path, cfg)
        
        # Validation
        if (epoch+1) % 2 == 0 or epoch == 0:
            split_to_run = "val" if best_epoch_mode else "all"
            test_stats = orthrus_gnn_testing.main(
                cfg=cfg,
                model=model,
                val_data=val_data,
                test_data=test_data,
                full_data=full_data,
                epoch=epoch,
                split=split_to_run,
            )
            all_test_stats.append(test_stats)
            val_ap = test_stats["val_ap"]
            
            if best_epoch_mode:
                if val_ap > best_val_ap:
                    best_val_ap = val_ap
                    best_model = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
                    best_epoch = epoch
                    patience_counter = 0
                else:
                    patience_counter += 1
            model.to_device(device)
            
            # if patience_counter >= patience:
            #     log(f"Early stopping: best score is {best_val_ap:.4f}")
            #     break
            
            wandb.log({
                "train_epoch": epoch,
                "train_loss": round(tot_loss, 4),
                # "val_ap": round(val_ap, 5),
                "val_loss": round(test_stats["val_loss"], 4),
                "test_loss": round(test_stats["test_loss"], 4),
            })
        
    # After training
    if best_epoch_mode:
        model.load_state_dict(best_model)
        test_stats = orthrus_gnn_testing.main(
            cfg=cfg,
            model=model,
            val_data=val_data,
            test_data=test_data,
            full_data=full_data,
            epoch=best_epoch,
            split="test",
        )

    wandb.log({
        "train_epoch_time": round(np.mean(epoch_times), 2),
        # "val_score": round(best_val_ap, 5),
        "peak_train_cpu_memory": round(peak_train_cpu_mem, 3),
        "peak_train_gpu_memory": round(peak_train_gpu_mem, 3),
        "peak_inference_cpu_memory": round(np.max([d["peak_inference_cpu_memory"] for d in all_test_stats]), 3),
        "peak_inference_gpu_memory": round(np.max([d["peak_inference_gpu_memory"] for d in all_test_stats]), 3),
        "time_per_batch_inference": round(np.mean([d["time_per_batch_inference"] for d in all_test_stats]), 3),
    })
    
    return best_val_ap


if __name__ == "__main__":
    args = get_runtime_required_args()
    cfg = get_yml_cfg(args)

    main(cfg)
