"""Training loop for PIDS models.

Handles model training with:
- Self-supervised pretraining with multiple objectives
- Optional few-shot fine-tuning for attack detection
- Gradient accumulation for large graphs
- Early stopping with patience
- Memory tracking (GPU and CPU)
- Validation-based model selection
"""

import copy
import tracemalloc
from time import perf_counter as timer

import numpy as np
import torch
import wandb

from pidsmaker.factory import (
    build_model,
    optimizer_factory,
    optimizer_few_shot_factory,
)
from pidsmaker.tasks.batching import get_preprocessed_graphs
from pidsmaker.utils.utils import get_device, log, log_start, log_tqdm, set_seed

from . import inference_loop


def main(cfg):
    """Main training loop executing self-supervised pretraining and optional few-shot fine-tuning.

    Training process:
    1. Self-supervised pretraining on reconstruction/prediction objectives
    2. Optional few-shot fine-tuning on labeled attack data
    3. Validation-based model selection (best epoch or each epoch)
    4. Early stopping with configurable patience

    Args:
        cfg: Configuration with training hyperparameters (epochs, lr, patience, etc.)

    Returns:
        float: Best validation score achieved during training
    """
    set_seed(cfg)

    log_start(__file__)
    device = get_device(cfg)
    use_cuda = device == torch.device("cuda")

    # Reset the peak memory usage counter
    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device=device)
    tracemalloc.start()

    train_data, val_data, test_data, max_node_num = get_preprocessed_graphs(cfg)

    model = build_model(
        data_sample=train_data[0][0], device=device, cfg=cfg, max_node_num=max_node_num
    )
    optimizer = optimizer_factory(cfg, parameters=set(model.parameters()))

    run_evaluation = cfg.training_loop.run_evaluation
    assert run_evaluation in ["best_epoch", "each_epoch"], (
        f"Invalid run evaluation {run_evaluation}"
    )
    best_epoch_mode = run_evaluation == "best_epoch"

    num_epochs = cfg.training.num_epochs
    tot_loss = 0.0
    epoch_times = []
    peak_train_cpu_mem = 0
    peak_train_gpu_mem = 0
    test_stats = None
    patience = cfg.training.patience
    patience_counter = 0
    all_test_stats = []
    global_best_val_score = float("-inf")
    use_few_shot = cfg.training.decoder.use_few_shot
    grad_acc = cfg.training.grad_accumulation

    if use_few_shot:
        num_epochs += 1  # in few-shot, the first epoch is without ssl training

    for epoch in range(0, num_epochs):
        best_val_score, best_model, best_epoch = float("-inf"), None, None

        if not use_few_shot or (use_few_shot and epoch > 0):
            start = timer()
            tracemalloc.start()

            # Before each epoch, we reset the memory
            model.reset_state()
            model.to_fine_tuning(False)

            loss_acc = torch.zeros(1, device=device)
            tot_loss = 0
            for dataset in train_data:
                for i, g in enumerate(log_tqdm(dataset, "Training")):
                    g.to(device=device)
                    g = remove_attacks_if_needed(g, cfg)
                    model.train()
                    optimizer.zero_grad()

                    results = model(g)
                    loss = results["loss"]
                    loss_acc += loss
                    tot_loss += loss.item()

                    if (i + 1) % grad_acc == 0:
                        loss_acc.backward()
                        optimizer.step()
                        loss_acc = torch.zeros(1, device=device)

                    g.to("cpu")
                    if use_cuda:
                        torch.cuda.empty_cache()

                # Last batch
                if loss_acc > 0:
                    loss_acc.backward()
                    optimizer.step()

            tot_loss /= sum(len(dataset) for dataset in train_data)
            epoch_times.append(timer() - start)

            _, peak_inference_cpu_memory = tracemalloc.get_traced_memory()
            peak_train_cpu_mem = max(peak_train_cpu_mem, peak_inference_cpu_memory / (1024**3))
            tracemalloc.stop()

            if use_cuda:
                peak_inference_gpu_memory = torch.cuda.max_memory_allocated(device=device) / (
                    1024**3
                )
                peak_train_gpu_mem = max(peak_train_gpu_mem, peak_inference_gpu_memory)
                torch.cuda.reset_peak_memory_stats(device=device)

            log(
                f"[@epoch{epoch:02d}] Training finished - GPU memory: {peak_train_gpu_mem:.2f} GB | CPU memory: {peak_train_cpu_mem:.2f} GB | Mean Loss: {tot_loss:.4f}",
                return_line=True,
            )

        # Few-shot learning fine tuning
        if use_few_shot:
            model.to_fine_tuning(True)
            optimizer = optimizer_few_shot_factory(cfg, parameters=set(model.parameters()))

            num_epochs_few_shot = cfg.training.decoder.few_shot.num_epochs_few_shot
            patience_few_shot = cfg.training.decoder.few_shot.patience_few_shot

            for tuning_epoch in range(0, num_epochs_few_shot):
                model.reset_state()

                loss_acc = torch.zeros(1, device=device)
                tot_loss = 0
                for dataset in train_data:
                    for g in log_tqdm(dataset, "Fine-tuning"):
                        if 1 in g.y:
                            g.to(device=device)
                            model.train()
                            optimizer.zero_grad()

                            results = model(g)
                            loss = results["loss"]
                            loss_acc += loss
                            tot_loss += loss.item()

                            if (i + 1) % grad_acc == 0:
                                loss_acc.backward()
                                optimizer.step()
                                loss_acc = torch.zeros(1, device=device)

                            g.to("cpu")
                            if use_cuda:
                                torch.cuda.empty_cache()

                    # Last batch
                    if loss_acc > 0:
                        loss_acc.backward()
                        optimizer.step()

                tot_loss /= sum(len(dataset) for dataset in train_data)

                # Validation
                val_stats = inference_loop.main(
                    cfg=cfg,
                    model=model,
                    val_data=val_data,
                    test_data=test_data,
                    epoch=epoch,
                    split="val",
                    logging=False,
                )
                val_loss = val_stats["val_loss"]
                val_score = val_stats["val_score"]

                if val_score > best_val_score:
                    best_val_score = val_score
                    best_model = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
                    patience_counter = 0
                else:
                    patience_counter += 1

                if val_score > global_best_val_score:
                    global_best_val_score = val_score
                    best_epoch = epoch

                log(
                    f"[@epoch{tuning_epoch:02d}] Fine-tuning - Train Loss: {tot_loss:.5f} | Val Loss: {val_loss:.4f}",
                    return_line=True,
                )

                if patience_counter >= patience_few_shot:
                    log(f"Early stopping: best few-shot loss is {best_val_score:.4f}")
                    break

            model.load_state_dict(best_model)
            model.to_device(device)

        # model_path = os.path.join(gnn_models_dir, f"model_epoch_{epoch}")
        # save_model(model, model_path, cfg)

        # Test
        if (epoch + 1) % 2 == 0 or epoch == 0:
            test_stats = inference_loop.main(
                cfg=cfg,
                model=model,
                val_data=val_data,
                test_data=test_data,
                epoch=epoch,
                split="all",
            )
            all_test_stats.append(test_stats)

            wandb.log(
                {
                    "epoch": epoch,
                    "train_epoch": epoch,
                    "train_loss": round(tot_loss, 4),
                    "val_score": round(test_stats["val_score"], 4),
                    "val_loss": round(test_stats["val_loss"], 4),
                    "test_loss": round(test_stats["test_loss"], 4),
                }
            )

    # After training
    if best_epoch_mode:
        model.load_state_dict(best_model)
        test_stats = inference_loop.main(
            cfg=cfg,
            model=model,
            val_data=val_data,
            test_data=test_data,
            epoch=best_epoch,
            split="test",
        )

    wandb.log(
        {
            "best_epoch": best_epoch,
            "train_epoch_time": round(np.mean(epoch_times), 2),
            "val_score": round(best_val_score, 5),
            "peak_train_cpu_memory": round(peak_train_cpu_mem, 3),
            "peak_train_gpu_memory": round(peak_train_gpu_mem, 3),
            "peak_inference_cpu_memory": round(
                np.max([d["peak_inference_cpu_memory"] for d in all_test_stats]), 3
            ),
            "peak_inference_gpu_memory": round(
                np.max([d["peak_inference_gpu_memory"] for d in all_test_stats]), 3
            ),
            "time_per_batch_inference": round(
                np.mean([d["time_per_batch_inference"] for d in all_test_stats]), 3
            ),
        }
    )

    return best_val_score


def remove_attacks_if_needed(graph, cfg):
    """Remove attack edges from graph for self-supervised training if configured.

    Args:
        graph: Graph batch with labels in graph.y
        cfg: Configuration with few_shot.include_attacks_in_ssl_training setting

    Returns:
        graph: Original graph or filtered graph without attacks (y=1)
    """
    if not cfg.training.decoder.few_shot.include_attacks_in_ssl_training:
        if 1 in graph.y:
            return graph.clone()[graph.y != 1]
    return graph
