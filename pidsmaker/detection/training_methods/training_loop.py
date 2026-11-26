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
import os
from time import perf_counter as timer

import numpy as np
import torch
import wandb

from pidsmaker.detection.flash_detection import flash_snapshot_ensemble
from pidsmaker.factory import (
    build_model,
    early_stop_tracker_factory,
    optimizer_factory,
    optimizer_few_shot_factory,
)
from pidsmaker.objectives.predict_node_type import NODE_LOSS_MASK_KEY
from pidsmaker.tasks.batching import get_preprocessed_graphs
from pidsmaker.utils import console
from pidsmaker.utils.data_utils import batch_to_device, save_model
from pidsmaker.utils.utils import PeakCPUMemory, get_device, log, log_tqdm, set_seed

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
    device = get_device(cfg)
    use_cuda = device == torch.device("cuda")

    # Reset the peak memory usage counter
    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device=device)
    cpu_mem = PeakCPUMemory()

    train_data, val_data, test_data, max_node_num = get_preprocessed_graphs(cfg)
    log("Graph data loaded.")
    # Seed must be after `get_preprocessed_graphs` as this function may perform random operations
    set_seed(cfg, seed=cfg.training.seed)

    # Opt-in (`--save_for_viz`): cache the test graphs so the embedding visualizer
    # can reload them instead of recomputing the whole batching pipeline. Off by
    # default — the cache is large and most runs never open the viewer; when it is
    # absent the exporter recomputes the graphs on demand. Only test_data is ever
    # read back, so there is no need to persist the train/val graphs too.
    if cfg._save_for_viz:
        viz_cache_dir = cfg.batching._preprocessed_graphs_dir
        os.makedirs(viz_cache_dir, exist_ok=True)
        viz_cache_file = os.path.join(viz_cache_dir, "viz_test_graphs.pkl")
        if not os.path.exists(viz_cache_file):
            log("Caching test graphs to disk for visualization reuse...")
            torch.save((test_data, max_node_num), viz_cache_file)

    model = build_model(
        data_sample=train_data[0][0], device=device, cfg=cfg, max_node_num=max_node_num,
        all_data=(val_data, test_data),
    )
    if cfg._from_weights:
        model.load_state_dict(torch.load(os.path.join(cfg._from_weights_path, "state_dict.pkl")))
    optimizer = optimizer_factory(cfg, parameters=set(model.parameters()))
    stable_optim = cfg.training.stable_optim

    run_evaluation = cfg.training_loop.run_evaluation
    assert run_evaluation in ["best_epoch", "each_epoch"], (
        f"Invalid run evaluation {run_evaluation}"
    )
    best_epoch_mode = run_evaluation == "best_epoch"

    num_epochs = 1 if cfg._from_weights else cfg.training.num_epochs
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

    es_tracker = early_stop_tracker_factory(cfg)
    # Stable optim: warmup cosine scheduler (computed over total training steps)
    scheduler = None
    if stable_optim:
        total_graphs = sum(len(ds) for ds in train_data)
        steps_per_epoch = max(1, total_graphs // grad_acc)
        total_steps = steps_per_epoch * num_epochs
        warmup_steps = max(1, total_steps // 20)
        from pidsmaker.pretrained.training_utils import WarmupCosineScheduler

        scheduler = WarmupCosineScheduler(optimizer, warmup_steps, total_steps)

    if use_few_shot:
        num_epochs += 1  # in few-shot, the first epoch is without ssl training

    # FLASH: a sequence of GNN snapshots, each trained on the nodes still misclassified
    model.flash_ensemble = flash_snapshot_ensemble(cfg)

    for epoch in range(0, num_epochs):
        print(flush=True)  # Blank line between the logs of epochs
        best_val_score, best_model, best_epoch = float("-inf"), None, None

        if not cfg._from_weights and (not use_few_shot or (use_few_shot and epoch > 0)):
            start = timer()
            cpu_mem.start()

            # Before each epoch, we reset the memory
            model.reset_state()
            model.to_fine_tuning(False)

            # Sum of the losses of the batches since the last optimizer step (without autograd
            # graph). Each batch is backpropagated right away and its gradients accumulate in
            # `.grad` until the step: the gradients are those of the summed losses, while only
            # the autograd graph of one batch is kept in memory (with `grad_accumulation` = N,
            # backpropagating the sum kept the graphs of N batches, e.g. of all the graphs of
            # the training set with OCR-APT).
            loss_acc = torch.zeros(1, device=device)
            # Accumulated on device to avoid a GPU->CPU sync at each batch
            tot_loss = torch.zeros((), dtype=torch.float64, device=device)
            model.train()
            optimizer.zero_grad()
            for dataset in train_data:
                for i, g in enumerate(log_tqdm(dataset, "Training")):
                    node_loss_mask = getattr(g, NODE_LOSS_MASK_KEY, None)
                    if node_loss_mask is not None and not node_loss_mask.any():
                        continue  # No node left to train on (FLASH: all learned by the snapshots)
                    g = to_device_without_attacks(g, cfg, device)

                    results = model(g)
                    loss = results["loss"]
                    # A batch without any trainable output (e.g. a graph of OCR-APT with no
                    # node to predict) gives a constant loss, which contributes no gradient
                    if loss.requires_grad:
                        loss.sum().backward()
                    loss_acc += loss.detach()
                    tot_loss += loss.detach().sum().double()

                    if (i + 1) % grad_acc == 0:
                        if stable_optim:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        if scheduler is not None:
                            scheduler.step()
                        optimizer.zero_grad()
                        loss_acc = torch.zeros(1, device=device)

                    del g, results, loss

                # Last batch
                if loss_acc > 0:
                    if stable_optim:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()

            if model.flash_ensemble is not None:
                model.flash_ensemble.after_epoch(model, train_data, device)

            tot_loss = tot_loss.item() / sum(len(dataset) for dataset in train_data)
            epoch_times.append(timer() - start)

            peak_train_cpu_mem = max(peak_train_cpu_mem, cpu_mem.peak_gb())

            if use_cuda:
                peak_inference_gpu_memory = torch.cuda.max_memory_allocated(device=device) / (
                    1024**3
                )
                peak_train_gpu_mem = max(peak_train_gpu_mem, peak_inference_gpu_memory)
                torch.cuda.reset_peak_memory_stats(device=device)

            gpu_mem = f"  GPU {peak_train_gpu_mem:.2f} GB" if use_cuda else ""
            log(
                f"epoch {epoch:02d}  train  loss {tot_loss:.4f}"
                + console.style(
                    f"  {console.format_duration(epoch_times[-1])}  "
                    f"CPU {peak_train_cpu_mem:.2f} GB{gpu_mem}",
                    "dim",
                )
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
                    for i, g in enumerate(log_tqdm(dataset, "Fine-tuning")):
                        if 1 in g.y:
                            g = batch_to_device(g, device)

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

        # Per-epoch checkpoints are only needed by the visualizer's epoch selector.
        # Opt-in via `--save_for_viz`; otherwise skip — the pipeline evaluates from
        # the in-memory best model and never reloads these from disk.
        if cfg._save_for_viz:
            gnn_models_dir = cfg.training._trained_models_dir
            model_path = os.path.join(gnn_models_dir, f"model_epoch_{epoch}")
            save_model(model, model_path)

        # Test
        if (epoch + 1) % 2 == 0 or epoch == 0:
            test_stats = inference_loop.main(
                cfg=cfg,
                model=model,
                val_data=val_data,
                test_data=test_data,
                train_data=train_data,
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
            if es_tracker is not None and es_tracker.after_eval_epoch(cfg, model, train_data, epoch):
                break

    # After training
    if best_epoch_mode:
        model.load_state_dict(best_model)
        test_stats = inference_loop.main(
            cfg=cfg,
            model=model,
            val_data=val_data,
            test_data=test_data,
            train_data=train_data,
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


def to_device_without_attacks(graph, cfg, device):
    """Copies `graph` to `device`, without its attack edges if they are excluded from the
    self-supervised training (see `remove_attacks_if_needed`).

    The CPU graph is not mutated, and whether it has attacks to remove is checked on it, to
    avoid a GPU sync at each batch.
    """
    has_attacks_to_remove = needs_attack_removal(graph, cfg)
    graph = batch_to_device(graph, device)
    if has_attacks_to_remove:
        graph = remove_attacks_if_needed(graph, cfg)
    return graph


def needs_attack_removal(graph, cfg):
    """Whether `remove_attacks_if_needed` would remove edges from `graph`."""
    return (not cfg.training.decoder.few_shot.include_attacks_in_ssl_training) and bool(
        (graph.y == 1).any()
    )


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
