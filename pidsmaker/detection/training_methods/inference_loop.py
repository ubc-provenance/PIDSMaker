import json
import math
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors

from pidsmaker.detection.flash_detection import flash_anomaly_score
from pidsmaker.utils.data_utils import batch_to_device, to_original_node_ids
from pidsmaker.utils.labelling import has_event_labels
from pidsmaker.utils.loss_files import write_losses
from pidsmaker.utils.utils import (
    PeakCPUMemory,
    calculate_average_from_file,
    get_device,
    log,
    log_tqdm,
    ns_time_to_datetime_US,
    set_seed,
)


def threatrace_log_ratio(log_probs, node_type):
    """log(p_true / max_{c != true} p_c) per node, from log-probabilities (N, C).

    ThreaTrace explains a node when its type is predicted with a top-1 / top-2 probability
    ratio of at least `thre`. This is equivalent to `ratio >= thre` with the ratio above: when
    the prediction is right it is the top-1 / top-2 ratio, and when it is wrong it is <= 1.
    """
    log_p_true = log_probs.gather(1, node_type.unsqueeze(1)).squeeze(1)
    others = log_probs.scatter(1, node_type.unsqueeze(1), float("-inf"))
    return log_p_true - others.max(dim=1).values


def threatrace_anomaly_score(outs, node_type, ratio_threshold):
    """ThreaTrace's anomaly score, positive iff the node is anomalous.

    A node is anomalous when no submodel predicts its type with a probability ratio of at least
    `ratio_threshold` (test_darpatc.py). The score is
    `log(ratio_threshold) - max_k log(p_true / max_{c != true} p_c)` over the submodels `outs`
    (log-probabilities), so that anomalous nodes are exactly those with a score > 0, and nodes
    whose type is the least well predicted rank first.
    """
    best = torch.stack([threatrace_log_ratio(out, node_type) for out in outs]).max(dim=0).values
    return math.log(ratio_threshold) - best


@torch.no_grad()
def test_edge_level(
    data,
    model,
    split,
    model_epoch_file,
    cfg,
    device,
):
    model.eval()

    validation = split == "val"

    results = model(data, inference=True, validation=validation)
    each_edge_loss = results["loss"]

    # If the data has been reindexed in the loader or batched, we retrieve original node IDs
    # to later find the labels
    edge_index = to_original_node_ids(data.original_edge_index)

    # A single device->host transfer for all columns
    edge_types = data.edge_type.max(dim=1).indices + 1
    srcnodes, dstnodes, t_vars, edge_types = (
        torch.stack([edge_index[0], edge_index[1], data.t.to(edge_index.dtype), edge_types.to(edge_index.dtype)])
        .cpu()
        .numpy()
    )
    losses = each_edge_loss.float().cpu().numpy()
    start_time = int(t_vars[0])

    # if 1 in data.y:
    #     log(f"Mean score of fake malicious edges: {losses[data.y.cpu() == 1].mean():.4f}")
    #     log(f"Mean score of benign malicious edges: {losses[data.y.cpu() == 0].mean():.4f}")

    edge_cols = {
        "loss": losses.astype(float),
        "srcnode": srcnodes.astype(int),
        "dstnode": dstnodes.astype(int),
        "time": t_vars.astype(int),
        "edge_type": edge_types.astype(int),
    }
    # Datasets with event-level labels: the label of each edge is kept for the evaluation
    if has_event_labels(cfg):
        edge_cols["y"] = data.y.cpu().numpy().astype(int)

    # Here is a checkpoint, which records all edge losses in the current time window
    time_interval = (
        ns_time_to_datetime_US(start_time, timezone=cfg.dataset.timezone) + "~" + ns_time_to_datetime_US(edge_cols["time"].max(), timezone=cfg.dataset.timezone)
    )

    logs_dir = os.path.join(cfg.training._edge_losses_dir, split, model_epoch_file)
    os.makedirs(logs_dir, exist_ok=True)
    write_losses(logs_dir, time_interval, edge_cols)
    return losses


@torch.no_grad()
def test_node_level(
    data,
    model,
    split,
    model_epoch_file,
    cfg,
    device,
):
    model.eval()

    node_list = None
    start_time = int(data.t[0])
    end_time = int(data.t[-1])

    validation = split == "val"

    data = batch_to_device(data, device)

    results = model(data, inference=True, validation=validation)
    loss = results["loss"]
    losses = loss.float().cpu().numpy()
    n_id = to_original_node_ids(getattr(data, "original_n_id_tgn", getattr(data, "original_n_id")))
    n_id_np = n_id.cpu().numpy()

    # ThreaTrace code
    if cfg.evaluation.node_evaluation.threshold_method == "threatrace":
        # One output per submodel (ThreaTrace's multi-model scheme), or the single model's
        node_type_num = data.node_type.argmax(1)
        outs = results.get("outs", [results["out"]])
        score = threatrace_anomaly_score(
            outs, node_type_num, cfg.training.threatrace.ratio_threshold
        )

        node_list = {
            "node": n_id_np,
            "loss": losses.astype(float),
            "threatrace_score": score.cpu().numpy().astype(float),
        }

    # Flash code
    elif cfg.evaluation.node_evaluation.threshold_method == "flash":
        # `out` holds log-probabilities: the confidence of FLASH is computed on probabilities
        score, correct = flash_anomaly_score(results["out"].exp(), data.node_type.argmax(1))
        ensemble = getattr(model, "flash_ensemble", None)
        if ensemble is not None:
            score = ensemble.score(model, data, score)
        node_list = {
            "node": n_id_np,
            "loss": losses.astype(float),
            "flash_score": score.cpu().numpy().astype(float),
            "correct_pred": correct.long().cpu().numpy(),
        }

    # Magic codes
    elif cfg.evaluation.node_evaluation.threshold_method == "magic":
        os.makedirs(cfg.training._magic_dir, exist_ok=True)
        if split == "val":
            x_train, *_ = model.embed(data, inference=True)
            x_train = x_train.cpu().numpy()
            num_nodes = x_train.shape[0]
            sample_size = 5000 if num_nodes > 5000 else num_nodes
            sample_indices = np.random.choice(num_nodes, sample_size, replace=False)
            x_train_sampled = x_train[sample_indices]
            x_train_mean = x_train_sampled.mean(axis=0)
            x_train_std = x_train_sampled.std(axis=0)
            x_train_sampled = (x_train_sampled - x_train_mean) / x_train_std

            x_train_sampled = pd.DataFrame.from_records(x_train_sampled)

            n_neighbors = 10
            nbrs = NearestNeighbors(n_neighbors=n_neighbors)
            nbrs.fit(x_train_sampled)
            idx = list(range(x_train_sampled.shape[0]))
            random.shuffle(idx)
            try:
                sample = x_train_sampled.iloc[idx[: min(50000, x_train_sampled.shape[0])]]
                distances_train, _ = nbrs.kneighbors(
                    sample, n_neighbors=min(len(sample), n_neighbors)
                )
            except KeyError as e:
                log(f"KeyError encountered: {e}")
                log(f"Available columns in x_train: {x_train_sampled.columns}")
                raise
            mean_distance_train = distances_train.mean().mean()
            if mean_distance_train == 0:
                mean_distance_train = 1e-9

            train_distance_file = os.path.join(cfg.training._magic_dir, "train_distance.txt")
            with open(train_distance_file, "a") as f:
                f.write(f"{mean_distance_train}\n")

            node_list = {"node": n_id_np, "loss": losses.astype(float)}

        elif split == "test":
            train_distance_file = os.path.join(cfg.training._magic_dir, "train_distance.txt")
            mean_distance_train = calculate_average_from_file(train_distance_file)

            x_test, *_ = model.embed(data, inference=True)
            x_test = x_test.cpu().numpy()
            num_nodes = x_test.shape[0]
            sample_size = 5000 if num_nodes > 5000 else num_nodes
            sample_indices = np.random.choice(num_nodes, sample_size, replace=False)
            x_test_sampled = x_test[sample_indices]
            x_test_mean = x_test_sampled.mean(axis=0)
            x_test_std = x_test_sampled.std(axis=0)
            x_test_sampled = (x_test_sampled - x_test_mean) / x_test_std

            torch.cuda.empty_cache()
            x_test_sampled = pd.DataFrame.from_records(x_test_sampled)

            n_neighbors = 10
            nbrs = NearestNeighbors(n_neighbors=n_neighbors)
            nbrs.fit(x_test_sampled)

            # Query in the same standardized space as the fitted sample.
            distances, _ = nbrs.kneighbors(
                (x_test - x_test_mean) / x_test_std, n_neighbors=n_neighbors
            )
            distances = distances.mean(axis=1)
            # distances = distances.to_numpy()
            score = distances / mean_distance_train

            node_list = {
                "node": n_id_np,
                "magic_score": score.astype(float),
                "loss": losses.astype(float),
            }

    else:
        node_list = {"node": n_id_np, "loss": losses.astype(float)}

    if node_list is None:  # e.g. magic on the train split
        node_list = {}

    time_interval = ns_time_to_datetime_US(start_time, timezone=cfg.dataset.timezone) + "~" + ns_time_to_datetime_US(end_time, timezone=cfg.dataset.timezone)

    logs_dir = os.path.join(cfg.training._edge_losses_dir, split, model_epoch_file)
    os.makedirs(logs_dir, exist_ok=True)
    write_losses(logs_dir, time_interval, node_list)
    return losses


def main(cfg, model, val_data, test_data, epoch, split, logging=True, train_data=None):
    set_seed(cfg)

    if split == "all":
        splits = [(val_data, "val"), (test_data, "test")]
    elif split == "val":
        splits = [(val_data, "val")]
    elif split == "test":
        splits = [(test_data, "test")]
    elif split == "train":
        splits = [(train_data, "train")]
    else:
        raise ValueError(f"Invalid split {split}")

    inference_device = cfg.training.inference_device
    if inference_device is not None:
        if inference_device not in ["cpu", "cuda"]:
            raise ValueError(f"Invalid inference device {inference_device}")
        device = torch.device(inference_device)
    else:
        device = get_device(cfg)
    use_cuda = device == torch.device("cuda")
    model.to_device(device)

    model_epoch_file = f"model_epoch_{epoch}"
    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device=device)

    val_score = 0.0
    peak_inference_cpu_mem = 0
    peak_inference_gpu_mem = 0
    tpb = []
    split2loss = {}

    for dataset, split_name in splits:
        desc = "Validation" if split_name == "val" else "Testing"

        cpu_mem = PeakCPUMemory().start()

        all_losses = []
        for graphs in dataset:
            for g in log_tqdm(graphs, desc=desc, logging=logging):
                g = batch_to_device(g, device)

                s = time.time()
                test_fn = test_node_level if (cfg._is_node_level and not cfg._is_hybrid_loss) else test_edge_level
                losses = test_fn(
                    data=g,
                    model=model,
                    split=split_name,
                    model_epoch_file=model_epoch_file,
                    cfg=cfg,
                    device=device,
                )
                all_losses.append(np.asarray(losses, dtype=np.float64).ravel())
                tpb.append(time.time() - s)
                del g

        peak_inference_cpu_mem = max(peak_inference_cpu_mem, cpu_mem.peak_gb())

        if use_cuda:
            peak_inference_gpu_memory = torch.cuda.max_memory_allocated(device=device) / (1024**3)
            peak_inference_gpu_mem = max(peak_inference_gpu_mem, peak_inference_gpu_memory)
            torch.cuda.reset_peak_memory_stats(device=device)

        mean_loss = np.mean(np.concatenate(all_losses)) if all_losses else np.nan
        split2loss[split_name] = mean_loss

        if split_name == "val":
            val_score = model.get_val_ap()
            # Read by the evaluation to select the best epoch on validation data
            # (`best_model_selection: best_val_score`)
            val_scores_dir = os.path.join(cfg.training._edge_losses_dir, "val_scores")
            os.makedirs(val_scores_dir, exist_ok=True)
            with open(os.path.join(val_scores_dir, f"{model_epoch_file}.json"), "w") as f:
                json.dump({"val_score": float(val_score)}, f)
            if logging:
                log(
                    f"[@epoch{epoch:02d}] Validation finished - Val Loss: {mean_loss:.4f}",
                    return_line=True,
                )
        else:
            if logging:
                log(
                    f"[@epoch{epoch:02d}] Test finished - Test Loss: {mean_loss:.4f}",
                    return_line=True,
                )

    del model

    stats = {
        "val_score": val_score,
        "val_loss": split2loss.get("val", None),
        "test_loss": split2loss.get("test", None),
        "peak_inference_cpu_memory": peak_inference_cpu_mem,
        "peak_inference_gpu_memory": peak_inference_gpu_mem,
        "time_per_batch_inference": np.mean(tpb),
    }
    return stats
