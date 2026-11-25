import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F

from pidsmaker.detection.flash_detection import flash_anomaly_score
from pidsmaker.detection.magic_knn import MagicKNNDetector
from pidsmaker.utils.data_utils import batch_to_device, to_original_node_ids
from pidsmaker.utils.labelling import has_event_labels
from pidsmaker.utils.loss_files import write_losses
from pidsmaker.utils.utils import (
    PeakCPUMemory,
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


class EdgeLossWriter:
    """Writes the edge losses of the batches (one file per batch, see `write_losses`).

    The columns of the batches stay on the device and are moved to the host every
    `flush_every` batches with a single transfer, and the files are compressed and written
    by threads, instead of a transfer (waiting for the GPU) and a write at each batch.
    The files are the same."""

    def __init__(self, cfg, logs_dir, flush_every=256, num_threads=4):
        self.cfg = cfg
        self.logs_dir = logs_dir
        self.flush_every = flush_every
        self.pending = []
        self.losses = []
        self.pool = ThreadPoolExecutor(num_threads)
        self.futures = {}
        os.makedirs(logs_dir, exist_ok=True)

    def add(self, int_cols, losses, y=None):
        """`int_cols`: (4, E) tensor of (src, dst, time, edge type), `losses`: (E,)."""
        self.pending.append((int_cols, losses, y))
        if len(self.pending) >= self.flush_every:
            self.flush()

    def flush(self):
        if not self.pending:
            return
        sizes = [losses.numel() for _, losses, _ in self.pending]
        int_cols = torch.cat([c for c, _, _ in self.pending], dim=1).cpu().numpy()
        losses = torch.cat([l.float() for _, l, _ in self.pending]).cpu().numpy()
        has_y = self.pending[0][2] is not None
        ys = torch.cat([y for _, _, y in self.pending]).cpu().numpy() if has_y else None
        self.pending = []

        start = 0
        for size in sizes:
            end = start + size
            srcnodes, dstnodes, t_vars, edge_types = int_cols[:, start:end]
            batch_losses = losses[start:end]
            self.losses.append(batch_losses)
            edge_cols = {
                "loss": batch_losses.astype(float),
                "srcnode": srcnodes.astype(int),
                "dstnode": dstnodes.astype(int),
                "time": t_vars.astype(int),
                "edge_type": edge_types.astype(int),
            }
            if has_y:
                edge_cols["y"] = ys[start:end].astype(int)
            # Here is a checkpoint, which records all edge losses in the current time window
            tz = self.cfg.dataset.timezone
            time_interval = (
                ns_time_to_datetime_US(int(t_vars[0]), timezone=tz)
                + "~"
                + ns_time_to_datetime_US(edge_cols["time"].max(), timezone=tz)
            )
            # Batches with the same time interval write the same file: the last one is kept
            previous = self.futures.get(time_interval)
            if previous is not None:
                previous.result()
            self.futures[time_interval] = self.pool.submit(
                write_losses, self.logs_dir, time_interval, edge_cols
            )
            start = end

    def close(self):
        """Writes the remaining batches and returns the losses of all the batches."""
        self.flush()
        for f in self.futures.values():
            f.result()
        self.pool.shutdown()
        return self.losses


@torch.no_grad()
def test_edge_level(
    data,
    model,
    split,
    model_epoch_file,
    cfg,
    device,
    writer=None,
):
    model.eval()

    validation = split == "val"

    results = model(data, inference=True, validation=validation)
    each_edge_loss = results["loss"]

    # If the data has been reindexed in the loader or batched, we retrieve original node IDs
    # to later find the labels
    edge_index = to_original_node_ids(data.original_edge_index)

    edge_types = data.edge_type.max(dim=1).indices + 1
    int_cols = torch.stack(
        [edge_index[0], edge_index[1], data.t.to(edge_index.dtype), edge_types.to(edge_index.dtype)]
    )
    # Datasets with event-level labels: the label of each edge is kept for the evaluation
    y = data.y if has_event_labels(cfg) else None

    if writer is None:
        logs_dir = os.path.join(cfg.training._edge_losses_dir, split, model_epoch_file)
        writer = EdgeLossWriter(cfg, logs_dir)
        writer.add(int_cols, each_edge_loss, y)
        return writer.close()[0]
    writer.add(int_cols, each_edge_loss, y)
    return None


@torch.no_grad()
def test_node_level(
    data,
    model,
    split,
    model_epoch_file,
    cfg,
    device,
    magic_detector=None,
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

    # MAGIC: k-NN distance of the test nodes to the benign training nodes
    elif cfg.evaluation.node_evaluation.threshold_method == "magic":
        if split == "test":
            if magic_detector is None:
                raise ValueError("MAGIC detection requires a detector fitted on the train split.")
            x_test, *_ = model.embed(data, inference=True)
            score = magic_detector.score(magic_detection_embedding(model, x_test, cfg))
            node_list = {
                "node": n_id_np,
                "magic_score": score.cpu().numpy().astype(float),
                "loss": losses.astype(float),
            }
        else:
            node_list = {"node": n_id_np, "loss": losses.astype(float)}

    else:
        node_list = {"node": n_id_np, "loss": losses.astype(float)}

    if node_list is None:  # e.g. magic on the train split
        node_list = {}

    time_interval = ns_time_to_datetime_US(start_time, timezone=cfg.dataset.timezone) + "~" + ns_time_to_datetime_US(end_time, timezone=cfg.dataset.timezone)

    logs_dir = os.path.join(cfg.training._edge_losses_dir, split, model_epoch_file)
    os.makedirs(logs_dir, exist_ok=True)
    write_losses(logs_dir, time_interval, node_list)
    return losses


def magic_detection_embedding(model, h, cfg=None):
    """Node embedding scored by MAGIC's k-NN detector: the last encoder layer (`last`, the
    original code) or all the concatenated layers (`concat`, the paper)."""
    mode = (cfg.evaluation.node_evaluation.magic_embedding if cfg is not None else None) or "last"
    if mode == "last" and hasattr(model.encoder, "detection_embedding"):
        return model.encoder.detection_embedding(h)
    return h


@torch.no_grad()
def fit_magic_detector(cfg, model, train_data, device):
    """Fits MAGIC's k-NN detector on the embeddings of the benign training nodes."""
    from pidsmaker.detection.training_methods.training_loop import (
        needs_attack_removal,
        remove_attacks_if_needed,
    )

    model.eval()
    s = time.time()
    embs = []
    for dataset in train_data:
        for g in dataset:
            has_attacks = needs_attack_removal(g, cfg)
            g = batch_to_device(g, device)
            if has_attacks:
                g = remove_attacks_if_needed(g, cfg)
            h, *_ = model.embed(g, inference=True)
            embs.append(magic_detection_embedding(model, h, cfg).float())
            del g
    x_train = torch.cat(embs)
    del embs
    node_eval = cfg.evaluation.node_evaluation
    detector = MagicKNNDetector(
        k=node_eval.magic_knn_k or 10,
        max_reference_size=node_eval.magic_max_reference_size or None,
        seed=cfg.training.seed or 0,
    ).fit(x_train)
    log(
        f"MAGIC k-NN detector fitted on {detector.num_reference_nodes:,} of {x_train.shape[0]:,} "
        f"benign training node embeddings ({detector.reference.shape[0]:,} distinct, "
        f"k={detector.k}, mean distance "
        f"{detector.mean_distance:.4f}) in {time.time() - s:.1f}s"
    )
    return detector


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

    magic_detector = None
    if cfg.evaluation.node_evaluation.threshold_method == "magic" and any(
        name == "test" for _, name in splits
    ):
        if train_data is None:
            raise ValueError("MAGIC detection requires the train split as k-NN reference set.")
        magic_detector = fit_magic_detector(cfg, model, train_data, device)

    val_score = 0.0
    peak_inference_cpu_mem = 0
    peak_inference_gpu_mem = 0
    tpb = []
    split2loss = {}

    for dataset, split_name in splits:
        desc = "Validation" if split_name == "val" else "Testing"

        cpu_mem = PeakCPUMemory().start()

        all_losses = []
        is_node_level = cfg._is_node_level and not cfg._is_hybrid_loss
        writer = None
        if not is_node_level:
            writer = EdgeLossWriter(
                cfg, os.path.join(cfg.training._edge_losses_dir, split_name, model_epoch_file)
            )
        for graphs in dataset:
            for g in log_tqdm(graphs, desc=desc, logging=logging):
                g = batch_to_device(g, device)

                s = time.time()
                if is_node_level:
                    losses = test_node_level(
                        data=g,
                        model=model,
                        split=split_name,
                        model_epoch_file=model_epoch_file,
                        cfg=cfg,
                        device=device,
                        magic_detector=magic_detector,
                    )
                    all_losses.append(np.asarray(losses, dtype=np.float64).ravel())
                else:
                    test_edge_level(
                        data=g,
                        model=model,
                        split=split_name,
                        model_epoch_file=model_epoch_file,
                        cfg=cfg,
                        device=device,
                        writer=writer,
                    )
                tpb.append(time.time() - s)
                del g
        if writer is not None:
            all_losses += [np.asarray(l, dtype=np.float64).ravel() for l in writer.close()]

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
