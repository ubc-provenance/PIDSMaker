import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from pidsmaker.factory import node_scorer_factory
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
    scorer,
):
    model.eval()

    start_time = int(data.t[0])
    end_time = int(data.t[-1])

    validation = split == "val"

    data = batch_to_device(data, device)

    results = model(data, inference=True, validation=validation)
    loss = results["loss"]
    losses = loss.float().cpu().numpy()
    n_id = to_original_node_ids(getattr(data, "original_n_id_tgn", getattr(data, "original_n_id")))
    n_id_np = n_id.cpu().numpy()

    node_list = scorer.columns(model, data, results, split, n_id_np, losses.astype(float))

    time_interval = (
        ns_time_to_datetime_US(start_time, timezone=cfg.dataset.timezone)
        + "~"
        + ns_time_to_datetime_US(end_time, timezone=cfg.dataset.timezone)
    )

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

    scorer = node_scorer_factory(cfg)
    scorer.prepare(model, train_data, [name for _, name in splits], device)

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
                        scorer=scorer,
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
                log(f"epoch {epoch:02d}  val    loss {mean_loss:.4f}")
        else:
            if logging:
                log(f"epoch {epoch:02d}  test   loss {mean_loss:.4f}")

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
