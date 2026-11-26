"""ThreaTrace's multi-model training (Wang et al., TIFS 2022, `train_darpatc.py`).

A first GraphSAGE node-type classifier is trained on all the training nodes. Then, in rounds:
the nodes the current model explains (type predicted with a top-1 / top-2 probability ratio of
at least `ratio_threshold`) are removed from the training set and the model is saved as a
submodel; the model then keeps training on the remaining nodes only, until its mean loss is
below `retrain_loss_threshold`. Training stops when all nodes are explained, or after
`max_rounds_without_progress` consecutive rounds that explain no new node.

At inference, a node is anomalous when no submodel explains it (see
`threatrace_detection.threatrace_anomaly_score`).

Training nodes are the (time window, node) pairs of the training graphs.
"""

import copy
from time import perf_counter as timer

import torch
import torch.nn as nn

from pidsmaker.detection.threatrace_detection import threatrace_log_ratio
from pidsmaker.factory import build_model, optimizer_factory
from pidsmaker.objectives.predict_node_type import NODE_LOSS_MASK_KEY
from pidsmaker.tasks.batching import get_preprocessed_graphs
from pidsmaker.utils.utils import PeakCPUMemory, get_device, log, log_start, log_tqdm, set_seed

from . import inference_loop
from .training_loop import to_device_without_attacks


class ThreaTraceEnsemble(nn.Module):
    """Runs the model with the weights of each submodel. Returns the outputs of all the
    submodels in `outs` (the other results are those of the first submodel)."""

    def __init__(self, model, state_dicts):
        super().__init__()
        self.model = model
        self.state_dicts = state_dicts

    def forward(self, batch, inference=False, validation=False):
        results, outs = None, []
        for state_dict in self.state_dicts:
            self.model.load_state_dict(state_dict)
            res = self.model(batch, inference=inference, validation=validation)
            outs.append(res["out"])
            results = results or res
        results["outs"] = outs
        return results

    def get_val_ap(self):
        return self.model.get_val_ap()

    def to_device(self, device):
        self.model.to_device(device)
        self.state_dicts = [{k: v.to(device) for k, v in sd.items()} for sd in self.state_dicts]
        return self


def train_epoch(model, optimizer, graphs, masks, cfg, device):
    """One epoch on the graphs with nodes left to explain. Returns the mean loss per node."""
    model.train()
    tot_loss, tot_nodes = 0.0, 0
    for g, mask in log_tqdm(list(zip(graphs, masks)), "Training"):
        num_nodes = int(mask.sum())
        if num_nodes == 0:
            continue
        g = to_device_without_attacks(g, cfg, device)
        setattr(g, NODE_LOSS_MASK_KEY, mask.to(device))
        optimizer.zero_grad()
        loss = model(g)["loss"]
        loss.backward()
        optimizer.step()
        tot_loss += loss.item() * num_nodes
        tot_nodes += num_nodes
    return tot_loss / max(tot_nodes, 1)


@torch.no_grad()
def explained_nodes(model, graphs, masks, cfg, device):
    """For each graph, the nodes still to explain that the model explains."""
    model.eval()
    log_thr = torch.log(torch.tensor(cfg.training.threatrace.ratio_threshold)).item()
    explained = []
    for g, mask in zip(graphs, masks):
        if not mask.any():
            explained.append(torch.zeros_like(mask))
            continue
        g = to_device_without_attacks(g, cfg, device)
        out = model(g, inference=True)["out"]
        log_ratio = threatrace_log_ratio(out, g.node_type.argmax(1))
        explained.append((log_ratio >= log_thr).cpu() & mask)
    return explained


def train_submodels(model, optimizer, graphs, cfg, device):
    """Trains the successive submodels. Returns their state dicts."""
    tt_cfg = cfg.training.threatrace
    masks = [torch.ones(g.node_type.shape[0], dtype=torch.bool) for g in graphs]
    total = sum(int(m.sum()) for m in masks)

    for epoch in range(cfg.training.num_epochs):
        loss = train_epoch(model, optimizer, graphs, masks, cfg, device)
        log(f"[submodel 0 | epoch {epoch:02d}] Mean loss: {loss:.4f}")

    submodels, rounds_without_progress = [], 0
    while len(submodels) < tt_cfg.max_submodels:
        explained = explained_nodes(model, graphs, masks, cfg, device)
        num_explained = sum(int(e.sum()) for e in explained)
        masks = [m & ~e for m, e in zip(masks, explained)]
        num_left = sum(int(m.sum()) for m in masks)
        log(
            f"Submodel {len(submodels)}: explains {num_explained} nodes, "
            f"{num_left}/{total} training nodes left"
        )

        rounds_without_progress = 0 if num_explained > 0 else rounds_without_progress + 1
        if rounds_without_progress >= tt_cfg.max_rounds_without_progress:
            break
        if num_explained > 0:
            submodels.append(copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()}))
            if num_left == 0 or len(submodels) >= tt_cfg.max_submodels:
                break

        for epoch in range(tt_cfg.max_retrain_epochs):
            loss = train_epoch(model, optimizer, graphs, masks, cfg, device)
            log(f"[submodel {len(submodels)} | epoch {epoch:02d}] Mean loss on nodes left: {loss:.4f}")
            if loss < tt_cfg.retrain_loss_threshold:
                break

    if not submodels:  # no node explained at all: keep the last model
        submodels.append(copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()}))
    log(f"Trained {len(submodels)} submodels")
    return submodels


def main(cfg):
    log_start(__file__)
    device = get_device(cfg)
    use_cuda = device == torch.device("cuda")
    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device=device)
    cpu_mem = PeakCPUMemory().start()

    train_data, val_data, test_data, max_node_num = get_preprocessed_graphs(cfg)
    log("Graph data loaded.")
    set_seed(cfg, seed=cfg.training.seed)

    model = build_model(
        data_sample=train_data[0][0], device=device, cfg=cfg, max_node_num=max_node_num,
        all_data=(val_data, test_data),
    )
    optimizer = optimizer_factory(cfg, parameters=set(model.parameters()))
    graphs = [g for dataset in train_data for g in dataset]

    start = timer()
    submodels = train_submodels(model, optimizer, graphs, cfg, device)
    gpu_mem = torch.cuda.max_memory_allocated(device=device) / (1024**3) if use_cuda else 0.0
    log(
        f"Training finished in {timer() - start:.1f}s - GPU memory: {gpu_mem:.2f} GB | "
        f"CPU memory: {cpu_mem.peak_gb():.2f} GB"
    )

    # The evaluation selects among the evaluated epochs: only the final ensemble is evaluated
    ensemble = ThreaTraceEnsemble(model, submodels)
    test_stats = inference_loop.main(
        cfg=cfg, model=ensemble, val_data=val_data, test_data=test_data, epoch=0, split="all",
    )
    return test_stats["val_score"]
