"""Storage of the per-epoch inference outputs (edge/node losses per time window).

Inference writes one file per batch/time window in
``{edge_losses_dir}/{split}/model_epoch_{i}/{start}~{end}{ext}``, for every epoch.
These used to be CSV files, which made them the largest output of a run after the
edge features. They are now stored as compressed `.npz` archives of typed columns
(about 9x smaller than the CSVs):

- float64 columns holding float32 values (e.g. losses computed in float32) are stored
  as float32, other floats as float64;
- integer columns are stored as int32 when the values fit, int64 otherwise.

`read_losses` always returns float64/int64 columns, i.e. the same DataFrame as
`pd.read_csv` on the former CSV files, so downstream code is unchanged. Legacy CSV
files remain readable.
"""

import os

import numpy as np
import pandas as pd

LOSS_EXT = ".npz"
_LEGACY_EXT = ".csv"
_EXTS = (LOSS_EXT, _LEGACY_EXT)

_I32 = np.iinfo(np.int32)


def _compact_column(values):
    a = np.asarray(values)
    if a.dtype == np.bool_:
        return a.astype(np.int8)
    if np.issubdtype(a.dtype, np.floating):
        if a.dtype == np.float64:
            a32 = a.astype(np.float32)
            # Only downcast when lossless (NaNs compare unequal, hence equal_nan)
            if np.array_equal(a32.astype(np.float64), a, equal_nan=True):
                return a32
        return a
    if np.issubdtype(a.dtype, np.integer):
        if a.size == 0 or (a.min() >= _I32.min and a.max() <= _I32.max):
            return a.astype(np.int32)
        return a.astype(np.int64)
    raise TypeError(f"Unsupported column dtype {a.dtype} in loss file")


def _expand_column(a):
    if np.issubdtype(a.dtype, np.floating):
        return a.astype(np.float64, copy=False)
    if np.issubdtype(a.dtype, np.integer):
        return a.astype(np.int64, copy=False)
    return a


def write_losses(out_dir, name, columns):
    """Writes the columns (dict name -> 1D array, or a DataFrame) of one loss file.

    Returns the written path.
    """
    if isinstance(columns, pd.DataFrame):
        columns = {c: columns[c].to_numpy() for c in columns.columns}
    path = os.path.join(out_dir, name + LOSS_EXT)
    # A CSV of the same window left by a previous run with the former format would be
    # read in addition to this file
    legacy = os.path.join(out_dir, name + _LEGACY_EXT)
    if os.path.exists(legacy):
        os.remove(legacy)
    np.savez_compressed(path, **{k: _compact_column(v) for k, v in columns.items()})
    return path


def read_losses(path, usecols=None):
    """Reads one loss file (.npz or legacy .csv) as a DataFrame of float64/int64 columns."""
    if path.endswith(_LEGACY_EXT):
        return pd.read_csv(path, usecols=usecols)
    with np.load(path) as f:
        names = f.files if usecols is None else [c for c in f.files if c in usecols]
        missing = set(usecols or ()) - set(f.files)
        if missing:
            raise ValueError(f"Columns {sorted(missing)} not found in {path}")
        return pd.DataFrame({c: _expand_column(f[c]) for c in names})


def is_loss_file(fname):
    return fname.endswith(_EXTS)


def strip_loss_ext(fname):
    """Returns the time window name ("start~end") of a loss file name."""
    for ext in _EXTS:
        if fname.endswith(ext):
            return fname[: -len(ext)]
    return fname


def compact_int_columns(df):
    """Losslessly stores the int64 columns of a DataFrame as int32 when the values fit.

    Used for the per-epoch `edge_scores_*.pkl` files, whose node IDs, types, time
    window indices and labels all fit in 32 bits.
    """
    df = df.copy(deep=False)
    for c in df.columns:
        if df[c].dtype == np.int64 and len(df) > 0:
            if df[c].min() >= _I32.min and df[c].max() <= _I32.max:
                df[c] = df[c].astype(np.int32)
    return df


_EPOCH_FILE_PREFIXES = ("edge_scores_", "result_", "scores_")


def _epochs_to_keep(cfg, metrics):
    keep = set()
    if not metrics or metrics.get("epoch") is None:
        return keep  # the evaluation's best epoch is unknown (e.g. cached): nothing is pruned
    keep.add(f"model_epoch_{int(metrics['epoch'])}")

    # Triage selects its own best epoch (highest MCC) from the per-epoch stats
    import torch

    pr_dir = cfg.evaluation._precision_recall_dir
    best_mcc = None
    if os.path.isdir(pr_dir):
        for f in os.listdir(pr_dir):
            if f.startswith("stats_model_epoch_") and f.endswith(".pth"):
                stats = torch.load(os.path.join(pr_dir, f), weights_only=False)
                mcc = stats.get("mcc") if isinstance(stats, dict) else None
                if mcc is not None and (best_mcc is None or mcc > best_mcc[0]):
                    best_mcc = (mcc, f[len("stats_") : -len(".pth")])
    if best_mcc is not None:
        keep.add(best_mcc[1])
    return keep


def prune_non_best_epochs(cfg, metrics):
    """Deletes the large per-epoch outputs of all epochs except the best one(s).

    Kept for every epoch: stats and plots. Deleted for the other epochs: the loss
    files of all splits, the evaluation pickles (edge_scores_, result_, scores_) and
    the saved model weights. The best epoch is the one selected by the evaluation,
    plus the one selected by triage (best MCC) when it differs.

    Re-running evaluation with another config afterwards only considers the kept epochs.
    """
    import shutil

    from pidsmaker.utils.utils import log

    keep = _epochs_to_keep(cfg, metrics)
    if not keep:
        log("Epoch pruning skipped: the best epoch is unknown.")
        return

    freed = 0

    def _rm(path):
        nonlocal freed
        if os.path.isdir(path):
            for root, _, files in os.walk(path):
                freed += sum(os.path.getsize(os.path.join(root, f)) for f in files)
            shutil.rmtree(path)
        elif os.path.isfile(path):
            freed += os.path.getsize(path)
            os.remove(path)

    losses_dir = cfg.training._edge_losses_dir
    if os.path.isdir(losses_dir):
        for split in os.listdir(losses_dir):
            split_dir = os.path.join(losses_dir, split)
            if not os.path.isdir(split_dir):
                continue
            for epoch_dir in os.listdir(split_dir):
                if epoch_dir.startswith("model_epoch_") and epoch_dir not in keep:
                    _rm(os.path.join(split_dir, epoch_dir))

    models_dir = getattr(cfg.training, "_trained_models_dir", None)
    if models_dir and os.path.isdir(models_dir):
        for f in os.listdir(models_dir):
            if f.startswith("model_epoch_") and f not in keep:
                _rm(os.path.join(models_dir, f))

    pr_dir = cfg.evaluation._precision_recall_dir
    if os.path.isdir(pr_dir):
        for f in os.listdir(pr_dir):
            for prefix in _EPOCH_FILE_PREFIXES:
                if f.startswith(prefix + "model_epoch_") and not f.endswith((".png", ".svg")):
                    epoch = os.path.splitext(f[len(prefix) :])[0]
                    if epoch not in keep:
                        _rm(os.path.join(pr_dir, f))

    log(f"Pruned per-epoch outputs except {sorted(keep)}: freed {freed / 1e9:.2f} GB")
