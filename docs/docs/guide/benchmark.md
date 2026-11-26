---
title: Benchmark
description: Wall time, peak RAM, peak GPU memory and disk usage of the provenance and network intrusion detection systems of PIDSMaker on six datasets, measured on full runs from scratch.
---

# Benchmark

How long each system takes to run, and how much memory and disk it needs, on five provenance datasets and one network dataset. Every system runs end to end from scratch, with its default config.
{ .lead }

## Results

One panel per dataset, on one scale per tab; the few bars far above the others are cut, with a break mark. Time and disk are split by [stage](pipeline.md#the-stages). Hover a bar for its values; all of them are in [`results.csv`](../assets/benchmark/results.csv).

!!! note "Runs in progress"
    ProGrapher on optc_h201 is still running, and marked *pending*.

=== "Time"

    <div class="pm-bench" markdown="0">
    --8<-- "scripts/benchmark/time_stacked_1.svg"
    --8<-- "scripts/benchmark/time_stacked_2.svg"
    </div>

=== "RAM"

    <div class="pm-bench" markdown="0">
    --8<-- "scripts/benchmark/ram_bars_1.svg"
    --8<-- "scripts/benchmark/ram_bars_2.svg"
    </div>

=== "GPU memory"

    <div class="pm-bench" markdown="0">
    --8<-- "scripts/benchmark/gpu_bars_1.svg"
    --8<-- "scripts/benchmark/gpu_bars_2.svg"
    </div>

=== "Disk"

    <div class="pm-bench" markdown="0">
    --8<-- "scripts/benchmark/disk_stacked_1.svg"
    --8<-- "scripts/benchmark/disk_stacked_2.svg"
    </div>

### Takeaways

- **Every run completes**, with 1 TB of RAM and an 80 GB GPU.
- **A large GPU is not needed:** no run uses more than 32 GB of GPU memory, and 87 of 96 stay under 12 GB.
- **Network systems other than Jbeil are the cheapest:** under 30 minutes outside TRACE_E3. They merge repeated edges and skip text embeddings.
- **Jbeil is the slowest:** 2.5 to 7.8 hours on all datasets but CADETS_E3, 93% of it training its TGN on batches of 200 edges. Training is also 69 to 87% of the time of Kairos, Orthrus and Velox.
- **ProGrapher's time goes to graph2vec,** 77 to 98% of it: 174 minutes on THEIA_E5 and 200 on TRACE_E3.
- **TRACE_E3 is the most expensive dataset:** 34 to 113 minutes for most systems, 28 GB of RAM for the median one and up to 102 GB (NodLink). Elsewhere, no run needs more than 40 GB of RAM.
- **Per-epoch outputs fill the disk:** 24 of the 30 GB of Euler on THEIA_E5. [`--keep_best_epoch_only`](pipeline.md#disk-usage) keeps only those of the reported epoch.

## Setup

| Dataset | Events | Nodes | Train / val / test days |
|---|--:|--:|---|
| CADETS_E3 | 36.5M | 2.68M | 7 / 1 / 4 |
| THEIA_E5 | 141.0M | 2.29M | 3 / 1 / 2 |
| CLEARSCOPE_E5 | 198.8M | 0.50M | 5 / 1 / 3 |
| TRACE_E3 | 284.0M | 44.5M | 7 / 1 / 4 |
| optc_h201 | 38.2M | 5.27M | 3 / 1 / 3 |
| LANL | 49.3M | 15,610 | 1 / 1 / 28 |

**Systems.** The 17 systems of the [Systems](systems.md) page, run with `python pidsmaker/main.py <system> <dataset>` from an empty artifact folder. On provenance datasets, Argus, LARES, E-GraphSAGE and Anomal-E use `--batching.edge_features=edge_type` (see [Running across families](systems.md#running-across-families)). LANL runs the NIDSs and the PIDSs that support network datasets.

**Hardware and software.** Three servers with 64 CPU threads, 1 TB of RAM and an NVIDIA A100 80 GB GPU; A and B have AMD EPYC 7343 CPUs, C slower Intel Xeon Gold 6326 CPUs (the `machine` column of `results.csv`). PIDSMaker at commit `ec63214` with Postgres 17 on the same server, PyTorch 2.1.2, CUDA 12.1 and PyTorch Geometric 2.6.1. Since version 5.0.0, datasets are read from Parquet files instead of Postgres, with identical results and a faster graph construction (e.g. 80 s instead of 124 s for Orthrus on THEIA_E5); the other stages are unchanged.

!!! warning "Runs shared their server"
    Each server ran up to 2 runs at a time, with 16 threads each (`OMP_NUM_THREADS=16`), so times can be longer than on a dedicated server, and longer on C. Memory and disk usage do not depend on it.

| Metric | Measured as |
|---|---|
| Wall time | From the start of `main.py` to the end of the run |
| Peak RAM | The highest proportional set size (PSS) of the run's processes, summed, sampled every second |
| Peak GPU memory | The memory reserved by PyTorch (`torch.cuda.max_memory_reserved`) |
| Disk | The size of all the outputs of the run once it ends |

## Reproducing a measurement

```shell
python scripts/benchmark/measure.py --out runs/kairos_theia kairos THEIA_E5
python docs/scripts/benchmark_charts.py  # regenerates the charts from results.csv
```

The first command runs a system from scratch and writes its measurements to `runs/kairos_theia/metrics.json`.
