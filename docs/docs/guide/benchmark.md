---
title: Benchmark
description: Wall time, peak RAM, peak GPU memory and disk usage of the provenance and network intrusion detection systems of PIDSMaker on four DARPA TC datasets, a host of DARPA OpTC and LANL, measured on full runs from scratch.
---

# Benchmark

How long each system takes to run, and how much memory and disk it needs, on five provenance datasets and one network dataset, of increasing size. Every system runs end to end from scratch, with its default config: from reading the events in the database to the final metrics.
{ .lead }

## Results

One panel per dataset, all on the same scale within a tab, so bars compare across systems and datasets. When a few runs are far above the others, the axis is cut so that the other bars stay readable: the bars of these runs stop at the cut with a break mark, and hovering them gives their value. Time and disk are split by the [stages of the pipeline](pipeline.md#the-stages) that spent or wrote them; peak memory is a single value per run. Systems are in the order of the [Systems](systems.md) page, the most recent first. Hover a bar for its values; the [tables](#all-values) below list them all.

!!! note "Runs in progress"
    A few runs had not finished when this page was written, and are marked *pending*: Jbeil on TRACE_E3, and ProGrapher on THEIA_E5 and optc_h201.

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

- **Every run completes.** No system runs out of memory on any dataset, with 1 TB of RAM and a GPU of 80 GB.
- **Network systems are the cheapest.** LARES runs in 2 to 6 minutes on five of the six datasets, and Anomal-E, E-GraphSAGE, Argus and Euler in under 30 minutes outside TRACE_E3. They merge the repeated edges of each time window and use label-free features, so they skip the text embeddings that PIDSs compute for every entity.
- **Jbeil is the slowest system.** It trains on batches of 200 edges with a TGN memory, 10 epochs over every edge in time order: 53 minutes on CADETS_E3, and 2.5 to 5.7 hours on LANL, THEIA_E5, optc_h201 and CLEARSCOPE_E5. Training is 93% of its time.
- **Training dominates the other TGN systems.** Kairos, Orthrus and Velox spend 69 to 87% of their time training on THEIA_E5 and CLEARSCOPE_E5, where each epoch goes through every edge of the training graphs.
- **TRACE_E3 is the most expensive dataset.** Most systems take 34 to 113 minutes on it. OCR-APT takes 302 minutes, 188 of them evaluating its epochs, and ProGrapher 200 minutes, 177 of them embedding the time windows with graph2vec. It also needs the most RAM: 33 GB for the median system, 55 GB for Orthrus, Velox and ShadeWatcher, and 102 GB for NodLink. On the other datasets, no run needs more than 40 GB of RAM, and 66 of 78 stay under 15 GB.
- **A large GPU is not needed.** 85 of the 94 runs stay under 12 GB of GPU memory, and none needs more than 32 GB: the largest are E-GraphSAGE on THEIA_E5 (31 GB) and LARES, Argus and Euler on TRACE_E3 (24 to 28 GB).
- **R-Caid's pseudo-graphs need bounds.** R-Caid links each root, a file or a socket, to all its descendants. A process that reads thousands of files makes each of them the root of everything below it: up to 902M edges in a single time window of THEIA_E5. PIDSMaker merges the roots with the same label and keeps up to 16 roots per node, so R-Caid stays under 9 GB of RAM and GPU memory on every dataset but TRACE_E3, where it needs 42 GB of RAM.
- **Per-epoch evaluation outputs fill the disk.** The scores of every epoch are most of the disk usage of network systems (24 of the 30 GB of Euler, trained for 50 epochs, on THEIA_E5), and about a third of that of Orthrus and Velox. [`--keep_best_epoch_only`](pipeline.md#disk-usage) keeps only those of the reported epoch. NodLink's node embeddings take 13 GB on THEIA_E5.

### All values

=== "CADETS_E3"

    --8<-- "scripts/benchmark/table_cadets_e3.md"

=== "THEIA_E5"

    --8<-- "scripts/benchmark/table_theia_e5.md"

=== "CLEARSCOPE_E5"

    --8<-- "scripts/benchmark/table_clearscope_e5.md"

=== "TRACE_E3"

    --8<-- "scripts/benchmark/table_trace_e3.md"

=== "optc_h201"

    --8<-- "scripts/benchmark/table_optc_h201.md"

=== "LANL"

    --8<-- "scripts/benchmark/table_lanl.md"

### Time per stage

Minutes spent in each [stage of the pipeline](pipeline.md#the-stages), as logged by PIDSMaker at the end of a run (`Task times (s)`). Training includes the inference of the losses on the validation and test graphs after each epoch, and evaluation computes the metrics of every epoch.

??? note "CADETS_E3"

    --8<-- "scripts/benchmark/stages_cadets_e3.md"

??? note "THEIA_E5"

    --8<-- "scripts/benchmark/stages_theia_e5.md"

??? note "CLEARSCOPE_E5"

    --8<-- "scripts/benchmark/stages_clearscope_e5.md"

??? note "TRACE_E3"

    --8<-- "scripts/benchmark/stages_trace_e3.md"

??? note "optc_h201"

    --8<-- "scripts/benchmark/stages_optc_h201.md"

??? note "LANL"

    --8<-- "scripts/benchmark/stages_lanl.md"

## Setup

**Datasets.** Four DARPA TC datasets, one host of DARPA OpTC and the LANL authentications, a network dataset, all loaded as [preprocessed databases](datasets.md). LANL counts authentications and hosts:

| Dataset | Events | Nodes | Train / val / test days |
|---|--:|--:|---|
| CADETS_E3 | 36.5M | 2.68M | 7 / 1 / 4 |
| THEIA_E5 | 141.0M | 2.29M | 3 / 1 / 2 |
| CLEARSCOPE_E5 | 198.8M | 0.50M | 5 / 1 / 3 |
| TRACE_E3 | 284.0M | 44.5M | 7 / 1 / 4 |
| optc_h201 | 38.2M | 5.27M | 3 / 1 / 3 |
| LANL | 49.3M | 15,610 | 1 / 1 / 28 |

**Systems.** The 17 systems of the [Systems](systems.md) page, with their default config: `python pidsmaker/main.py <system> <dataset>`. Argus, LARES, E-GraphSAGE and Anomal-E use the attributes of events, which provenance datasets don't have, so they run with `--batching.edge_features=edge_type`, as described in [Running across families](systems.md#running-across-families). LANL, a network dataset, runs the NIDSs with the attributes of its authentications, and the PIDSs that run on network datasets: Velox, Orthrus, Kairos, MAGIC, OCR-APT and ShadeWatcher. Every run starts from an empty artifact folder: nothing is reused from a previous run.

**Hardware.** Three servers, each with 64 CPU threads (two 16-core CPUs), 1 TB of RAM and one NVIDIA A100 80 GB PCIe GPU. Servers A and B have AMD EPYC 7343 CPUs, and server C Intel Xeon Gold 6326 CPUs, which are slower on a single thread: runs bound by one CPU thread take longer on C, up to about twice as long for Jbeil. The `machine` column of [`results.csv`](../assets/benchmark/results.csv) gives the server of each run. Postgres 17 runs on the same server as the pipeline.

**Software.** PIDSMaker at commit `ec63214`, with its Postgres database backend, and PyTorch 2.1.2 with CUDA 12.1 and PyTorch Geometric 2.6.1, the versions of the [Docker image](../getting-started/installation.md) of PIDSMaker.

!!! warning "Runs shared their server"
    To run the 97 combinations in about a day, each server ran up to 2 of them at the same time, sharing its CPUs, GPU and database. Each run was limited to 16 threads (`OMP_NUM_THREADS=16`). Times can therefore be somewhat longer than on a dedicated server. Memory and disk usage do not depend on it. The database of server A is also used by other work: the runs that it slowed down were run again on B or C.

**What is measured.**

Wall time
:   From the start of `main.py` to the end of the run, including reading the database, the inference after each epoch, and the evaluation of every epoch.

Peak RAM
:   The highest memory of the run, sampled every second: the proportional set size (PSS) of the main process and all its children, summed. PSS splits the memory shared between processes, so the forked workers of graph construction are not counted twice.

Peak GPU memory
:   The memory reserved by PyTorch (`torch.cuda.max_memory_reserved`), without the ~0.5 GB of the CUDA context.

Disk
:   The size of all the outputs of the run once it ends: graphs, features, models, losses and the evaluation outputs of every epoch.

## Reproducing a measurement

`scripts/benchmark/measure.py` runs a system from scratch in an empty folder and measures it as above:

```shell
python scripts/benchmark/measure.py --out runs/kairos_theia kairos THEIA_E5
```

The run's log is written to `runs/kairos_theia/run.log`, and its measurements to `runs/kairos_theia/metrics.json`: wall time, the time and disk usage of each task, peak RAM and peak GPU memory. Its outputs are deleted once measured, unless `--keep` is given. Arguments after the dataset are passed to PIDSMaker.

The charts and tables of this page are generated from [`results.csv`](../assets/benchmark/results.csv), one row per run, by:

```shell
python docs/scripts/benchmark_charts.py
```
