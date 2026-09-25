# ThreaTrace Ground Truth

Ground truth as published by **ThreaTrace** (`github.com/threaTrace-detector/threaTrace`,
`groundtruth/*.txt`), provided here as an alternative to the `orthrus` and `reapr`
ground truths. Select it with `--evaluation.ground_truth_version threatrace` on
the CLI (or `evaluation.ground_truth_version: threatrace` in a YAML config).

## Datasets

| Dataset | File | Unique nodes | Lines |
|---|---|---|---|
| CADETS_E3 | `E3-CADETS/ground_truth.txt` | 12,858 | 12,858 |
| THEIA_E3 | `E3-THEIA/ground_truth.txt` | 25,358 | 25,363 |
| TRACE_E3 | `E3-TRACE/ground_truth.txt` | 68,172 | 68,265 |
| FIVEDIRECTIONS_E3 | `E3-FIVEDIRECTIONS/ground_truth.txt` | 762 | 18,420 |

The upstream files (all DARPA TC E3) contain duplicate lines; FIVEDIRECTIONS in
particular is 18,420 lines but only 762 unique UUIDs. The loader deduplicates.

ThreaTrace only releases ground truth for these four E3 datasets. CLEARSCOPE (E3) and
all E5 datasets have no ThreaTrace ground truth and must use `orthrus`/`reapr`.

## Format

A single flat list of node UUIDs, one per line — no labels, no per-attack split, no
time windows:

```
9EF37E2E-3E80-11E8-A5CB-3FA3753A265A
A4DD7C60-3E80-11E8-A5CB-3FA3753A265A
...
```

This differs from `orthrus`/`reapr`, which provide one CSV per attack
(`uuid, label, node_id`) with an explicit time window per attack.

## Labeling philosophy vs. orthrus

ThreaTrace labels the **entire attack neighborhood**, whereas `orthrus` labels only the
precise attack-chain nodes. The two are an order of magnitude apart and overlap little:

| Dataset | orthrus attack nodes | ThreaTrace nodes | orthrus ∩ ThreaTrace | ThreaTrace-only |
|---|---|---|---|---|
| CADETS_E3 | 76 (4 attacks) | 12,858 | 21 (28%) | 12,837 |
| THEIA_E3 | 118 (2 attacks) | 25,358 | 25 (21%) | 25,333 |

Per orthrus attack, the fraction whose nodes appear in the ThreaTrace ground truth:

| Dataset | attack (day) | orthrus nodes | in ThreaTrace |
|---|---|---|---|
| CADETS_E3 | Nginx_Backdoor (04-06) | 8 | 0% |
| CADETS_E3 | Nginx_Backdoor (04-11)¹ | 4 | 0% |
| CADETS_E3 | Nginx_Backdoor (04-12) | 43 | 44% |
| CADETS_E3 | Nginx_Backdoor (04-13) | 24 | 17% |
| THEIA_E3 | Browser_Extension_Drakon (04-12) | 61 | 18% |
| THEIA_E3 | Firefox_Backdoor_Drakon (04-10) | 58 | 26% |

¹ Commented out in the dataset config (not used by velox). THEIA's two phishing attacks
are likewise commented out (one failed, one is network-only).

The 04-06 and 04-11 CADETS attacks have **no** overlap with the ThreaTrace ground truth
at the node level; ThreaTrace instead contributes ~12–25k additional
neighborhood nodes not present in orthrus.

## How velox consumes it

ThreaTrace has no per-attack split, so it is loaded as a **single combined attack** over
the dataset's full test period (`test_dates` → one `[start, end]` window). UUIDs absent
from the constructed graph are skipped and counted in the log. Loading is otherwise
transparent to the pipeline (`pidsmaker/utils/labelling.py`).

Mapped node counts (nodes whose UUID exists in the constructed graph):

| Dataset | Unique UUIDs | Mapped to graph | Absent |
|---|---|---|---|
| CADETS_E3 | 12,858 | 12,851 | 7 |
| THEIA_E3 | 25,358 | 25,354 | 4 |
| TRACE_E3 | 68,172 | 68,086 | 96 |
| FIVEDIRECTIONS_E3 | 762 | 424 | 338 |

velox has been run end-to-end (construction → training → evaluation) against the
ThreaTrace ground truth on CADETS_E3, THEIA_E3, and FIVEDIRECTIONS_E3. TRACE_E3's
ground truth loads correctly (68,086 mapped), but its full velox run needs more RAM
than a 94 GB host provides (construction expands 283M events / 39M process nodes
beyond memory), so run it on higher-memory hardware.

## Usage

```bash
python pidsmaker/main.py velox CADETS_E3 --evaluation.ground_truth_version threatrace
```

Requires the dataset's database to be constructed (node tables present). CADETS_E3,
THEIA_E3, and FIVEDIRECTIONS_E3 are ready and run end-to-end. TRACE_E3's database is
also loaded, but a full run needs more RAM than a 94 GB host (283M events, 39M process
nodes) — run it on higher-memory hardware.

## Verifying a run

At the evaluation stage, `compute_tw_labels` logs one line per time window plus a summary
of how many distinct ground-truth nodes were labeled:

```
7 ground-truth UUIDs not present in the graph (skipped)
Computing time-window labels...
TW 0   -> 7 malicious nodes + 3828 malicious edges
...
TW 231 -> 12831 malicious nodes + 32304 malicious edges
...
Total distinct malicious nodes across time windows: 12851 / 12851 ground-truth nodes
```

The final `distinct / ground-truth` line is the quick correctness check: equal numbers
mean every in-graph GT node was labeled. It also distinguishes ThreaTrace from orthrus —
ThreaTrace spans the whole test period (hundreds of `TW k ->` lines), orthrus only the
attack windows (a handful). Expected distinct counts:

| Dataset | distinct / GT |
|---|---|
| CADETS_E3 | 12,851 / 12,851 |
| THEIA_E3 | 25,354 / 25,354 |
| FIVEDIRECTIONS_E3 | ≤424 / 424 |
| TRACE_E3 | 68,086 / 68,086 |
