<p align="center">
  <img width="80%" src="./assets/pidsmaker_title.png" alt="PIDSMAKER logo"/>
  </p>

# PIDSMaker
The first framework designed to build and experiment with provenance-based intrusion detection systems (PIDSs) using deep learning architectures.
It provides a single codebase to run most recent state-of-the-art systems and easily customize them to develop new variants.

## Purpose

PIDSMaker is an open-source framework designed to be collaboratively developed and maintained by the security research community. It was born out of the observation that recent papers in top-tier security venues often evaluate on the same datasets but differ in labeling strategies and in the implementation of baseline methods.

Until now, no standardized open-source framework has existed to facilitate fair comparisons.
PIDSMaker addresses this gap by providing the following key features:

1.	**Consistent evaluation** and benchmarking of SOTA baselines using unified datasets, labeling strategies, and reference implementations.
2.	A modular testbed of existing components extracted from published systems, enabling experimentation and the **discovery of improved variants**.
3.	A centralized repository where authors can contribute and **share code for new systems**, ensuring fair and reproducible benchmarking.


## What is system provenance?

**System provenance** is a detailed record of all activities occurring on a computer system. Operating systems like Linux and Windows can be configured to capture these events through audit frameworks (e.g., Linux Audit, ETW on Windows).

Provenance data captures:

- **Process execution**: Which programs ran, with what arguments
- **File operations**: Reads, writes, creates, deletes
- **Network activity**: Connections, data transfers
- **Inter-process communication**: Pipes, signals, shared memory

## Provenance graphs

Raw provenance logs are transformed into a **provenance graph**—a directed graph where:

| Element | Description | Examples |
|---------|-------------|----------|
| **Nodes** | System entities | Processes, files, network sockets |
| **Edges** | Interactions between entities | Process reads file, process connects to socket |

```
┌─────────┐   READ    ┌─────────┐
│ Process │ ────────▶ │  File   │
│  nginx  │           │ config  │
└─────────┘           └─────────┘
     │
     │ CONNECT
     ▼
┌─────────┐
│ Socket  │
│ :8080   │
└─────────┘
```

### Node types

PIDSMaker uses three primary node types:

| Type | Description | Attributes |
|------|-------------|------------|
| `subject` | Processes/threads | Command line, executable path |
| `file` | Files and directories | File path |
| `netflow` | Network connections | IP address, port |

### Edge types

Edges represent system calls or events. PIDSMaker uses 10 edge types:

| Edge Type | Description |
|-----------|-------------|
| `EVENT_READ` | Process reads from file |
| `EVENT_WRITE` | Process writes to file |
| `EVENT_OPEN` | Process opens file |
| `EVENT_EXECUTE` | Process executes file |
| `EVENT_CONNECT` | Process connects to network |
| `EVENT_RECVFROM` | Process receives network data |
| `EVENT_RECVMSG` | Process receives network message |
| `EVENT_SENDTO` | Process sends network data |
| `EVENT_SENDMSG` | Process sends network message |
| `EVENT_CLONE` | Process creates child process (fork) |

## How PIDSs detect attacks

PIDSs use **self-supervised learning** to detect intrusions:

### 1. Training phase (benign data only)

```
┌─────────────────────────────────────────────────────────┐
│                    Training Pipeline                    │
├─────────────────────────────────────────────────────────┤
│  Benign        Graph           GNN          Learn       │
│  Provenance ─▶ Construction ─▶ Encoder ─▶   Normal      │
│  Logs                                       Behavior    │
└─────────────────────────────────────────────────────────┘
```

The model learns patterns of normal system behavior:
- Which processes typically access which files
- Normal network communication patterns
- Typical sequences of system calls

### 2. Detection phase (test data with attacks)

```
┌─────────────────────────────────────────────────────────┐
│                   Detection Pipeline                    │
├─────────────────────────────────────────────────────────┤
│  Test          Compute         Compare to    Flag       │
│  Provenance ─▶ Predictions ─▶  Threshold  ─▶ Anomalies  │
└─────────────────────────────────────────────────────────┘
```

At inference time:
1. The trained model makes predictions about system behavior
2. Prediction errors (reconstruction loss) indicate anomalies
3. Entities with errors above a threshold are flagged as malicious

## Graph Neural Networks (GNNs)

PIDSMaker mainly uses **Graph Neural Networks** to learn from provenance graphs. GNNs are neural networks designed to operate on graph-structured data.

### Message passing

GNNs work through **message passing**: each node aggregates information from its neighbors to update its representation.

```
     Round 1                    Round 2
  ┌───┐                      ┌───┐
  │ A │◄── neighbor info     │ A │◄── 2-hop info
  └───┘                      └───┘
   ▲ ▲                        ▲ ▲
  ╱   ╲                      ╱   ╲
┌───┐ ┌───┐               ┌───┐ ┌───┐
│ B │ │ C │               │ B │ │ C │
└───┘ └───┘               └───┘ └───┘
```

After multiple rounds, each node's embedding captures its local neighborhood structure.

PIDSMaker provides several GNN encoders (GraphSAGE, GAT, TGN, etc.) and self-supervised objectives (edge type prediction, node type prediction, feature reconstruction, etc.). See the [Arguments](config/encoders.md) section for the full list of available components.

## The 7-stage pipeline

PIDSMaker structures detection into seven stages, from graph construction to evaluation and optional triage. See the [Pipeline](pipeline.md) page for detailed documentation of each stage.

## Citing the Framework

If you use this framework, please cite the following paper:
```
@inproceedings{bilot2025simpler,
	title={{Sometimes Simpler is Better: A Comprehensive Analysis of State-of-the-Art Provenance-Based Intrusion Detection Systems}},
	author={Bilot, Tristan and Jiang, Baoxiang and  Li, Zefeng and  El Madhoun, Nour and Al Agha, Khaldoun and Zouaoui, Anis and Pasquier, Thomas},
	booktitle={Security Symposium (USENIX Sec'25)},
	year={2025},
	organization={USENIX}
}
```
