# PIDSMaker

The first framework designed to build and experiment with provenance-based intrusion detection systems (PIDSs) using deep learning architectures. 
It provides a single codebase to run most recent state-of-the-art systems and easily customize them to develop new variants.

## Purpose

🥷 PIDSMaker is an open-source framework designed to be collaboratively developed and maintained by the security research community. It was born out of the observation that recent papers in top-tier security venues often evaluate on the same datasets but differ in labeling strategies and in the implementation of baseline methods.

Until now, no standardized open-source framework has existed to facilitate fair comparisons.
PIDSMaker addresses this gap by providing the following key features:

1.	**Consistent evaluation** and benchmarking of SOTA baselines using unified datasets, labeling strategies, and reference implementations.
2.	A modular testbed of existing components extracted from published systems, enabling experimentation and the **discovery of improved variants**.
3.	A centralized repository where authors can contribute and **share code for new systems**, ensuring fair and reproducible benchmarking.

## Supported PIDSs

- Velox (USENIX Sec'25): Sometimes Simpler is Better: A Comprehensive Analysis of State-of-the-Art Provenance-Based Intrusion Detection Systems
- Orthrus (USENIX Sec'25): [ORTHRUS: Achieving High Quality of Attribution in Provenance-based Intrusion Detection Systems](https://tfjmp.org/publications/2025-usenixsec.pdf)
- R-Caid (IEEE S\&P'24): [R-CAID: Embedding Root Cause Analysis within Provenance-based Intrusion Detection](https://gangw.web.illinois.edu/rcaid-sp24.pdf)
- Flash (IEEE S\&P'24): [Flash: A Comprehensive Approach to Intrusion Detection via Provenance Graph Representation Learning](https://dartlab.org/assets/pdf/flash.pdf)
- Kairos (IEEE S\&P'24): [Kairos: Practical Intrusion Detection and Investigation using Whole-system Provenance](https://arxiv.org/pdf/2308.05034)
- Magic (USENIX Sec'24): [MAGIC: Detecting Advanced Persistent Threats via Masked Graph Representation Learning](https://www.usenix.org/system/files/usenixsecurity24-jia-zian.pdf)
- NodLink (NDSS'24): [NODLINK: An Online System for Fine-Grained APT Attack Detection and Investigation](https://arxiv.org/pdf/2311.02331)
- ThreaTrace (IEEE TIFS'22): [THREATRACE: Detecting and Tracing Host-Based Threats in Node Level Through Provenance Graph Learning](https://arxiv.org/pdf/2111.04333)
