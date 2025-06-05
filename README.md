[![Support Dev Containers](https://img.shields.io/static/v1?label=Dev%20Containers&message=Open&color=blue)](https://code.visualstudio.com/docs/devcontainers/create-dev-container)
[![Documentation](https://img.shields.io/badge/docs-online-pink.svg)](https://ubc-provenance.github.io/PIDSMaker/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15603122.svg)](https://doi.org/10.5281/zenodo.15603122)

# Sometimes Simpler is Better: A Comprehensive Analysis of State-of-the-Art Provenance-Based Intrusion Detection Systems

# Citation

If you use this work, please cite the following paper:
```
@inproceedings{jian2025,
	title={{ORTHRUS: Achieving High Quality of Attribution in Provenance-based Intrusion
	Detection Systems}},
	author={Jiang, Baoxiang and Bilot, Tristan  and El Madhoun, Nour and Al Agha, Khaldoun  and Zouaoui, Anis and Iqbal, Shahrear and Han, Xueyuan and Pasquier, Thomas},
	booktitle={Security Symposium (USENIX Sec'25)},
	year={2025},
	organization={USENIX}
}
```

# 🥷 PIDSMaker

The first framework designed to build and experiment with provenance-based intrusion detection systems (PIDSs) using deep learning architectures.
It provides a single codebase to run most recent state-of-the-arts systems and easily customize them to develop new variants.

**Currently supported PIDSs**:
- Orthrus (USENIX Sec'25): [ORTHRUS: Achieving High Quality of Attribution in Provenance-based Intrusion Detection Systems](https://www.usenix.org/system/files/conference/usenixsecurity25/sec25cycle1-prepub-103-jiang-baoxiang.pdf)
- R-Caid (IEEE S\&P'24): [R-CAID: Embedding Root Cause Analysis within Provenance-based Intrusion Detection](https://gangw.web.illinois.edu/rcaid-sp24.pdf)
- Flash (IEEE S\&P'24): [Flash: A Comprehensive Approach to Intrusion Detection via Provenance Graph Representation Learning](https://dartlab.org/assets/pdf/flash.pdf)
- Kairos (IEEE S\&P'24): [Kairos: Practical Intrusion Detection and Investigation using Whole-system Provenance](https://arxiv.org/pdf/2308.05034)
- Magic (USENIX Sec'24): [MAGIC: Detecting Advanced Persistent Threats via Masked Graph Representation Learning](https://www.usenix.org/system/files/usenixsecurity24-jia-zian.pdf)
- NodLink (NDSS'24): [NODLINK: An Online System for Fine-Grained APT Attack Detection and Investigation](https://arxiv.org/pdf/2311.02331)
- ThreaTrace (IEEE TIFS'22): [THREATRACE: Detecting and Tracing Host-Based Threats in Node Level Through Provenance Graph Learning](https://arxiv.org/pdf/2111.04333)

## Setup

### ⬇️ Clone the repo
```
git clone https://github.com/ubc-provenance/PIDSMaker.git
```

### ⏰ 10-min Docker Install with DARPA TC/OpTC Datasets

We have made the installation of DARPA TC/OpTC easy and fast, simply follow [these guidelines](docs/docs/ten-minute-install.md).

## Basic usage of the framework

Once you have a shell in the pids container, experiments can be run in multiple ways.

- Replace `SYSTEM` by `velox | orthrus | nodlink | threatrace | kairos | rcaid | flash`.
- Replace `DATASET` by `CLEARSCOPE_E3 | CADETS_E3 | THEIA_E3 | CLEARSCOPE_E5 | THEIA_E5 | optc_h201 | optc_h501 | optc_h051`.

1. Run in the shell, no W&B:
    ```shell
    python pidsmaker/main.py SYSTEM DATASET --tuned
    ```

2. Run in the shell, monitored to W&B:
    ```shell
    python pidsmaker/main.py SYSTEM DATASET --tuned --wandb
    ```

3. Run in background, monitored to W&B (ideal for multiple parallel runs):
    ```shell
    ./run.sh SYSTEM DATASET --tuned
    ```
    You can still watch the logs in your shell using `tail -f nohup.out`

## License

See [licence](LICENSE).
