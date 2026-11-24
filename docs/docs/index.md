---
title: Home
description: PIDSMaker is an open-source framework to build, benchmark and extend graph-based intrusion detection systems on provenance and network data.
hide:
  - navigation
  - toc
---

<div class="pm-hero" markdown>

<img class="pm-hero__logo" src="assets/pidsmaker_new.png" alt="PIDSMaker logo">

<div markdown>

<h1>PIDS<span class="pm-maker">Maker</span></h1>

<p class="pm-hero__tagline">
The open-source framework to <strong>build, benchmark and extend graph-based intrusion detection systems</strong>, on host provenance and network data.
State-of-the-art provenance-based and network IDSs, their datasets, and one modular pipeline shared by both, all in a single codebase.
</p>

<div class="pm-hero__actions" markdown>

[:lucide-rocket: Get started](getting-started/installation.md){ .md-button .md-button--primary }
[:lucide-terminal: Quickstart](getting-started/quickstart.md){ .md-button }
[:fontawesome-brands-github: GitHub](https://github.com/ubc-provenance/PIDSMaker){ .md-button }
[:fontawesome-brands-discord: Discord](https://discord.gg/jecmkTABUg){ .md-button }

</div>

</div>

</div>

<style>.md-content__button { display: none; }</style>

<div class="pm-stats">
  <div class="pm-stat"><strong>15</strong><span>intrusion detection systems reimplemented</span></div>
  <div class="pm-stat"><strong>17</strong><span>datasets</span></div>
  <div class="pm-stat"><strong>7</strong><span>configurable pipeline stages</span></div>
  <div class="pm-stat"><strong>15+</strong><span>pretrained encoders</span></div>
</div>

## One framework, two views

<div class="grid cards pm-family-cards" markdown>

-   :lucide-cpu:{ .lg .middle } __Provenance__

    ---

    System events inside each host: processes, files and netflows. Provenance-based IDSs (Velox, Orthrus, Kairos, MAGIC...) on DARPA TC, OpTC, ATLASv2 and CARBANAKv2, detecting the entities of an attack.

    [:octicons-arrow-right-24: Provenance graphs](guide/background.md#provenance-graphs)

-   :lucide-network:{ .lg .middle } __Network__

    ---

    Authentications and flows between hosts. Network IDSs (Euler, Argus, Jbeil, LARES...) on LANL and the network flows of OpTC, detecting the malicious events of lateral movement.

    [:octicons-arrow-right-24: Network graphs](guide/background.md#network-graphs)

</div>

Both go through the same seven stages and share the same components, so most systems [run on either kind of data](guide/systems.md#running-across-families), and a new idea can be tested on both.

## What's new

<div class="grid cards" markdown>

-   :lucide-radar:{ .lg .middle } __Network intrusion detection__

    ---

    Network IDSs (Euler, Argus, Jbeil, LARES, E-GraphSAGE and Anomal-E) and network datasets (LANL and the network flows of OpTC), in the same pipeline as the provenance systems. Systems of both families run on network and provenance datasets.

    [:octicons-arrow-right-24: Running across families](guide/systems.md#running-across-families)

-   :lucide-gauge:{ .lg .middle } __Faster, leaner pipeline__

    ---

    Same results, in a fraction of the time and memory: Orthrus on THEIA_E5 runs end to end in ~61 min instead of ~3.4 h, with 16.6 GB of peak RAM instead of 370 GB. Graphs are ~5x smaller on disk.

    [:octicons-arrow-right-24: Performance](guide/pipeline.md#performance)

</div>

## Why PIDSMaker?

Papers at top security venues often evaluate on the same datasets, but each uses its own labels and its own implementation of the baselines, so their results can't be compared fairly.
PIDSMaker gives the community **one framework**, maintained in the open, that fixes this.

<div class="grid cards" markdown>

-   :lucide-scale:{ .lg .middle } __Fair benchmarking__

    ---

    State-of-the-art baselines evaluated with the same datasets, labels and reference implementations.

-   :lucide-boxes:{ .lg .middle } __Modular components__

    ---

    The components of published systems, provenance-based and network, extracted into a modular testbed. Mix and match them to **discover better variants**.

-   :lucide-git-pull-request:{ .lg .middle } __Shared code__

    ---

    A central place where authors contribute their systems, so new work can be benchmarked reproducibly.

</div>

## Up and running in three steps

=== ":lucide-download: 1. Install"

    Download a preprocessed dataset, load it into Postgres and open a shell in the `pids` container ([full guide](getting-started/installation.md)). We recommend one of two setups:

    === ":fontawesome-brands-linux: Linux + :simple-nvidia: NVIDIA GPU"

        A server or workstation with Docker and the [NVIDIA Container Toolkit](getting-started/installation.md#2-enable-gpu-support-in-docker), for all datasets.

        ```shell
        git clone https://github.com/ubc-provenance/PIDSMaker.git && cd PIDSMaker
        ./download_datasets.sh cadets_e3 YOUR_GOOGLE_ACCESS_TOKEN  # (1)!
        cp .env.local .env                                          # (2)!
        docker compose -p postgres -f compose-postgres.yml up -d --build
        docker compose -p postgres exec postgres /scripts/load_dumps.sh
        docker compose -f compose-pidsmaker.yml up -d --build
        docker compose exec pids bash
        ```

        1.  See [how to get an access token](getting-started/installation.md#provenance-datasets). Dumps are saved to `./data`. The network datasets are [built from the files of LARES](getting-started/from-scratch.md) for now.
        2.  Set `INPUT_DIR` to the absolute path of `./data` in `.env`.

    === ":fontawesome-brands-apple: macOS (Apple Silicon)"

        A Mac with [Docker Desktop](https://docs.docker.com/desktop/setup/install/mac-install/), on CPU, for most datasets: 32 GB of RAM is ideal ([details](getting-started/installation.md#macos)).

        ```shell
        git clone https://github.com/ubc-provenance/PIDSMaker.git && cd PIDSMaker
        ./download_datasets.sh cadets_e3 YOUR_GOOGLE_ACCESS_TOKEN  # (1)!
        cp .env.local .env && echo "VIZ_PORT=5001" >> .env          # (2)!
        plutil -replace MemoryMiB -integer 24576 ~/Library/Group\ Containers/group.com.docker/settings-store.json
        docker desktop restart                                      # (3)!
        docker compose -p postgres -f compose-postgres.yml up -d --build
        docker compose -p postgres exec postgres /scripts/load_dumps.sh
        docker compose -f compose-pidsmaker.yml -f compose-pidsmaker.cpu.yml up -d --build  # (4)!
        docker compose exec pids bash
        ```

        1.  See [how to get an access token](getting-started/installation.md#provenance-datasets). Dumps are saved to `./data`. The network datasets are [built from the files of LARES](getting-started/from-scratch.md) for now.
        2.  Set `INPUT_DIR` to the absolute path of `./data` in `.env`. Port 5000 is used by AirPlay Receiver on macOS.
        3.  Gives 24 GB to the VM of Docker Desktop (8 GB by default), for a 32 GB Mac: about 75% of your RAM.
        4.  The CPU override removes the NVIDIA GPU reservation.

=== ":lucide-play: 2. Run a system"

    Run a provenance or a network system, from inside the `pids` container.

    ```shell
    python pidsmaker/main.py velox CADETS_E3          # a provenance IDS
    python pidsmaker/main.py euler LANL               # a network IDS
    python pidsmaker/main.py kairos THEIA_E3 --wandb  # logged to Weights & Biases
    ```

=== ":lucide-sliders-horizontal: 3. Make it yours"

    Override any argument from the CLI, or write a new YAML config that inherits from an existing system.

    ```shell
    python pidsmaker/main.py orthrus CADETS_E3 \
      --featurization.used_method=doc2vec \
      --training.encoder.used_methods=tgn,sage \
      --training.encoder.sage.num_layers=3
    ```

## Explore the documentation

<div class="grid cards" markdown>

-   :lucide-rocket:{ .lg .middle } __Getting started__

    ---

    Install PIDSMaker with Docker or Apptainer, load the datasets and run your first system.

    [:octicons-arrow-right-24: Installation](getting-started/installation.md)

-   :lucide-book-open:{ .lg .middle } __Background__

    ---

    New to the field? Provenance and network graphs, GNNs and how graph-based IDSs detect attacks.

    [:octicons-arrow-right-24: Background](guide/background.md)

-   :lucide-workflow:{ .lg .middle } __The pipeline__

    ---

    Seven stages, from graph construction to triage, for provenance and network data, with caching of every intermediate result.

    [:octicons-arrow-right-24: Pipeline](guide/pipeline.md)

-   :lucide-shield:{ .lg .middle } __Systems__

    ---

    All the systems side by side, and which datasets each one runs on.

    [:octicons-arrow-right-24: Systems](guide/systems.md)

-   :lucide-database:{ .lg .middle } __Datasets__

    ---

    Provenance (DARPA TC E3/E5, OpTC, ATLASv2, CARBANAKv2) and network (LANL, OpTC flows) datasets, with attack descriptions and ground truth.

    [:octicons-arrow-right-24: Datasets](guide/datasets.md)

-   :lucide-graduation-cap:{ .lg .middle } __Tutorial__

    ---

    Build a new system from existing components and a custom encoder, then run it on provenance and network data.

    [:octicons-arrow-right-24: Build a system](guide/tutorial.md)

-   :lucide-brain-circuit:{ .lg .middle } __Pretrained encoders__

    ---

    Replace word2vec with SPIDER, language models or self-supervised GNNs in any provenance-based IDS.

    [:octicons-arrow-right-24: Pretrained encoders](features/pretrained_encoders.md)

-   :lucide-flask-conical:{ .lg .middle } __Tuning & instability__

    ---

    Hyperparameter sweeps with W&B, and measuring the variance between runs.

    [:octicons-arrow-right-24: Tuning](features/tuning.md)

-   :lucide-target:{ .lg .middle } __Evaluation & labeling__

    ---

    Node, edge, time-window, queue and host-level detection, thresholds, and metrics such as ADP.

    [:octicons-arrow-right-24: Evaluation](guide/evaluation.md)

-   :lucide-swords:{ .lg .middle } __Attack simulation__

    ---

    Test the robustness of PIDSs with mimicry, synthetic and renaming attacks.

    [:octicons-arrow-right-24: Attack simulation](features/attacks.md)

-   :lucide-route:{ .lg .middle } __Triage__

    ---

    Reconstruct attacks from node detections with DepImpact or OCR-APT subgraphs.

    [:octicons-arrow-right-24: Triage](features/triage.md)

-   :lucide-file-code:{ .lg .middle } __Reference__

    ---

    Every CLI flag and configuration argument, generated from the source code.

    [:octicons-arrow-right-24: Reference](reference/cli.md)

</div>

## Citing PIDSMaker

If you use PIDSMaker in your research, please cite the following papers.

=== "Framework paper (KDD 2026)"

    ```bibtex
    @inproceedings{bilot2026pidsmaker,
      title={PIDSMaker: Building and Evaluating Provenance-based Intrusion Detection Systems},
      author={Bilot, Tristan and Jiang, Baoxiang and Pasquier, Thomas},
      booktitle={Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '26)},
      year={2026}
    }
    ```

    [:lucide-book-open: Read the paper](https://dl.acm.org/doi/10.1145/3770855.3817514)

=== "Experiments paper (USENIX Security 2025)"

    ```bibtex
    @inproceedings{bilot2025simpler,
      title={{Sometimes Simpler is Better: A Comprehensive Analysis of State-of-the-Art Provenance-Based Intrusion Detection Systems}},
      author={Bilot, Tristan and Jiang, Baoxiang and Li, Zefeng and El Madhoun, Nour and Al Agha, Khaldoun and Zouaoui, Anis and Pasquier, Thomas},
      booktitle={Security Symposium (USENIX Sec'25)},
      year={2025},
      organization={USENIX}
    }
    ```

    [:lucide-book-open: Read the paper](https://www.usenix.org/conference/usenixsecurity25/presentation/bilot)

!!! tip "Join the community"
    Questions, ideas, or want to discuss a contribution? Come say hi on [Discord](https://discord.gg/jecmkTABUg), or read the [contribution guide](community/contributing.md).
