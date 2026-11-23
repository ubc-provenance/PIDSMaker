---
title: Home
description: PIDSMaker is an open-source framework to build, benchmark and extend provenance-based intrusion detection systems (PIDSs).
hide:
  - navigation
  - toc
---

<div class="pm-hero" markdown>

<img class="pm-hero__logo" src="assets/pidsmaker_new.png" alt="PIDSMaker logo">

<div markdown>

<h1>PIDS<span class="pm-maker">Maker</span></h1>

<p class="pm-hero__tagline">
The open-source framework to <strong>build, benchmark and extend provenance-based intrusion detection systems</strong>.
State-of-the-art PIDSs, the main DARPA datasets and one modular pipeline, all in a single codebase.
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
  <div class="pm-stat"><strong>9</strong><span>PIDSs reimplemented</span></div>
  <div class="pm-stat"><strong>15</strong><span>ready-to-load datasets</span></div>
  <div class="pm-stat"><strong>7</strong><span>configurable pipeline stages</span></div>
  <div class="pm-stat"><strong>15+</strong><span>pretrained encoders</span></div>
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

    The components of published systems, extracted into a modular testbed. Mix and match them to **discover better variants**.

-   :lucide-git-pull-request:{ .lg .middle } __Shared code__

    ---

    A central place where authors contribute their systems, so new work can be benchmarked reproducibly.

</div>

## Up and running in three steps

=== ":lucide-download: 1. Install"

    Download a preprocessed dataset, load it into Postgres and open a shell in the `pids` container ([full guide](getting-started/installation.md)).

    ```shell
    git clone https://github.com/ubc-provenance/PIDSMaker.git && cd PIDSMaker
    ./download_datasets.sh cadets_e3 YOUR_GOOGLE_ACCESS_TOKEN  # (1)!
    cp .env.local .env                                          # (2)!
    docker compose -p postgres -f compose-postgres.yml up -d --build
    docker compose -p postgres exec postgres /scripts/load_dumps.sh
    docker compose -f compose-pidsmaker.yml up -d --build
    docker compose exec pids bash
    ```

    1.  See [how to get an access token](getting-started/installation.md#1-download-the-datasets). Dumps are saved to `./data`.
    2.  Set `INPUT_DIR` to the absolute path of `./data` in `.env`.

=== ":lucide-play: 2. Run a PIDS"

    Run any system on any dataset, from inside the `pids` container.

    ```shell
    python pidsmaker/main.py velox CADETS_E3          # in the shell
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

-   :lucide-network:{ .lg .middle } __Background__

    ---

    New to the field? Provenance graphs, GNNs and how PIDSs detect attacks.

    [:octicons-arrow-right-24: Background](guide/background.md)

-   :lucide-workflow:{ .lg .middle } __The pipeline__

    ---

    Seven stages, from graph construction to triage, with caching of every intermediate result.

    [:octicons-arrow-right-24: Pipeline](guide/pipeline.md)

-   :lucide-shield:{ .lg .middle } __Systems__

    ---

    Velox, Orthrus, Kairos, MAGIC, Flash, NodLink, R-Caid, ThreaTrace and OCR-APT, side by side.

    [:octicons-arrow-right-24: Systems](guide/systems.md)

-   :lucide-database:{ .lg .middle } __Datasets__

    ---

    DARPA TC E3/E5, OpTC, ATLASv2 and CARBANAKv2, with attack descriptions and ground truth.

    [:octicons-arrow-right-24: Datasets](guide/datasets.md)

-   :lucide-graduation-cap:{ .lg .middle } __Tutorial__

    ---

    Build a new PIDS from existing components and a custom encoder, step by step.

    [:octicons-arrow-right-24: Build a PIDS](guide/tutorial.md)

-   :lucide-brain-circuit:{ .lg .middle } __Pretrained encoders__

    ---

    Replace word2vec with SPIDER, language models or self-supervised GNNs in any PIDS.

    [:octicons-arrow-right-24: Pretrained encoders](features/pretrained_encoders.md)

-   :lucide-flask-conical:{ .lg .middle } __Tuning & instability__

    ---

    Hyperparameter sweeps with W&B, and measuring the variance between runs.

    [:octicons-arrow-right-24: Tuning](features/tuning.md)

-   :lucide-target:{ .lg .middle } __Evaluation & labeling__

    ---

    Node, edge, time-window and queue-level detection, thresholds, and metrics such as ADP.

    [:octicons-arrow-right-24: Evaluation](guide/evaluation.md)

-   :lucide-swords:{ .lg .middle } __Attack simulation__

    ---

    Test robustness with mimicry, synthetic and renaming attacks.

    [:octicons-arrow-right-24: Attack simulation](features/attacks.md)

-   :lucide-route:{ .lg .middle } __Triage__

    ---

    Reconstruct attacks from detections with DepImpact or OCR-APT subgraphs.

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

    [:lucide-book-open: Read the paper](https://www.arxiv.org/abs/2601.22983)

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
