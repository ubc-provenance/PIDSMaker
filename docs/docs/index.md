---
title: Home
description: PIDSMaker is an open-source framework to build, benchmark and extend graph-based intrusion detection systems on provenance and network data.
hide:
  - navigation
  - toc
---

<div class="pm-hero" markdown>

<img class="pm-hero__logo off-glb" src="assets/pidsmaker_new.png" alt="PIDSMaker logo">

<div markdown>

<h1>PIDS<span class="pm-maker">Maker</span></h1>

<p class="pm-hero__tagline">Build, benchmark and extend graph-based intrusion detection systems.</p>

<p class="pm-hero__sub">State-of-the-art provenance and network IDSs, their datasets and one modular pipeline, all open source.</p>

<div class="pm-hero__actions" markdown>

[:lucide-rocket: Get started](getting-started/installation.md){ .md-button .md-button--primary }
[:lucide-terminal: Quickstart](getting-started/quickstart.md){ .md-button }
[:fontawesome-brands-github: GitHub](https://github.com/ubc-provenance/PIDSMaker){ .md-button }
[:fontawesome-brands-discord: Discord](https://discord.gg/jecmkTABUg){ .md-button }

</div>

<div class="pm-hero__platforms" markdown>

Runs on [:fontawesome-brands-linux: Linux (GPU or CPU)](getting-started/installation.md) [:fontawesome-brands-apple: macOS (Apple Silicon)](getting-started/installation.md#macos)

</div>

</div>

</div>

<style>.md-content__button { display: none; }</style>

<div class="pm-install" markdown>

<div class="pm-install__intro" markdown>

## Up and running in two commands

From v5.0.0, PIDSMaker needs no database server or Docker: datasets are downloaded from [![](assets/huggingface_logo.svg){ .pm-hf-logo .off-glb } Hugging Face](https://huggingface.co/datasets/pidsmaker/datasets){ .pm-hf } on first use.

</div>

<div class="pm-install__code" markdown>

```shell
pip install pidsmaker
pidsmaker orthrus CADETS_E3
```

Requires Python 3.9–3.11 · [Installation guide](getting-started/installation.md) · [Docker](getting-started/docker.md)
{ .pm-install__note }

</div>

</div>

<div class="pm-stats">
  <div class="pm-stat"><strong>17</strong><span>intrusion detection systems</span></div>
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

## Why PIDSMaker?

Papers at top security venues often evaluate on the same datasets, but each uses its own labels and its own implementation of the baselines, so their results can't be compared fairly.
PIDSMaker gives the community **one framework**, maintained in the open, that fixes this.

- **Fair benchmarking**: state-of-the-art baselines evaluated with the same datasets, labels and reference implementations.
- **Modular components**: the components of published systems, extracted into a testbed where you mix and match them to **discover better variants**.
- **Shared code**: a central place where authors contribute their systems, so new work can be benchmarked reproducibly.

## Next steps

=== ":lucide-play: Run other systems"

    Run any system on a provenance or a network dataset, by name:

    ```shell
    pidsmaker velox CADETS_E3          # a provenance IDS
    pidsmaker euler LANL               # a network IDS
    pidsmaker kairos THEIA_E3 --wandb  # logged to Weights & Biases
    ```

=== "![](assets/huggingface_logo.svg){ .pm-hf-logo .off-glb } Download datasets"

    All the [datasets](guide/datasets.md) are Parquet files in the Hugging Face repository [`pidsmaker/datasets`](https://huggingface.co/datasets/pidsmaker/datasets), from 0.16 GB (`ATLASV2_EDR`) to 74 GB (`TRACE_E5`), 178 GB in total. A run downloads its dataset when it is missing, or you can download datasets ahead of time:

    ```shell
    pidsmaker download CADETS_E3 LANL  # some datasets
    pidsmaker download --list          # all the datasets and their sizes
    ```

    Runs on :fontawesome-brands-linux: Linux, ideally with an :simple-nvidia: NVIDIA GPU, for all datasets, and on :fontawesome-brands-apple: macOS (Apple Silicon), on CPU, for most datasets: 32 GB of RAM is ideal ([details](getting-started/installation.md#macos)).

=== ":lucide-sliders-horizontal: Make it yours"

    Override any argument from the CLI, or write a new YAML config that inherits from an existing system.

    ```shell
    pidsmaker orthrus CADETS_E3 \
      --featurization.used_method=doc2vec \
      --training.encoder.used_methods=tgn,sage \
      --training.encoder.sage.num_layers=3
    ```

## Explore the documentation

<div class="grid cards" markdown>

-   :lucide-book-open:{ .lg .middle } __Background__

    ---

    New to the field? Provenance and network graphs, GNNs and how graph-based IDSs detect attacks.

    [:octicons-arrow-right-24: Background](guide/background.md)

-   :lucide-workflow:{ .lg .middle } __The pipeline__

    ---

    Seven stages, from graph construction to triage, with caching of every intermediate result.

    [:octicons-arrow-right-24: Pipeline](guide/pipeline.md)

-   :lucide-shield:{ .lg .middle } __Systems & datasets__

    ---

    All the systems side by side, and the provenance and network datasets with their attacks and ground truth.

    [:octicons-arrow-right-24: Systems](guide/systems.md) · [Datasets](guide/datasets.md)

-   :lucide-target:{ .lg .middle } __Evaluation__

    ---

    Node, edge, time-window, queue and host-level detection, thresholds, and metrics such as ADP.

    [:octicons-arrow-right-24: Evaluation](guide/evaluation.md)

-   :lucide-graduation-cap:{ .lg .middle } __Build your own system__

    ---

    Combine existing components with a custom encoder, then run it on provenance and network data.

    [:octicons-arrow-right-24: Tutorial](guide/tutorial.md)

-   :lucide-flask-conical:{ .lg .middle } __Features__

    ---

    [Pretrained encoders](features/pretrained_encoders.md), [tuning & instability](features/tuning.md), [adversarial attacks](features/attacks.md), [triage](features/triage.md), and every argument in the [Reference](reference/cli.md).

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
