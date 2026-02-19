<p align="center">
  <img width="80%" src="./.github/img/pidsmaker_title.png" alt="PIDSMAKER logo"/>
</p>

<div align="center">

[![Docs](https://img.shields.io/badge/Docs-Online-ed6a2f?style=flat&labelColor=gray)](https://ubc-provenance.github.io/PIDSMaker/)
[![DOI](https://img.shields.io/badge/DOI-10.5281/zenodo.15603122-ed6a2f?style=flat&labelColor=gray)](https://doi.org/10.5281/zenodo.15603122)
[![License](https://img.shields.io/github/license/ubc-provenance/PIDSMaker?style=flat&color=ed6a2f&labelColor=gray)](LICENSE)
[![Release](https://img.shields.io/github/v/release/ubc-provenance/PIDSMaker?style=flat&color=ed6a2f&labelColor=gray)](https://github.com/ubc-provenance/PIDSMaker/releases)
[![Stars](https://img.shields.io/github/stars/ubc-provenance/PIDSMaker?style=flat&color=ed6a2f&labelColor=white&logo=github&logoColor=black)](https://github.com/ubc-provenance/PIDSMaker/stargazers)

</div>

<p align="center">
  <strong>
    <a href="https://www.usenix.org/conference/usenixsecurity25/presentation/bilot">📄 Experiments Paper</a>
    &nbsp;|&nbsp;
	<a href="https://www.arxiv.org/abs/2601.22983">📄 Framework Paper</a>
    &nbsp;|&nbsp;
    <a href="https://ubc-provenance.github.io/PIDSMaker/">📘 Documentation</a>
    &nbsp;|&nbsp;
    <a href="https://ubc-provenance.github.io/PIDSMaker/ten-minute-install/">⚙️ Installation</a>
  </strong>
</p>

---

The first framework designed to build and experiment with provenance-based intrusion detection systems (PIDSs) using deep learning architectures.
It provides a single codebase to run most recent state-of-the-arts systems and easily customize them to develop new variants.

### Supported Systems

The framework currently integrates the following PIDSs.

| PIDS       | Venue               | Paper |
|------------|---------------------|-------|
| Velox      | USENIX Security 2025 | [Link](https://tfjmp.org/publications/2025-usenixsec-2.pdf) |
| Orthrus    | USENIX Security 2025 | [Link](https://www.usenix.org/system/files/conference/usenixsecurity25/sec25cycle1-prepub-103-jiang-baoxiang.pdf) |
| R-Caid     | IEEE S&P 2024        | [Link](https://gangw.web.illinois.edu/rcaid-sp24.pdf) |
| Flash      | IEEE S&P 2024        | [Link](https://dartlab.org/assets/pdf/flash.pdf) |
| Kairos     | IEEE S&P 2024        | [Link](https://arxiv.org/pdf/2308.05034) |
| Magic      | USENIX Security 2024 | [Link](https://www.usenix.org/system/files/usenixsecurity24-jia-zian.pdf) |
| NodLink    | NDSS 2024           | [Link](https://arxiv.org/pdf/2311.02331) |
| ThreaTrace | IEEE TIFS 2022      | [Link](https://arxiv.org/pdf/2111.04333) |

### Supported Datasets

It also includes several easy-to-install provenance datasets for APT detection.

| Dataset | OS | Attacks | Size (GB) |
|---------|------|---------|-----------|
| CADETS_E3 | FreeBSD | 3 | 10 |
| THEIA_E3 | Linux | 2 | 12 |
| CLEARSCOPE_E3 | Android | 1 | 4.8 |
| FIVEDIRECTIONS_E3 | Windows | 2 | 22 |
| TRACE_E3 | Linux | 3 | 100 |
| CADETS_E5 | FreeBSD | 2 | 276 |
| THEIA_E5 | Linux | 1 | 36 |
| CLEARSCOPE_E5 | Android | 2 | 49 |
| FIVEDIRECTIONS_E5 | Windows | 4 | 280 |
| TRACE_E5 | Linux | 1 | 710 |
| optc_h201 | Windows | 1 | 9 |
| optc_h501 | Windows | 1 | 6.7 |
| optc_h051 | Windows | 1 | 7.7 |

## 📄 Documentation

A [comprehensive documentation](https://ubc-provenance.github.io/PIDSMaker/) is available, explaining all possible arguments and providing examples on how integrating new systems.

### Pipeline

The framework integrates a [pipeline](https://ubc-provenance.github.io/PIDSMaker/pipeline) composed of seven stages, each parameterizable via configurable arguments, enabling flexible customization of new systems.

<img src="docs/docs/img/pipeline.svg" style="width: 100%"/>


## Setup

### ⬇️ Clone the repo
```
git clone https://github.com/ubc-provenance/PIDSMaker.git
```

### 💻 Installation with Docker

We have made the installation of PIDSMaker inclusing pre-processed databases for DARPA TC and OpTC datasets easy and fast. Simply follow [these guidelines](https://ubc-provenance.github.io/PIDSMaker/ten-minute-install/).

## 🧪 Basic usage of the framework

Once you have a followed the installation guidelines, you can open a shell in the `pids container` and experiment in multiple ways.
Replace `SYSTEM` by `velox`, `orthrus`, `nodlink`, `threatrace`, `kairos`, `rcaid`, `flash`, `magic`.

1. Run in the shell:
    ```shell
    python pidsmaker/main.py SYSTEM DATASET
    ```

2. Run in the shell, monitored to weights & biases (W&B):
    ```shell
    python pidsmaker/main.py SYSTEM DATASET --wandb
    ```

3. Run in background, monitored to W&B (recommended for multiple parallel runs and for research):
    ```shell
    ./run.sh SYSTEM DATASET
    ```

You can still watch the logs in your shell using `tail -f nohup.out`.

We generally using using W&B for experiment monitoring and historization (see installation guidelines). 

**Warning:** Before performing evaluations, you should tune all systems (see docs [here](https://ubc-provenance.github.io/PIDSMaker/features/tuning/)).

## Reproducing results

PIDSs exhibit significant instability—that is, high sensitivity to training perturbations—due to their self-supervised training nature. 
Running the same configuration with different random seeds or minor hyperparameter changes often yields substantially different results. 
Consequently, reproducing results as the framework evolves presents a real challenge.

Based on our experiments, we provide [tuned hyperparameters](https://ubc-provenance.github.io/PIDSMaker/tuned_systems) for the main systems.
However, we can't guarantee that these hyperparameters will lead to satisfactory results due to instability.

We recommend [running each system multiple times](https://ubc-provenance.github.io/PIDSMaker/features/instability/) to increase the likelihood of obtaining a run with good metrics. Alternatively, you can perform [hyperparameter tuning](https://ubc-provenance.github.io/PIDSMaker/features/tuning/) for each system.

## Customize existing systems

The default configuration files in `config/*.yml` represent the architecture of existing PIDSs in YAML format. They contain the original hyperparameters used by each system. 

The main strength of PIDSMaker is the customization of existing systems for easy experimentation. 
A few examples below.

### From CLI

<i>Running Kairos with embedding size of 128 instead of 100, and last neighbor sampling set to last 10 neighbors instead of 20.</i>

```shell
python pidsmaker/main.py kairos CADETS_E3 \
  --training.node_hid_dim=128 \
  --batching.intra_graph_batching.tgn_last_neighbor.tgn_neighbor_size=10
```

<i>Running Orthrus with Doc2vec instead of word2vec, and 3 GraphSAGE layers instead of 2 attention layers.</i>

```shell
python pidsmaker/main.py orthrus CADETS_E3 \
  --featurization.used_method=doc2vec \
  --featurization.emb_dim=128 \
  --training.encoder.used_methods=tgn,sage \
  --training.encoder.sage.num_layers=3
```

### From a new YAML config file

Want to create a new PIDS? Create a new config under `config/your_system.yml`, inherit from existing PIDSs and tune it as you want.

<i>Magic with node type prediction instead of its hybrid masked feature reconstruction and structure prediction objective function, and use a 2-layer MLP with ReLU as decoder, and use NodLink's thresholding method.</i>

``` yaml
_include_yml: magic

training:
  decoder:
    used_methods: predict_node_type
    predict_node_type:
      node_mlp:
        architecture_str: linear(0.5) | relu

evaluation:
  node_evaluation:
    threshold_method: nodlink
```

### Visualization

You can then visualize the results using the many generated figures, locally or on Weights and Biases.

![alt text](.github/img/scores.png)

## Hyperparameter tuning

PIDSMaker supports easy hyperparameter tuning for existing or new models. 
Follow the [instructions](https://ubc-provenance.github.io/PIDSMaker/features/tuning/) available in our documentation.

You can specify the range of hyperparameters to search in a yaml config.

```yaml
method: grid 

parameters:
  training.lr:
    values: [0.001, 0.0001]
  training.node_hid_dim:
    values: [32, 64, 128, 256]
  featurization.used_method:
    values: [fasttext, word2vec]
```

Then run the framework in tuning mode.

```sh
./run.sh my_system CADETS_E3 --tuning_mode=hyperparameters
```

Once you find the best hyperparameters, store them in a yaml file and run your tuned model.

```sh
./run.sh my_system CADETS_E3 --tuned
```

## Citation

If you use this work, please cite the two following papers:
```
@article{bilot2026pidsmaker,
  title={PIDSMaker: Building and Evaluating Provenance-based Intrusion Detection Systems},
  author={Bilot, Tristan and Jiang, Baoxiang and Pasquier, Thomas},
  journal={arXiv preprint arXiv:2601.22983},
  year={2026}
}
@inproceedings{bilot2025simpler,
	title={{Sometimes Simpler is Better: A Comprehensive Analysis of State-of-the-Art Provenance-Based Intrusion Detection Systems}},
	author={Bilot, Tristan and Jiang, Baoxiang and  Li, Zefeng and  El Madhoun, Nour and Al Agha, Khaldoun and Zouaoui, Anis and Pasquier, Thomas},
	booktitle={Security Symposium (USENIX Sec'25)},
	year={2025},
	organization={USENIX}
}
```

## Contributing

Pull requests are welcome! Please follow the [contribution guidelines](https://ubc-provenance.github.io/PIDSMaker/contributing/).

## License

See [licence](LICENSE).
