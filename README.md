[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15603122.svg)](https://doi.org/10.5281/zenodo.15603122)

# Sometimes Simpler is Better: A Comprehensive Analysis of State-of-the-Art Provenance-Based Intrusion Detection Systems

This repo contains the original code of the paper.

<<<<<<< HEAD
> [!IMPORTANT]
> For a practical usage of the framework, we recommend using the last version of PIDSMaker, available on the [main branch](https://github.com/ubc-provenance/PIDSMaker).
=======

# Citation

If you use this work, please cite the following paper:
```
@inproceedings{bilot2025simpler,
	title={{Sometimes Simpler is Better: A Comprehensive Analysis of State-of-the-Art Provenance-Based Intrusion Detection Systems}},
	author={Bilot, Tristan and Jiang, Baoxiang and  Li, Zefeng and  El Madhoun, Nour and Al Agha, Khaldoun and Zouaoui, Anis and Pasquier, Thomas},
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
>>>>>>> fee1406 (fix the reference)

## Setup

### Clone the repo
```
git clone https://github.com/ubc-provenance/PIDSMaker.git -b velox velox
cd velox
```

### 10-min install of Docker and Datasets

We have made the installation of DARPA TC/OpTC easy and fast, simply follow [these guidelines](settings/ten-minute-install.md).

## Reproduce experiments

### Reproducing Velox results

> [!NOTE]
> Due to significant training instability, reproducing the exact results from the paper is unlikely; multiple runs with different seeds may be required.

### Final detection results (Tables 4, 5, 6)

- Replace `{system}` by `velox | orthrus | nodlink | threatrace | kairos | rcaid | flash`.
- Replace `{dataset}` by `CLEARSCOPE_E3 | CADETS_E3 | THEIA_E3 | CLEARSCOPE_E5 | CADETS_E5 | THEIA_E5 | optc_h201 | optc_h501 | optc_h051`.

```shell
./run_local.sh {system} {dataset} --experiment=run_n_times --tuned
```

Note: Flash runs from gnn training as its featurization is too long to re-run.
```shell
./run_local.sh flash {dataset} --experiment=run_n_times --tuned --experiment.uncertainty.deep_ensemble.restart_from=gnn_training
```

### Untuned/tuned systems (Fig. 5)

```shell
./run_local.sh {system} {dataset} # untuned
./run_local.sh {system} {dataset} --tuned # tuned
```

### ADP range (Fig.6)

The results are obtained from the final detection results experiments, taking the range from `adp_score_min` to `adp_score_max`.

### Relative ADP std (Fig. 7)

The results are obtained from the final detection results experiments, doing `adp_score_std` / `adp_score_mean`.

### Featurization methods (Fig. 8)

The config file for the featurization is found in `experiments/tuning/components/tuning_featurization_methods.yml`.
A wandb sweep is run with a run for each combination of hyperparams/featurization method.

```shell
./run_local.sh {system} CADETS_E3 --tuning_mode=featurization --tuned --restart_from_scratch
```

### Ablation heatmap (Fig. 9)

Here, we start from orthrus' config without snooping components (i.e., featurization trained on test data and clutering), referred to as `orthrus_non_snooped`, and we compute the ablations as in the paper's figure.

```shell
./run_local.sh orthrus_non_snooped CLEARSCOPE_E3 --restart_from_scratch --experiment=run_n_times --tuned --tuning_mode=hyperparameters --tuning_file_path=systems/default/tuning_orthrus_non_snooped
```

### Runtime and memory (Fig. 11)

The results are obtained from the final detection results experiments, using `time_gnn_training`, `time_featurization`, `time_per_batch_inference` and `peak_inference_gpu_memory` metrics.

### Predicted scores (Fig. 12)

The figures are obtained from the final detection results experiments.

### Overall ADP scores (Fig. 13)

We simply do the mean of ADP and relative ADP std across all datasets.

### ADP wrt runtime and memory (Fig. 14)

We simply plot ADP mean wrt to metrics in Fig. 11.

### Adversarial attacks (Fig. 15)

We have added mimicry edges during training using `mimicry_edge_num`.

### Real-Time Scalability metrics (Fig. 16)

To simulate real-time detection, we set the batch size to 1 edge instead of 15min, and we monitor metrics.

```shell
# Get peak memory inference with velox using only 1 edge in inference
python plot_real_time_cpu_watt_memory.py velox CADETS_E3 --detection.gnn_training.edge_batch_size_inference=1 --detection.gnn_training.batch_mode=edges --tuned
```

## How to perform hyperparameter tuning?

Reads the YML configuration in `experiments/tuning/systems/default/tuning_fix_uncertainty.yml` and runs a wandb sweep where each combination of parameters has a dedicated wandb run.
The best run and set of hyperparameters is selected from the run with best ADP score.
The best hyperparameters are then updated in `tuned_baselines/{dataset}/tuned_{system}.yml`.
The tuned system with best hyperparameters can be used by adding the `--tuned` arg when running the pipeline.
The config will be merged with the original config from the system.

In each run, the node anomaly scores are saved in the wandb experiments for later use with the metrics downloaded as csv files from the interface, in the `viz_scripts/plot_*.ipynb` notebooks to generate the SVG figures in the paper.

```shell
./run_all_datasets.sh orthrus --tuning_mode=hyperparameters
./run_all_datasets.sh nodlink --tuning_mode=hyperparameters
./run_all_datasets.sh threatrace --tuning_mode=hyperparameters
./run_all_datasets.sh kairos --tuning_mode=hyperparameters
./run_all_datasets.sh flash --tuning_mode=hyperparameters
./run_all_datasets.sh magic --tuning_mode=hyperparameters
./run_all_datasets.sh sigl --tuning_mode=hyperparameters
./run_all_datasets.sh rcaid --tuning_mode=hyperparameters
```
