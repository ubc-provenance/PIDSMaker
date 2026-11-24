---
title: Instability
description: Measure and reduce the run-to-run instability of provenance and network IDSs with run_n_times and stable_optim.
---

# Instability

Systems trained with self-supervised learning, provenance-based (PIDSs) and network (NIDSs) alike, exhibit significant **instability**—that is, high sensitivity to training perturbations. Running the same configuration with different random seeds or minor hyperparameter changes often yields substantially different detection performance.

## Why are these systems unstable?

Several factors contribute to this instability:

1. **Self-supervised learning**: Detectors are trained to model normal behavior without labeled attack data. Small variations in how "normal" is learned can dramatically affect what gets flagged as anomalous.

2. **Random initialization**: Neural network weights are randomly initialized, leading to different optimization trajectories.

3. **Stochastic training**: Mini-batch sampling, negative sampling, dropout, and other stochastic elements introduce variability between runs.

For example, on LANL, whether LARES selects the root attacker in the first stage of its evaluation depends on the seed and the epoch (see the header of `config/lares.yml`), and on `OPTC_NET`, label-free epoch selection picks a nearly arbitrary epoch (see [Systems](../guide/systems.md#results-and-reproducibility)).

## Measuring instability

To quantify instability, PIDSMaker supports running a system multiple times and computing statistics across runs. Use the `--experiment=run_n_times` flag:

```shell
./run.sh orthrus CADETS_E3 --experiment=run_n_times
```

This executes the pipeline N times using the same configuration, with each run using a different random seed. Run it from the `scripts/` folder.

### Configuration

Parameters are configured in `config/experiments/uncertainty/run_n_times.yml`:

```yaml
training_loop:
  run_evaluation: each_epoch

experiment:
  used_method: uncertainty
  uncertainty:
    deep_ensemble:
      method: increasing_seed      # a new seed at each run
      iterations: 5                # number of runs
      restart_from: featurization  # task to restart from
```

| Parameter | Description |
|-----------|-------------|
| `method` | `increasing_seed`: run *i* uses the seed + *i* of the `restart_from` task, which must be `featurization` or `training`. `same_seed`: every run uses the same seed, so the variance only comes from non-deterministic operations. |
| `iterations` | Number of times to run the pipeline (default: 5) |
| `restart_from` | The pipeline stage to restart from for each iteration. Earlier stages (e.g., `construction`) are computed once and reused. |

Any value can be overridden from the CLI, e.g. `--experiment.uncertainty.deep_ensemble.iterations=10`.

!!! tip
    Setting `restart_from: featurization` or `restart_from: training` saves time by reusing graph construction and transformation outputs across runs. Set it to the earliest stage where randomness is introduced.

### Reported metrics

Upon completion, each metric is reported in three variants:

| Suffix | Description |
|--------|-------------|
| `*_mean` | Mean value across all runs |
| `*_std` | Standard deviation across runs |
| `*_std_rel` | Relative standard deviation, in percent (100 × std / mean), useful for comparing instability across metrics with different scales |

For example, if measuring precision score across 5 runs:

- `precision_mean`: Average precision across runs
- `precision_std`: Standard deviation of precision
- `precision_std_rel`: Coefficient of variation, in percent (lower is more stable)

The metrics of the runs with the lowest and highest ADP are also logged with the `_min` and `_max` suffixes, the best run is logged under the plain metric names, and all runs are saved in `uncertainty_exp/method_to_metrics.pkl` in the evaluation folder.

### Other experiments

`config/experiments/uncertainty/` provides other ways to perturb runs. Each key under `experiment.uncertainty` is a method, run one after the other:

| Experiment | Method | What changes between runs |
|---|---|---|
| `run_n_times` | `deep_ensemble` | The random seed (see above) |
| `run_n_times_hyperparameters` | `hyperparameter` | Each hyperparameter of `hyperparameters` (`lr`, `num_epochs`, `text_h_dim`, `gnn_h_dim`) is moved by a relative step of ±`delta`, e.g. −40%, −20%, 0, +20%, +40% with 5 iterations (must be odd) |
| `uncertainty` | all four | Runs `hyperparameter`, `mc_dropout`, `deep_ensemble` and `bagged_ensemble` in turn |

`mc_dropout` adds dropout of rate `dropout` to every layer of the encoder and keeps it active at inference, so each run makes different predictions. `bagged_ensemble` is meant to train on a growing number of training days, starting at `min_num_days`.

```shell
./run.sh orthrus CADETS_E3 --experiment=run_n_times_hyperparameters
```

To run a single method, write your own YAML in `config/experiments/uncertainty/` and pass its name to `--experiment`. All arguments are listed in the [Tasks reference](../reference/tasks.md#experiments).

!!! note
    The framework is deterministic by default, instability appears when running multiple iterations within a same run (e.g., with `run_n_times`).

## Reducing instability

`training.stable_optim: True` trains the detector with an optimization setup designed to reduce the variance between runs:

- **AdamW** (betas 0.9 and 0.99, weight decay 0.02) instead of Adam. `training.weight_decay` is then ignored.
- **Learning rate schedule:** a linear warmup over the first 5% of the training steps, then a cosine decay down to 10% of `training.lr`.
- **Gradient clipping** to a norm of 1.0.

It is off by default, and on in the [pretrained encoder](pretrained_encoders.md) configs. It can be enabled for any system, for example with `run_n_times` to measure its effect on the `*_std` metrics:

```shell
./run.sh orthrus CADETS_E3 --training.stable_optim=True --experiment=run_n_times
```

## Recommendations

Based on empirical observations:

1. **Run multiple times**: We recommend running each configuration at least 3-5 times to get reliable performance estimates.

2. **Report ranges**: When publishing results, report mean ± standard deviation rather than single-run numbers.

3. **Use the best run**: For practical deployment, you may select the best-performing run from multiple attempts.


!!! warning
    A single run with good metrics may not be reproducible. Always validate important results with multiple runs.
