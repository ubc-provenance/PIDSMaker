# Instability

PIDSs trained with self-supervised learning exhibit significant **instability**—that is, high sensitivity to training perturbations. Running the same configuration with different random seeds or minor hyperparameter changes often yields substantially different detection performance.

## Why are PIDSs unstable?

Several factors contribute to this instability:

1. **Self-supervised learning**: PIDSs are trained to model normal behavior without labeled attack data. Small variations in how "normal" is learned can dramatically affect what gets flagged as anomalous.

2. **Random initialization**: Neural network weights are randomly initialized, leading to different optimization trajectories.

3. **Stochastic training**: Mini-batch sampling, dropout, and other stochastic elements introduce variability between runs.

## Measuring instability

To quantify instability, PIDSMaker supports running a system multiple times and computing statistics across runs. Use the `--experiment=run_n_times` flag:

```shell
./run.sh orthrus CADETS_E3 --experiment=run_n_times
```

This executes the pipeline N times using the same configuration, with each run using a different random seed.

### Configuration

Parameters are configured in `config/experiments/uncertainty/run_n_times.yml`:

```yaml
training_loop:
  run_evaluation: each_epoch

experiment:
  used_method: uncertainty
  uncertainty:
    deep_ensemble:
      iterations: 5        # number of runs
      restart_from: featurization  # task to restart from
```

| Parameter | Description |
|-----------|-------------|
| `iterations` | Number of times to run the pipeline (default: 5) |
| `restart_from` | The pipeline stage to restart from for each iteration. Earlier stages (e.g., `construction`) are computed once and reused. |

!!! tip
    Setting `restart_from: featurization` or `restart_from: training` saves time by reusing graph construction and transformation outputs across runs. Set it to the earliest stage where randomness is introduced.

### Reported metrics

Upon completion, each metric is reported in three variants:

| Suffix | Description |
|--------|-------------|
| `*_mean` | Mean value across all runs |
| `*_std` | Standard deviation across runs |
| `*_std_rel` | Relative standard deviation (std / mean), useful for comparing instability across metrics with different scales |

For example, if measuring precision score across 5 runs:
- `precision_mean`: Average precision across runs
- `precision_std`: Standard deviation of precision
- `precision_std_rel`: Coefficient of variation (lower is more stable)

!!! note
    The framework is deterministic by default, instability appears when running multiple iterations within a same run (e.g., with `run_n_times`).

## Recommendations

Based on empirical observations:

1. **Run multiple times**: We recommend running each configuration at least 3-5 times to get reliable performance estimates.

2. **Report ranges**: When publishing results, report mean ± standard deviation rather than single-run numbers.

3. **Use the best run**: For practical deployment, you may select the best-performing run from multiple attempts.


!!! warning
    A single run with good metrics may not be reproducible. Always validate important results with multiple runs.
