# Variability

## Measure variability across multiple iterations

Most systems are prone to uncertainty, with some runs reaching high performance, while others fail dramatically. To quantify this variability, we run the system multiple times using different random seeds and compute the mean and standard deviation of key performance metrics. This can be done easily by using the `--experiment=run_n_times` tag:

```shell
./run.sh orthrus CADETS_E3 --tuned --experiment=run_n_times
```

This process executes the pipeline N times using the same configuration, with each run starting from a specified task. All parameters can be configured in `config/experiments/uncertainty/run_n_times.yml`, where the number of iterations is defined by `iterations`, and the initial task by `restart_from`.

Upon completion of all runs, each metric will be reported in three variants: `*_mean`, `*_std`, and `*_std_rel`, corresponding to the mean, standard deviation, and relative standard deviation (i.e., standard deviation normalized by the mean), respectively.
