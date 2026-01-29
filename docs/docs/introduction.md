# Introduction

In this introduction, we go through the basics that allow you to run existings PIDSs and create your own variant.

!!! note
    To follow the tutorial, you should have completed the [installation guidelines](./ten-minute-install.md) and open a shell within the pids container.

## Basic usage of the framework

The framework currently support the following systems and datasets:

**Systems**

- `velox`
- `orthrus`
- `nodlink`
- `threatrace`
- `kairos`
- `rcaid`
- `flash`

**Datasets**

- `CADETS_E3`
- `THEIA_E3`
- `CLEARSCOPE_E3`
- `CADETS_E5`
- `THEIA_E5`
- `CLEARSCOPE_E5`
- `optc_h201`
- `optc_h501`
- `optc_h051`

**Run the framework**

The basic usage of the framework is:

```shell
python pidsmaker/main.py SYSTEM DATASET --arg1=x --arg2=y
```

The entrypoint is always `main.py` and only two arguments are mandatory:

- `SYSTEM`: should point to an existing YML file with the same system name in `config/`. The file contains the configuration of the particular system.
- `DATASET`: should point to an existing dataset defined within `DATASET_DEFAULT_CONFIG` in `config/config.py`. It's there that `DATASET` is mapped to the actual database name, located in the postgres container.

After running the framework, the content of `config/SYSTEM.yml` is parsed and verified based on the available arguments located in `TASK_ARGS` from `config.py`. These arguments are presented in the `Arguments` section of the documentation.

Below are the different ways to run the framework.

1. Run in the shell, no W&B:
    ```shell
    python pidsmaker/main.py SYSTEM DATASET
    ```

2. Run in the shell, monitored to W&B:
    ```shell
    python pidsmaker/main.py SYSTEM DATASET --wandb
    ```

3. Run in background, monitored to W&B (ideal for multiple parallel runs):
    ```shell
    ./run.sh SYSTEM DATASET
    ```
    You can still watch the logs in your shell using `tail -f nohup.out`

## Device

By default, the framework runs on GPU and searches for an existing device on `CUDA:0`. If no GPU is detected, it switches to CPU and a warning message is printed to the console.
The utilization of CPU can be forced using the `--cpu` CLI argument.

## The framework

To familiarize yourself with PIDSMaker, consider going through the [pipeline](pipeline.md) and [tutorial](tutorial.md) pages.
