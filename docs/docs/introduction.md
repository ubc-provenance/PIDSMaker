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
python pidsmaker/main.py SYSTEM DATASET
```

The entrypoint is always `main.py` and only two arguments are mandatory:

- `SYSTEM`: should point to an existing YML file with the same system name in `config/`. The file contains the configuration of the particular system.
- `DATASET`: should point to an existing dataset defined within `DATASET_DEFAULT_CONFIG` in `config/config.py`. It's there that `DATASET` is mapped to the actual database name, located in the postgres container.

After running the framework, the content of `config/SYSTEM.yml` is parsed and verified based on the available arguments located in `TASK_ARGS` from `config.py`. The full list of available arguments can be found [here](/configuration).

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

