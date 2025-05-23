# Hyperparameter Tuning

PIDSMaker simplifies hyperparameter tuning by combining its efficient pipeline design with the power of [W&B Sweeps](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Organizing_Hyperparameter_Sweeps_in_PyTorch_with_W%26B.ipynb).


## Dataset-specific tuning
Tuning is configured using YAML files, just like system definitions. For example, suppose you've created a new system named `my_system`, and its configuration is stored in `config/my_system.yml`. To search for optimal hyperparameters on the `THEIA_E3` dataset, you can create a new tuning configuration file at `config/experiments/tuning/systems/theia_e3/tuning_my_system.yml` following the [W&B syntax](https://docs.wandb.ai/guides/sweeps/define-sweep-configuration/):

``` yaml linenums="1" title="tuning_my_system.yml"
method: grid # (1)!

parameters:
  detection.gnn_training.lr:
    values: [0.001, 0.0001]
  detection.gnn_training.node_hid_dim:
    values: [32, 64, 128, 256]
  featurization.feat_training.used_method:
    values: [fasttext, word2vec]
```

1.  Other hyperparameter search strategies like `random` and `bayesian` can be used (more info [here](https://docs.wandb.ai/guides/sweeps/sweep-config-keys/#method)).

Before starting the hyperparameter search, make sure you are logged in to W&B by running: `wandb login` inside the container’s shell. Then, launch the hyperparameter tuning with: `--tuning_mode=hyperparameters`:

``` shell
./run.sh my_system CADETS_E3 --tuning_mode=hyperparameters
```

This flag will automatically load the tuning configuration from `config/experiments/tuning/systems/theia_e3/tuning_my_system.yml`, based on the provided dataset and system names. Any overlapping arguments defined in `config/my_system.yml` will be overridden by those specified in the tuning file. If the specified tuning file does not exist (i.e., no dataset-specific configuration is available), the pipeline falls back to the default tuning configuration: `config/experiments/tuning/systems/default/tuning_default_baselines.yml`.

!!! note
    You can also pass arguments directly via the CLI. CLI arguments always take precedence and will override both the system configuration (`config/my_system.yml`) and the tuning configuration (`tuning_my_system.yml`).

## Cross-dataset tuning

If you wish to use the same hyperparameter tuning configuration across all datasets, you can explicitly specify the tuning file as a command-line argument.
For instance, you might create a tuning file named `config/experiments/tuning/systems/default/tuning_my_system_all_datasets.yml`, then apply it to all dataset by running:

``` shell
./run_all_datasets.py my_system \
    --tuning_mode=hyperparameters \
    --tuning_file_path=systems/default/tuning_my_system_all_datasets
```

or on a single dataset:

``` shell
./run.sh my_system CLEARSCOPE_E3 \
    --tuning_mode=hyperparameters \
    --tuning_file_path=systems/default/tuning_my_system_all_datasets
```

The path passed to `--tuning_file_path` should start from `systems/`.

!!! tips
    For a better historization of experiments, we recommend to assign a name to each sweep (`--exp`) and a dedicated project name (`--project`):

    ``` shell
    ./run.sh my_system CADETS_E3 \
        --tuning_mode=hyperparameters \
        --exp=bench_fasttext_word2vec \
        --project=best_featurization_method
    ```

## Best model selection

Once the sweep has finished, the best run can be obtained from W&B by sorting based on your desired metric.

![W&B sweep](../img/sweep_sorted.png)

Get the hyperparameters associated with the best run and put them into a `config/tuned_baselines/{dataset}/tuned_my_system.yml`.

``` yaml linenums="1" title="tuned_my_system.yml"
featurization:
  feat_training:
    used_method: word2vec

detection:
  gnn_training:
    lr: 0.0001
    node_hid_dim: 128
```

Each system should have a tuned file per dataset, or the best hyperparameters can be directly set in its `my_system.yml` file if it uses the same hyperparameters in all datasets.

!!! note
    The process of best hyperparameter selection is currently done by hand but will likely be automated in future.

## Run a tuned system

Running a system with its best hyperparameters on a particular dataset is as using the `--tuned` arg:

``` shell
./run.sh my_system CADETS_E3 --tuned
```

This will search for the `config/tuned_baselines/cadets_e3/tuned_my_system.yml` file and override the default system config by the best hyperparameters.
