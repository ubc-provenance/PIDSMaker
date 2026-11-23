---
title: Tuned hyperparameters
description: The best hyperparameters found by grid search for Velox, Orthrus, NodLink and Kairos on each dataset.
---

# Tuned hyperparameters

As explained in SC5 of [Bilot et al.](https://www.usenix.org/system/files/usenixsecurity25-bilot.pdf), PIDSs exhibit significant training instability: running the same configuration with different random seeds or minor hyperparameter changes often yields substantially different results.
{ .lead }

We provide the best hyperparameters identified through grid search for the main systems in the framework.

!!! warning "No guarantee of results"
    The framework has evolved since these hyperparameters were found, so we can't guarantee that they still yield the best performance in the current version.

We recommend [running each system multiple times](../features/instability.md) to increase the likelihood of obtaining a run with good metrics. This can be done with `--experiment=run_n_times`, which runs the same configuration five times and reports the mean and standard deviation of the metrics.
Alternatively, you can [tune the hyperparameters](../features/tuning.md) of each system yourself.

## Hyperparameters

Pick a system, then copy the command of your dataset.

=== "Velox"

    ```shell
    # CADETS_E3
    python pidsmaker/main.py velox CADETS_E3 --training.encoder.dropout=0.3 --training.lr=0.0001 --training.node_hid_dim=256 --training.node_out_dim=256 --training.num_epochs=12 --featurization.emb_dim=256

    # THEIA_E3
    python pidsmaker/main.py velox THEIA_E3 --training.encoder.dropout=0.3 --training.lr=0.001 --training.node_hid_dim=320 --training.node_out_dim=320 --training.num_epochs=12 --featurization.emb_dim=256

    # CLEARSCOPE_E3
    python pidsmaker/main.py velox CLEARSCOPE_E3 --training.encoder.dropout=0.3 --training.lr=0.001 --training.node_hid_dim=64 --training.node_out_dim=64 --training.num_epochs=12 --featurization.emb_dim=128 --construction.time_window_size=1

    # THEIA_E5
    python pidsmaker/main.py velox THEIA_E5 --training.encoder.dropout=0.3 --training.lr=0.0001 --training.node_hid_dim=128 --training.node_out_dim=128 --training.num_epochs=12 --featurization.emb_dim=128 --construction.time_window_size=15.0

    # CLEARSCOPE_E5
    python pidsmaker/main.py velox CLEARSCOPE_E5 --training.encoder.dropout=0.3 --training.lr=0.001 --training.node_hid_dim=128 --training.node_out_dim=128 --training.num_epochs=12 --featurization.emb_dim=128

    # optc_h201
    python pidsmaker/main.py velox optc_h201 --training.encoder.dropout=0.3 --training.lr=0.001 --training.node_hid_dim=128 --training.node_out_dim=128 --training.num_epochs=12 --featurization.emb_dim=256

    # optc_h501
    python pidsmaker/main.py velox optc_h501 --training.encoder.dropout=0.3 --training.lr=0.001 --training.node_hid_dim=320 --training.node_out_dim=320 --training.num_epochs=12 --featurization.emb_dim=256

    # optc_h051
    python pidsmaker/main.py velox optc_h051 --training.encoder.dropout=0.3 --training.lr=0.001 --training.node_hid_dim=320 --training.node_out_dim=320 --training.num_epochs=12 --featurization.emb_dim=256
    ```

=== "Orthrus (non-snooped)"

    ```shell
    # CADETS_E3
    python pidsmaker/main.py orthrus_non_snooped CADETS_E3 --training.encoder.dropout=0.3 --training.lr=0.001 --training.node_hid_dim=256 --training.node_out_dim=256 --training.num_epochs=12 --featurization.emb_dim=256

    # THEIA_E3
    python pidsmaker/main.py orthrus_non_snooped THEIA_E3 --training.encoder.dropout=0.3 --training.lr=0.001 --training.node_hid_dim=128 --training.node_out_dim=128 --training.num_epochs=12 --featurization.emb_dim=256 --construction.time_window_size=5

    # CLEARSCOPE_E3
    python pidsmaker/main.py orthrus_non_snooped CLEARSCOPE_E3 --training.encoder.dropout=0.3 --training.lr=0.0001 --training.node_hid_dim=64 --training.node_out_dim=64 --training.num_epochs=12 --featurization.emb_dim=128 --construction.time_window_size=1

    # THEIA_E5
    python pidsmaker/main.py orthrus_non_snooped THEIA_E5 --training.encoder.dropout=0.3 --training.lr=0.001 --training.node_hid_dim=256 --training.node_out_dim=256 --training.num_epochs=12 --featurization.emb_dim=64

    # CLEARSCOPE_E5
    python pidsmaker/main.py orthrus_non_snooped CLEARSCOPE_E5 --training.encoder.dropout=0.3 --training.lr=0.001 --training.node_hid_dim=256 --training.node_out_dim=256 --training.num_epochs=12 --featurization.emb_dim=128

    # optc_h201
    python pidsmaker/main.py orthrus_non_snooped optc_h201 --training.encoder.dropout=0.3 --training.lr=0.001 --training.node_hid_dim=128 --training.node_out_dim=128 --training.num_epochs=12 --featurization.emb_dim=128

    # optc_h501
    python pidsmaker/main.py orthrus_non_snooped optc_h501 --training.encoder.dropout=0.3 --training.lr=0.001 --training.node_hid_dim=256 --training.node_out_dim=256 --training.num_epochs=12 --featurization.emb_dim=256

    # optc_h051
    python pidsmaker/main.py orthrus_non_snooped optc_h051 --training.encoder.dropout=0.3 --training.lr=0.001 --training.node_hid_dim=128 --training.node_out_dim=128 --training.num_epochs=12 --featurization.emb_dim=256
    ```

=== "NodLink"

    ```shell
    # CADETS_E3
    python pidsmaker/main.py nodlink CADETS_E3 --training.encoder.dropout=0.3 --training.lr=0.001 --training.node_hid_dim=320 --training.node_out_dim=320 --training.num_epochs=12 --featurization.emb_dim=128 --featurization.epochs=20 --construction.time_window_size=15.0 --transformation.used_methods="none"

    # THEIA_E3
    python pidsmaker/main.py nodlink THEIA_E3 --training.encoder.dropout=0.3 --training.lr=0.0001 --training.node_hid_dim=320 --training.node_out_dim=320 --training.num_epochs=12 --featurization.emb_dim=128 --featurization.epochs=20 --construction.time_window_size=15.0 --transformation.used_methods="none"

    # CLEARSCOPE_E3
    python pidsmaker/main.py nodlink CLEARSCOPE_E3 --training.encoder.dropout=0.3 --training.lr=0.0001 --training.node_hid_dim=64 --training.node_out_dim=64 --training.num_epochs=12 --featurization.emb_dim=128 --construction.time_window_size=1

    # THEIA_E5
    python pidsmaker/main.py nodlink THEIA_E5 --training.encoder.dropout=0.3 --training.lr=0.001 --training.node_hid_dim=320 --training.node_out_dim=320 --training.num_epochs=12 --featurization.emb_dim=128 --featurization.epochs=20 --construction.time_window_size=15.0 --transformation.used_methods="none"

    # CLEARSCOPE_E5
    python pidsmaker/main.py nodlink CLEARSCOPE_E5 --training.encoder.dropout=0.3 --training.lr=0.0001 --training.node_hid_dim=128 --training.node_out_dim=128 --training.num_epochs=12 --featurization.emb_dim=256

    # optc_h201
    python pidsmaker/main.py nodlink optc_h201 --training.encoder.dropout=0.3 --training.lr=0.0001 --training.node_hid_dim=320 --training.node_out_dim=320 --training.num_epochs=12 --featurization.emb_dim=128

    # optc_h501
    python pidsmaker/main.py nodlink optc_h501 --training.encoder.dropout=0.3 --training.lr=0.0001 --training.node_hid_dim=128 --training.node_out_dim=128 --training.num_epochs=12 --featurization.emb_dim=256

    # optc_h051
    python pidsmaker/main.py nodlink optc_h051 --training.encoder.dropout=0.3 --training.lr=0.0001 --training.node_hid_dim=256 --training.node_out_dim=256 --training.num_epochs=12 --featurization.emb_dim=256
    ```

=== "Kairos"

    ```shell
    # CADETS_E3
    python pidsmaker/main.py kairos CADETS_E3 --training.encoder.dropout=0.3 --training.lr=0.001 --training.node_hid_dim=256 --training.node_out_dim=256 --training.num_epochs=12 --featurization.emb_dim=256

    # THEIA_E3
    python pidsmaker/main.py kairos THEIA_E3 --training.encoder.dropout=0.3 --training.lr=0.001 --training.node_hid_dim=320 --training.node_out_dim=320 --training.num_epochs=12 --featurization.emb_dim=256

    # CLEARSCOPE_E3
    python pidsmaker/main.py kairos CLEARSCOPE_E3 --training.encoder.dropout=0.3 --training.lr=0.0001 --training.node_hid_dim=256 --training.node_out_dim=256 --training.num_epochs=12 --featurization.emb_dim=128 --construction.time_window_size=1

    # THEIA_E5
    python pidsmaker/main.py kairos THEIA_E5 --training.encoder.dropout=0.3 --training.lr=0.001 --training.node_hid_dim=256 --training.node_out_dim=256 --training.num_epochs=12 --featurization.emb_dim=256

    # CLEARSCOPE_E5
    python pidsmaker/main.py kairos CLEARSCOPE_E5 --training.encoder.dropout=0.3 --training.lr=0.001 --training.node_hid_dim=64 --training.node_out_dim=64 --training.num_epochs=12 --featurization.emb_dim=32

    # optc_h201
    python pidsmaker/main.py kairos optc_h201 --training.encoder.dropout=0.3 --training.lr=0.001 --training.node_hid_dim=64 --training.node_out_dim=64 --training.num_epochs=12 --featurization.emb_dim=32

    # optc_h501
    python pidsmaker/main.py kairos optc_h501 --training.encoder.dropout=0.3 --training.lr=0.001 --training.node_hid_dim=64 --training.node_out_dim=64 --training.num_epochs=12 --featurization.emb_dim=16

    # optc_h051
    python pidsmaker/main.py kairos optc_h051 --training.encoder.dropout=0.3 --training.lr=0.001 --training.node_hid_dim=128 --training.node_out_dim=128 --training.num_epochs=12 --featurization.emb_dim=32
    ```
