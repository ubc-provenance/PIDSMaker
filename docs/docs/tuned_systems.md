# Running Tuned Systems

As explained in SC5 of [Bilot et al.](https://www.usenix.org/system/files/usenixsecurity25-bilot.pdf), PIDSs exhibit significant training instability.
Running the same configuration with different random seeds or minor hyperparameter changes often yields substantially different results. 

We provide the best hyperparameters identified through grid search for the main systems in the framework. However, because the framework has since evolved, we do not guarantee that these settings still yield strong performance in the current version.

We recommend [running each system multiple times](https://ubc-provenance.github.io/PIDSMaker/features/instability/) to increase the likelihood of obtaining a run with good metrics. This can be done using `--experiment=run_n_times`, which will run the same configuration for five iterations and compute mean and std metrics.
Alternatively, you can perform [hyperparameter tuning](https://ubc-provenance.github.io/PIDSMaker/features/tuning/) for each system.

## Hyperparameters

### Velox
```sh
python pidsmaker/main.py velox CADETS_E3 --training.encoder.dropout=0.3 --training.lr=0.0001 --training.node_hid_dim=256 --training.node_out_dim=256 --training.num_epochs=12 --featurization.emb_dim=256
```


```sh
python pidsmaker/main.py velox THEIA_E3 --training.encoder.dropout=0.3 --training.lr=0.001 --training.node_hid_dim=320 --training.node_out_dim=320 --training.num_epochs=12 --featurization.emb_dim=256
```


```sh
python pidsmaker/main.py velox CLEARSCOPE_E3 --training.encoder.dropout=0.3 --training.lr=0.001 --training.node_hid_dim=64 --training.node_out_dim=64 --training.num_epochs=12 --featurization.emb_dim=128 --construction.time_window_size=1
```


```sh
python pidsmaker/main.py velox THEIA_E5 --training.encoder.dropout=0.3 --training.lr=0.0001 --training.node_hid_dim=128 --training.node_out_dim=128 --training.num_epochs=12 --featurization.emb_dim=128 --construction.time_window_size=15.0
```


```sh
python pidsmaker/main.py velox CLEARSCOPE_E5 --training.encoder.dropout=0.3 --training.lr=0.001 --training.node_hid_dim=128 --training.node_out_dim=128 --training.num_epochs=12 --featurization.emb_dim=128
```


```sh
python pidsmaker/main.py velox optc_h201 --training.encoder.dropout=0.3 --training.lr=0.001 --training.node_hid_dim=128 --training.node_out_dim=128 --training.num_epochs=12 --featurization.emb_dim=256
```


```sh
python pidsmaker/main.py velox optc_h501 --training.encoder.dropout=0.3 --training.lr=0.001 --training.node_hid_dim=320 --training.node_out_dim=320 --training.num_epochs=12 --featurization.emb_dim=256
```


```sh
python pidsmaker/main.py velox optc_h051 --training.encoder.dropout=0.3 --training.lr=0.001 --training.node_hid_dim=320 --training.node_out_dim=320 --training.num_epochs=12 --featurization.emb_dim=256

```

### Orthrus non snooped
```sh
python pidsmaker/main.py orthrus_non_snooped CADETS_E3 --training.encoder.dropout=0.3 --training.lr=0.001 --training.node_hid_dim=256 --training.node_out_dim=256 --training.num_epochs=12 --featurization.emb_dim=256
```


```sh
python pidsmaker/main.py orthrus_non_snooped THEIA_E3 --training.encoder.dropout=0.3 --training.lr=0.001 --training.node_hid_dim=128 --training.node_out_dim=128 --training.num_epochs=12 --featurization.emb_dim=256 --construction.time_window_size=5
```


```sh
python pidsmaker/main.py orthrus_non_snooped CLEARSCOPE_E3 --training.encoder.dropout=0.3 --training.lr=0.0001 --training.node_hid_dim=64 --training.node_out_dim=64 --training.num_epochs=12 --featurization.emb_dim=128 --construction.time_window_size=1
```


```sh
python pidsmaker/main.py orthrus_non_snooped THEIA_E5 --training.encoder.dropout=0.3 --training.lr=0.001 --training.node_hid_dim=256 --training.node_out_dim=256 --training.num_epochs=12 --featurization.emb_dim=64
```


```sh
python pidsmaker/main.py orthrus_non_snooped CLEARSCOPE_E5 --training.encoder.dropout=0.3 --training.lr=0.001 --training.node_hid_dim=256 --training.node_out_dim=256 --training.num_epochs=12 --featurization.emb_dim=128
```


```sh
python pidsmaker/main.py orthrus_non_snooped optc_h201 --training.encoder.dropout=0.3 --training.lr=0.001 --training.node_hid_dim=128 --training.node_out_dim=128 --training.num_epochs=12 --featurization.emb_dim=128
```


```sh
python pidsmaker/main.py orthrus_non_snooped optc_h501 --training.encoder.dropout=0.3 --training.lr=0.001 --training.node_hid_dim=256 --training.node_out_dim=256 --training.num_epochs=12 --featurization.emb_dim=256
```


```sh
python pidsmaker/main.py orthrus_non_snooped optc_h051 --training.encoder.dropout=0.3 --training.lr=0.001 --training.node_hid_dim=128 --training.node_out_dim=128 --training.num_epochs=12 --featurization.emb_dim=256

```

### NodLink
```sh
python pidsmaker/main.py nodlink CADETS_E3 --training.encoder.dropout=0.3 --training.lr=0.001 --training.node_hid_dim=320 --training.node_out_dim=320 --training.num_epochs=12 --featurization.emb_dim=128 --featurization.epochs=20 --construction.time_window_size=15.0 --transformation.used_methods="none"
```


```sh
python pidsmaker/main.py nodlink THEIA_E3 --training.encoder.dropout=0.3 --training.lr=0.0001 --training.node_hid_dim=320 --training.node_out_dim=320 --training.num_epochs=12 --featurization.emb_dim=128 --featurization.epochs=20 --construction.time_window_size=15.0 --transformation.used_methods="none"
```


```sh
python pidsmaker/main.py nodlink CLEARSCOPE_E3 --training.encoder.dropout=0.3 --training.lr=0.0001 --training.node_hid_dim=64 --training.node_out_dim=64 --training.num_epochs=12 --featurization.emb_dim=128 --construction.time_window_size=1
```


```sh
python pidsmaker/main.py nodlink THEIA_E5 --training.encoder.dropout=0.3 --training.lr=0.001 --training.node_hid_dim=320 --training.node_out_dim=320 --training.num_epochs=12 --featurization.emb_dim=128 --featurization.epochs=20 --construction.time_window_size=15.0 --transformation.used_methods="none"
```


```sh
python pidsmaker/main.py nodlink CLEARSCOPE_E5 --training.encoder.dropout=0.3 --training.lr=0.0001 --training.node_hid_dim=128 --training.node_out_dim=128 --training.num_epochs=12 --featurization.emb_dim=256
```


```sh
python pidsmaker/main.py nodlink optc_h201 --training.encoder.dropout=0.3 --training.lr=0.0001 --training.node_hid_dim=320 --training.node_out_dim=320 --training.num_epochs=12 --featurization.emb_dim=128
```


```sh
python pidsmaker/main.py nodlink optc_h501 --training.encoder.dropout=0.3 --training.lr=0.0001 --training.node_hid_dim=128 --training.node_out_dim=128 --training.num_epochs=12 --featurization.emb_dim=256
```


```sh
python pidsmaker/main.py nodlink optc_h051 --training.encoder.dropout=0.3 --training.lr=0.0001 --training.node_hid_dim=256 --training.node_out_dim=256 --training.num_epochs=12 --featurization.emb_dim=256

```

### Kairos
```sh
python pidsmaker/main.py kairos CADETS_E3 --training.encoder.dropout=0.3 --training.lr=0.001 --training.node_hid_dim=256 --training.node_out_dim=256 --training.num_epochs=12 --featurization.emb_dim=256
```

```sh
python pidsmaker/main.py kairos THEIA_E3 --training.encoder.dropout=0.3 --training.lr=0.001 --training.node_hid_dim=320 --training.node_out_dim=320 --training.num_epochs=12 --featurization.emb_dim=256
```

```sh
python pidsmaker/main.py kairos CLEARSCOPE_E3 --training.encoder.dropout=0.3 --training.lr=0.0001 --training.node_hid_dim=256 --training.node_out_dim=256 --training.num_epochs=12 --featurization.emb_dim=128 --construction.time_window_size=1
```

```sh
python pidsmaker/main.py kairos THEIA_E5 --training.encoder.dropout=0.3 --training.lr=0.001 --training.node_hid_dim=256 --training.node_out_dim=256 --training.num_epochs=12 --featurization.emb_dim=256
```

```sh
python pidsmaker/main.py kairos CLEARSCOPE_E5 --training.encoder.dropout=0.3 --training.lr=0.001 --training.node_hid_dim=64 --training.node_out_dim=64 --training.num_epochs=12 --featurization.emb_dim=32
```

```sh
python pidsmaker/main.py kairos optc_h201 --training.encoder.dropout=0.3 --training.lr=0.001 --training.node_hid_dim=64 --training.node_out_dim=64 --training.num_epochs=12 --featurization.emb_dim=32
```

```sh
python pidsmaker/main.py kairos optc_h501 --training.encoder.dropout=0.3 --training.lr=0.001 --training.node_hid_dim=64 --training.node_out_dim=64 --training.num_epochs=12 --featurization.emb_dim=16
```

```sh
python pidsmaker/main.py kairos optc_h051 --training.encoder.dropout=0.3 --training.lr=0.001 --training.node_hid_dim=128 --training.node_out_dim=128 --training.num_epochs=12 --featurization.emb_dim=32
```
