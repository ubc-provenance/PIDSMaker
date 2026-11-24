---
title: Multi-dataset training
description: Train the featurization model and the detector on several datasets, and evaluate on one.
---

# Multi-dataset training

Train on the benign activity of several datasets, then evaluate on one of them. This tests whether a detector generalizes beyond the machine it was trained on, and gives it more benign data.
{ .lead }

The dataset passed on the command line is the **primary** dataset: it is the only one evaluated. The others are listed in `construction.multi_dataset`, and two flags choose which stages learn from all of them:

| Argument | Effect |
|---|---|
| `construction.multi_dataset` | Comma-separated datasets to use besides the primary one (`none` by default). Their graphs are built and transformed. |
| `featurization.multi_dataset_training` | Trains the text embedding model on the labels of all the datasets. Only changes `word2vec` and `fasttext`. |
| `batching.multi_dataset_training` | Trains the detector on the training graphs of all the datasets, and validates on all their validation graphs. |

For example, Orthrus trained on CADETS E3 and THEIA E3, and evaluated on CADETS E3:

```shell
python pidsmaker/main.py orthrus CADETS_E3 \
  --construction.multi_dataset=CADETS_E3,THEIA_E3 \
  --featurization.multi_dataset_training=True \
  --batching.multi_dataset_training=True
```

Listing the primary dataset in `multi_dataset` is optional. Setting a `multi_dataset_training` flag without `multi_dataset` raises an error.

!!! info "Provenance datasets"
    Multi-dataset training is designed for provenance datasets, and is not tested on network datasets. The other datasets are processed with the settings of the primary dataset, including its schema (node types, edge types and event attributes), so a provenance dataset and a network dataset can't be combined.

!!! tip "Mixing DARPA TC and OpTC"
    OpTC uses its own event types. With `construction.consistent_edge_types: True`, OpTC events are mapped to their DARPA TC equivalents (e.g. `MESSAGE` from a process to a network node becomes `EVENT_SENDMSG`), so that both share the same edge types. The [pretrained encoder](pretrained_encoders.md) configs enable it.

## Related: pretraining datasets

Two other arguments train a featurization model on several datasets, without the rest of the pipeline:

- `featurization.pretrain_datasets`: the datasets whose labels train `word2vec`. Their graphs must already have been built.
- `featurization.pretrained.pretrain_datasets`: the datasets used to [pretrain](pretrained_encoders.md#pretraining) SPIDER and the other pretrained encoders.
