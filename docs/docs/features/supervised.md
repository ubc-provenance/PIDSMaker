# Supervised Fine-Tuning with Attack Edges

PIDSs usually learn from benign data only.
The `predict_edge_supervised` objective adds some supervision: a binary classifier learns to separate the benign edges of the training graphs from a set of attack edges that you provide, then scores every test edge by how much it looks like an attack.

## How it works

- **Input:** for each edge, the decoder receives the source and destination node features, each concatenated with the one-hot edge type. The GNN encoder is bypassed, so the node features themselves must be informative: this objective is meant to be used with [pretrained encoder](pretrained_encoders.md) embeddings, such as SPIDER.
- **Training:** the negatives are the edges of the training graphs. The attack edges are the positives, oversampled with replacement to match each batch of benign edges. `pos_weight` adds weight to the attack class in the binary cross-entropy loss.
- **Detection:** the anomaly score of a test edge is its loss against the benign label, which grows with the predicted probability of being an attack.

Enable it by setting the objective in the `training.decoder` section:

```yaml
training:
  decoder:
    used_methods: predict_edge_supervised
    predict_edge_supervised:
      mode: synthetic
      attack_edges_path: config/attack_edges/cadets_e3.yml
      pos_weight: 1.0
      decoder: edge_mlp
      edge_mlp:
        architecture_str: linear(0.5) | relu
        src_dst_projection_coef: 2
```

## Attack edges from a YAML file (`mode: synthetic`)

In this mode, the attack edges are written by hand in a YAML file and don't need to exist in the dataset.
During the `feat_inference` task, SPIDER embeds the labels of their nodes, so this mode requires `featurization.used_method: spider`.

Each edge gives the type and label of its two nodes and the event type:

```yaml
attack_edges:
  # nginx (exploited web server) spawns a shell
  - src_type: subject
    src_label: "subject None nginx"
    edge_type: EVENT_CLONE
    dst_type: subject
    dst_label: "subject None sh"
```

Labels follow the format of the dataset's node labels, set by `construction.node_label_features`: with the default `subject: type, path, cmd_line`, a process label is `subject <path> <command line>`, and `None` stands for a missing attribute.

`config/attack_edges/` provides one file for CADETS_E3, CADETS_E5, THEIA_E3, THEIA_E5, TRACE_E3 and TRACE_E5.
Each file describes an attack similar to the dataset's scenario, but with different indicators (such as C2 IP addresses), so that the classifier can't simply memorize the real attack.

`spider_supervised.yml` runs Velox with SPIDER embeddings and this objective on CADETS_E3:

```shell
python pidsmaker/main.py spider_supervised CADETS_E3 --featurization.spider.spider_path=$SPIDER_WEIGHTS
```

For another dataset, point `attack_edges_path` to its file:

```shell
python pidsmaker/main.py spider_supervised THEIA_E3 --featurization.spider.spider_path=$SPIDER_WEIGHTS \
    --training.decoder.predict_edge_supervised.attack_edges_path=config/attack_edges/theia_e3.yml
```

## Attack edges matched in the data (`mode: patterns`)

In this mode, the attack edges are collected from the validation and test graphs with a list of patterns.
Each pattern can set any of `src_type`, `src_label_contains`, `dst_type`, `dst_label_contains` and `edge_type`; a missing field matches anything.
`max_edges_per_pattern` caps the number of edges collected per pattern.

```yaml
training:
  decoder:
    used_methods: predict_edge_supervised
    predict_edge_supervised:
      mode: patterns
      max_edges_per_pattern: 50
      attack_patterns:
        - src_type: subject
          src_label_contains: nginx
          edge_type: EVENT_CLONE
```

!!! warning
    In this mode the positives come from the evaluation data, so the results are not comparable with those of unsupervised PIDSs.

All options are listed in the [objective arguments](../config/objectives.md).
