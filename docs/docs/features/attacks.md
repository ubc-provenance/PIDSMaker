---
title: Adversarial attacks
description: Evaluate the robustness of PIDSs with mimicry attacks, synthetic attacks injected in the graphs, and renaming attacks.
---

# Adversarial attacks

How robust is a detector to an attacker who tries to blend in? PIDSMaker can modify the data to simulate attacks, at three different stages of the pipeline.
{ .lead }

| Attack | What changes | Stage | Argument |
|---|---|---|---|
| [Mimicry attack](#mimicry-attack) | Benign-looking file activity is added to the attack processes, in the test graphs | Construction | `construction.mimicry_edge_num` |
| [Synthetic attacks](#synthetic-attacks) | Fake attacks are injected in the training and validation graphs, labeled for few-shot detection | Transformation | `transformation.used_methods: synthetic_attack_naive` |
| [Renaming attack](pretrained_encoders.md#renaming-attack) | Attack processes get the embedding of a benign name | Feature inference | `feat_inference.rename_attack` |

!!! info "Provenance datasets only"
    The three simulations act on provenance activity: they add file accesses and connections of processes with the event types of DARPA TC, or rename processes. They don't apply to network datasets.

## Mimicry attack

A mimicry attack hides malicious processes by making them behave like benign ones: on top of the attack, they perform ordinary file accesses, which dilutes the anomaly that the detector learned to spot.

With `construction.mimicry_edge_num` set to N > 0, PIDSMaker adds N such events to the attacks of the dataset, split evenly between them. For each attack:

<ol class="pm-steps" markdown>
<li markdown>**Attack processes.** The attack's nodes are taken from the ground truth. Its entry points (the attack nodes without incoming events) and the processes reachable from them during the attack's time range are used as attack processes.</li>
<li markdown>**Benign-looking events.** Until the quota is reached, a random attack process opens a random file of the dataset and performs one of 14 typical sequences of reads and writes, at a time when the process was active, a few seconds apart.</li>
<li markdown>**Injection.** The events are added to the graph of the day of the attack. They are not written to the database.</li>
<li markdown>**Ground truth.** The processes and files used are saved in the `mimicry/` folder of the construction task, and added to the ground truth as attack nodes, so evaluation checks whether the detector still finds them.</li>
</ol>

```shell
python pidsmaker/main.py orthrus CADETS_E3 --construction.mimicry_edge_num=1000
```

Compare the metrics with those of the same run without mimicry (`mimicry_edge_num: 0`, the default in every system). Since it is a construction argument, the whole pipeline runs again.

!!! warning "Limitations"
    - The generated events use the event types of DARPA TC: use mimicry on the E3 and E5 datasets only.
    - It is not supported with the `threatrace` ground truth, nor with `construction.used_method: magic`.
    - The injected events are added after the real events of the day, so they end up in the **last** time-window graph of the day rather than in the windows of their timestamps.

## Synthetic attacks

`synthetic_attack_naive` injects simple, labeled attacks in the **training and validation** graphs, so that a detector can learn from attack examples (few-shot detection). Test graphs are left unchanged.

1. It selects `num_malicious_process` processes that receive network data (`EVENT_RECVFROM`) in both the training and validation graphs. The only selection method is `random`.
2. In up to `num_attacks` graphs where each process appears, it adds unauthorized accesses (`EVENT_OPEN`, `EVENT_READ`, `EVENT_WRITE`) to `num_unauthorized_file_access` files that the process never accessed, then a connection (`EVENT_CONNECT`) to a network node, as an exfiltration.
3. Injected edges are labeled as attacks (`y = 1`). They are removed from self-supervised training, and used by the few-shot objective (`training.decoder.use_few_shot`).

```shell
python pidsmaker/main.py orthrus CADETS_E3 \
  --transformation.used_methods=synthetic_attack_naive \
  --transformation.synthetic_attack_naive.num_malicious_process=10
```

!!! note
    `synthetic_attack_naive` must be the only transformation, and its labels are only used by few-shot training, a legacy feature. Evaluation always uses the real ground truth. To train a detector with attack examples, prefer the [supervised fine-tuning](supervised.md) objective.

## Renaming attack

Attackers often rename their malware after a benign program. With [pretrained encoders](pretrained_encoders.md), which embed entities from their names, `feat_inference.rename_attack` simulates this at inference time: the embedding of each attack process is replaced by the embedding of a frequent benign name. See [Renaming attack](pretrained_encoders.md#renaming-attack).
