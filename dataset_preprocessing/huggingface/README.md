---
pretty_name: PIDSMaker datasets
license: other
tags:
  - cybersecurity
  - intrusion-detection
  - provenance
  - graph
  - darpa-tc
  - optc
size_categories:
  - 1B<n<10B
---

<!-- Dataset card of the Hugging Face repository of the datasets (uploaded as its README.md).
     Kept in the PIDSMaker repository: dataset_preprocessing/huggingface/README.md -->

# PIDSMaker datasets

The preprocessed datasets of [PIDSMaker](https://github.com/ubc-provenance/PIDSMaker), a framework to build and evaluate provenance-based and network intrusion detection systems: the provenance datasets DARPA TC (Engagements 3 and 5), DARPA OpTC, ATLASv2 and CARBANAKv2, and the network datasets LANL and the network flows of OpTC.

They are read by PIDSMaker in place, with DuckDB: install PIDSMaker and download them with its command.

```shell
pip install pidsmaker
pidsmaker download CADETS_E3          # or several datasets, or `all`; --list to list them
pidsmaker orthrus CADETS_E3           # a run also downloads its dataset if it is missing
```

## Layout

One folder per dataset (named after its database), with a `manifest.json` and one folder per table:

```
cadets_e3/
  manifest.json
  event_table/part-2018-04-02.parquet ...       # events (one file per day, UTC, for most datasets)
  file_node_table/part-00000.parquet
  netflow_node_table/part-00000.parquet
  subject_node_table/part-00000.parquet
```

The manifest lists the files, columns, number of rows and order of each table. Events are sorted by `timestamp_rec, event_uuid`; node tables keep the order in which the framework reads them (results depend on it). The files were exported from the Postgres databases used by previous versions of PIDSMaker (`dataset_preprocessing/export_parquet.py`), and give the same results.

| Dataset | Folder | Events | Size (GB) | Source |
|---|---|--:|--:|---|
| CADETS_E3 | `cadets_e3` | 36.5M | 1.3 | [DARPA TC E3](https://github.com/darpa-i2o/Transparent-Computing/blob/master/README-E3.md) |
| THEIA_E3 | `theia_e3` | 44.4M | 0.71 | DARPA TC E3 |
| CLEARSCOPE_E3 | `clearscope_e3` | 18.3M | 0.51 | DARPA TC E3 |
| FIVEDIRECTIONS_E3 | `fivedirections_e3` | 78.9M | 2.3 | DARPA TC E3 |
| TRACE_E3 | `trace_e3` | 284M | 9.4 | DARPA TC E3 |
| CADETS_E5 | `cadets_e5` | 1.08B | 32.9 | [DARPA TC E5](https://github.com/darpa-i2o/Transparent-Computing) |
| THEIA_E5 | `theia_e5` | 141M | 3.9 | DARPA TC E5 |
| CLEARSCOPE_E5 | `clearscope_e5` | 199M | 5.0 | DARPA TC E5 |
| FIVEDIRECTIONS_E5 | `fivedirections_e5` | 1.09B | 30.3 | DARPA TC E5 |
| TRACE_E5 | `trace_e5` | 2.64B | 74.3 | DARPA TC E5 |
| optc_h051 | `optc_051` | 31.6M | 1.3 | [DARPA OpTC](https://github.com/FiveDirections/OpTC-data) |
| optc_h201 | `optc_201` | 38.2M | 1.6 | DARPA OpTC |
| optc_h501 | `optc_501` | 28.0M | 1.2 | DARPA OpTC |
| ATLASV2_EDR | `atlasv2_edr` | 2.70M | 0.16 | [ATLASv2](https://arxiv.org/abs/2401.01341) |
| CARBANAKV2_EDR | `carbanakv2_edr` | 17.1M | 1.1 | [CARBANAKv2](https://www.ndss-symposium.org/wp-content/uploads/prism2026-12.pdf) |
| LANL | `lanl` | 49.3M | 0.40 | [LANL](https://csr.lanl.gov/data/cyber1/), preprocessed by [LARES](https://github.com/TristanBilot/lares) |
| OPTC_NET | `optc_net` | 1.86B | 11.8 | DARPA OpTC, preprocessed by LARES |

The attack scenarios, ground truth and statistics of each dataset are described in the [documentation](https://ubc-provenance.github.io/PIDSMaker/guide/datasets/). The ground truth of the provenance datasets is distributed with PIDSMaker; the events of the network datasets are labelled (`label` column).

## License

These files are derived from the datasets above, and are distributed under the terms of their original releases: refer to each source before use.

## Citation

```bibtex
@inproceedings{bilot2026pidsmaker,
  title={PIDSMaker: Building and Evaluating Provenance-based Intrusion Detection Systems},
  author={Bilot, Tristan and Jiang, Baoxiang and Pasquier, Thomas},
  booktitle={Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '26)},
  year={2026}
}
@inproceedings{bilot2025simpler,
  title={{Sometimes Simpler is Better: A Comprehensive Analysis of State-of-the-Art Provenance-Based Intrusion Detection Systems}},
  author={Bilot, Tristan and Jiang, Baoxiang and Li, Zefeng and El Madhoun, Nour and Al Agha, Khaldoun and Zouaoui, Anis and Pasquier, Thomas},
  booktitle={Security Symposium (USENIX Sec'25)},
  year={2025},
  organization={USENIX}
}
```
