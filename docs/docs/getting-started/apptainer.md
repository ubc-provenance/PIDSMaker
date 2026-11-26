---
title: HPC clusters
description: Install and run PIDSMaker on HPC clusters with conda and pip, without Docker, and run the legacy Postgres backend with Singularity/Apptainer.
---

# HPC clusters

PIDSMaker needs no container and no database server: on an HPC cluster, install it with conda and pip as on any Linux machine. [Singularity/Apptainer](https://apptainer.org/) is only needed for the [legacy Postgres backend](installation.md#postgres).
{ .lead }

## 1. Install PIDSMaker

From a login node, follow [the installation guide](installation.md#1-install-pidsmaker) (if your cluster provides conda as a module, load it first):

```shell
conda create -n pids python=3.11
conda activate pids
pip install pidsmaker
```

## 2. Choose where files are stored

Home directories are often small on clusters. Store the datasets and the artifacts of your runs on a large filesystem, e.g. your project or scratch space, by adding these lines to your `~/.bashrc` or to your job scripts:

```shell
export PIDSMAKER_DATASETS_DIR=/path/to/large/disk/pidsmaker/datasets
export PIDSMAKER_ARTIFACT_DIR=/path/to/large/disk/pidsmaker/artifacts
```

See [Where files are stored](installation.md#paths).

## 3. Download the datasets from a login node

Compute nodes often have no internet access. From a login node, download the datasets, and the NLTK tokenizer data that PIDSMaker otherwise downloads the first time it is imported:

```shell
pidsmaker download CADETS_E3 THEIA_E3
python -m nltk.downloader punkt
```

In your jobs, set `PIDSMAKER_AUTO_DOWNLOAD=0`: a missing dataset then stops the run with an error, instead of a download attempt.

## 4. Run the framework

In a job, with the conda environment activated:

```shell
pidsmaker SYSTEM DATASET
```

Continue with the [Quickstart](quickstart.md).

## Legacy: Postgres with Apptainer

For the [legacy Postgres backend](installation.md#postgres), the Makefile in `scripts/apptainer/` of the repository runs a Postgres server with Singularity/Apptainer, and loads the database dumps of the folder `INPUT_DIR` (by default `scripts/apptainer/data/`):

```shell
cd scripts/apptainer
INPUT_DIR=/path/to/dumps make full-setup   # start Postgres, then load the dumps
make down                                  # stop the database
make up                                    # start it again
```

Then install the Postgres client library, and point PIDSMaker to the local server:

```shell
pip install "pidsmaker[postgres]"
pidsmaker SYSTEM DATASET --database_backend postgres --database_host localhost
```

The other database settings (`--database_user`, `--database_password`, `--database_port`) are listed in the [command line reference](../reference/cli.md#datasets-and-storage).
