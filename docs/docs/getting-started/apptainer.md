---
title: Installation (Apptainer)
description: Install PIDSMaker with Singularity/Apptainer and conda, for HPC clusters where Docker is not available.
---

# Installation with Apptainer

For environments without Docker, such as HPC clusters, PIDSMaker can run with [Singularity/Apptainer](https://apptainer.org/) for the database and a conda environment for Python.
{ .lead }

!!! info "Requirements"
    Singularity/Apptainer and conda are already installed on the machine, and you have downloaded the database dumps of your datasets (see [step 1 of the Docker guide](installation.md#1-download-the-datasets)).

## 1. Set up the database

The Makefile in `scripts/apptainer/` sets up the database in one command:

```shell
cd ./scripts/apptainer
make full-setup
```

This command:

- downloads and runs a PostgreSQL container through Singularity/Apptainer;
- loads the database dumps with the `load_dumps.sh` script.

Once the database is ready, you can stop and restart its container at any time:

```shell
make down   # stop the database
make up     # start it again
```

## 2. Install the Python environment

Create and activate the conda environment:

```shell
conda env create -f ./scripts/apptainer/environment.yaml
conda activate pids
```

## 3. Run the framework

From the root of the repository, point PIDSMaker to the local database and to a folder for the artifacts:

```shell
python pidsmaker/main.py SYSTEM DATASET --artifact_dir ./artifacts/ --database_host localhost
```

The other database settings (`--database_user`, `--database_password`, `--database_port`) are listed in the [command line reference](../reference/cli.md#database-and-storage).

Continue with the [Quickstart](quickstart.md).
