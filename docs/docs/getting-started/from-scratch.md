---
title: Building datasets from scratch
description: Build the PIDSMaker Postgres databases from the official DARPA TC and OpTC files instead of the preprocessed dumps.
---

# Building datasets from scratch

PIDSMaker ships [preprocessed database dumps](installation.md#1-download-the-datasets) of all its datasets. This guide is for when you want to rebuild them yourself from the official DARPA files, for instance to change the parsing.
{ .lead }

!!! warning "Plan for time and disk space"
    Downloading, uncompressing and parsing raw files takes hours and a lot of storage, especially for E5 datasets. Running experiments takes more: a single run can produce more than 15 GB of artifacts on E3 datasets, and much more on E5.

## 1. Download the raw files

Create an empty folder for the raw data:

```shell
DATA_FOLDER=./data
mkdir ${DATA_FOLDER}
pip install gdown
```

!!! note
    `DATA_FOLDER` must be empty before downloading and parsing a dataset. Remove the files of a previous dataset first.

=== "DARPA TC (E3, E5)"

    ```shell
    ./dataset_preprocessing/darpa_tc/scripts/download_DATASET.sh ${DATA_FOLDER}
    ```

    `DATASET` is one of `clearscope_e3`, `cadets_e3`, `theia_e3`, `clearscope_e5`, `cadets_e5` or `theia_e5`, and `DATA_FOLDER` is the **absolute** path of the output folder.

=== "DARPA OpTC"

    ```shell
    python ./dataset_preprocessing/optc/download_dataset.py DATASET ${DATA_FOLDER}
    ```

    `DATASET` is one of `optc_h051`, `optc_h201` or `optc_h501`, and `DATA_FOLDER` is the **absolute** path of the output folder.

### Downloading the files manually

For DARPA TC, you can also pick the files by hand on Google Drive. All files go in `DATA_FOLDER`:

1. **Data**: all the `.gz` files of the dataset, from the [E3](https://drive.google.com/drive/folders/1fOCY3ERsEmXmvDekG-LUUSjfWs6TRdp-) or [E5](https://drive.google.com/drive/folders/1GVlHQwjJte3yz0n1a1y4H4TfSe8cu6WJ) folder.
2. **Java consumer**: `ta3-java-consumer.tar.gz`, which converts the Avro files, for [E3](https://drive.google.com/drive/folders/1kCRC5CPI8MvTKQFvPO4hWIRHeuUXLrr1) or [E5](https://drive.google.com/drive/folders/1YDxodpEmwu4VTlczsrLGkZMnh_o70lUh).
3. **Schema**: the `.avdl` and `.avsc` files, for [E3](https://drive.google.com/drive/folders/1gwm2gAlKHQnFvETgPA8kJXLLm3L-Z3H1) or [E5](https://drive.google.com/drive/folders/1fCdYCIMBCm7gmBpmqDTuoMhbOoIY6Wzq).

From the command line, [use gdown](https://stackoverflow.com/a/50670037/10183259) with the ID of each file, taken from its URL. If a download stops, press ++ctrl+c++ and re-run the same command with `--continue`.

## 2. Mount the raw files in the container

1. In `compose-pidsmaker.yml`, uncomment the `- /path/to/raw/data:/data` volume and replace `/path/to/raw/data` with your `DATA_FOLDER`, which contains the downloaded files (`.gz`, the Java consumer `.tar.gz` and the schema files).
2. Build the image and open a shell in the `pids` container, as in the [installation guide](installation.md#5-open-the-pidsmaker-container).

## 3. Extract the files

In the `pids` container:

=== "DARPA TC (E3, E5)"

    ```shell
    ./dataset_preprocessing/darpa_tc/scripts/uncompress_darpa_files.sh /data/
    ```

    This may take several hours, depending on the dataset.

=== "DARPA OpTC"

    ```shell
    ./dataset_preprocessing/optc/extract_data.sh /data/
    ```

## 4. Fill the database

!!! info "Using your own Postgres server?"
    If you don't use the Postgres container, set the connection settings in `DATABASE_DEFAULT_CONFIG` in `pidsmaker/config/pipeline.py`, then run [`init-create-empty-databases.sh`](https://github.com/ubc-provenance/PIDSMaker/blob/main/postgres/init-create-empty-databases.sh) on your server to create the databases, and [`init-create-databases.sh`](https://github.com/ubc-provenance/PIDSMaker/blob/main/postgres/init-create-databases.sh) to create the tables.

Still in the `pids` container:

=== "DARPA TC E3"

    ```shell
    python dataset_preprocessing/darpa_tc/create_database_e3.py orthrus DATASET
    ```

    `DATASET` is one of `CLEARSCOPE_E3`, `CADETS_E3` or `THEIA_E3`.

=== "DARPA TC E5"

    ```shell
    python dataset_preprocessing/darpa_tc/create_database_e5.py orthrus DATASET
    ```

    `DATASET` is one of `CLEARSCOPE_E5`, `CADETS_E5` or `THEIA_E5`.

=== "DARPA OpTC"

    ```shell
    python dataset_preprocessing/optc/create_database_optc.py orthrus DATASET
    ```

    `DATASET` is one of `optc_h051`, `optc_h201` or `optc_h501`.

## 5. Check the result

If the containers are not running, start them and open a shell in `pids`:

```shell
docker compose -p postgres -f compose-postgres.yml up -d --build
docker compose -f compose-pidsmaker.yml up -d --build
docker compose exec pids bash
```

Then run a system on the new database:

```shell
python pidsmaker/main.py orthrus DATASET
```

To share your database, export it as a dump:

```shell
PGPASSWORD=postgres pg_dump -U postgres -h postgres -p 5432 -F c -d DATASET -f DATASET.dump
```

!!! tip "Adding a new dataset"
    To add a dataset that PIDSMaker does not support yet, see [Adding custom datasets](../guide/datasets.md#adding-custom-datasets).
