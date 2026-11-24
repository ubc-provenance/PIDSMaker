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
PGPASSWORD=postgres pg_dump -U postgres -h postgres -p 5432 -F c -d DATABASE -f DATABASE.dump
```

`DATABASE` is the name of the database (e.g. `cadets_e3`, `lanl` or `optc_net`). Dumps are restored in a database named after their file, so keep the name of the database. The `pg_dump` client must have the same major version as the Postgres server.

## Network datasets (LANL and OpTC flows)

Network datasets have hosts as nodes and authentications or network flows as edges. They are built from the preprocessed files of [LARES](https://github.com/TristanBilot/lares), which also provide the labels of malicious events used as ground truth.
Unlike provenance datasets, their tables follow the dataset's schema (`pidsmaker/config/schemas.py`): no SQL initialization is needed, as the scripts below create the database and its tables.
Any system runs on them: for example, `construction.edge_type=logon_type` makes the edge type predicted by Velox the logon type of LANL authentications, and `batching.edge_features=edge_type,edge_attrs` gives the edge attributes (ports, protocols, logon types, ...) to the GNN.

### Download the files

In `DATA_FOLDER`, download and extract the LARES archive (14.35 GB, 18.44 GB once extracted):

```shell
curl -L -o ${DATA_FOLDER}/lanl_optc_datasets.tar.gz "https://drive.usercontent.google.com/download?id=10fdDttLXP8zYwlrhqFzOe_kEvDGokfVY&export=download&confirm=t"
tar -xzf ${DATA_FOLDER}/lanl_optc_datasets.tar.gz -C ${DATA_FOLDER}
```

### Fill the database (LANL)

In the `pids` container:

```shell
python dataset_preprocessing/lanl/create_database_lanl.py LANL --lares_dir /data/lanl_optc_datasets/LANL/preprocessed
```

This creates the `lanl` database with the NTLM authentications of the [LANL "Comprehensive, Multi-Source Cyber-Security Events"](https://csr.lanl.gov/data/cyber1/) dataset, as used by Euler, Argus, Jbeil and LARES. Add `--with_flows` to also load its network flows.

The database can also be built from the official files, requested on the [LANL website](https://csr.lanl.gov/data/cyber1/) (authentications only):

```shell
python dataset_preprocessing/lanl/create_database_lanl.py LANL --auth_file /data/auth.txt.gz --redteam_file /data/redteam.txt.gz
```

!!! note
    LANL timestamps are seconds from an undisclosed origin. They are mapped to UTC dates such that the first red team event happens on 2015-01-03: by default, models are trained on 2015-01-01, validated on 2015-01-02 and tested from 2015-01-03 to 2015-01-30.

### Fill the database (OpTC flows)

In the `pids` container:

```shell
python dataset_preprocessing/optc/create_database_optc_net.py OPTC_NET --lares_dir /data/lanl_optc_datasets/OPTC/preprocessed
```

This creates the `optc_net` database with the FLOW START and FLOW MESSAGE events of all the hosts of [DARPA OpTC](https://github.com/FiveDirections/OpTC-data) (about 1.9 billion events). Files are read by `--num_workers` processes (16 by default), each loading its events directly into the database.

!!! note
    As in LARES and Argus, graphs are built from FLOW START events only. Set `construction.operations=all` to also use FLOW MESSAGE events, or a comma-separated list of operations (e.g. `construction.operations=FLOW_START`) to select events of any dataset.

**Labels.** A flow is recorded several times in OpTC (FLOW START and FLOW MESSAGE events, by the hosts at both ends), and LARES labels only some of these records: e.g. in the lateral movement of 2019-09-24, the FLOW MESSAGE events of 561 flows are malicious but the FLOW START of only 56 of them. By default, the script labels malicious all the records of a malicious flow (same 5-tuple, in either direction, in the same LARES file): 5,165 malicious events (1,015 FLOW START) instead of 2,370 (115 FLOW START). Add `--no_label_propagation` to keep the labels of LARES as they are. A database built without propagation can be relabelled in place with `--relabel`; graphs built before must then be built again (the labels of edges are stored in the graphs).

Both scripts index the events by time (used by the construction) and the malicious events (used by the evaluation).

### Dump the network databases

The dumps of `lanl` and `optc_net` are created from the databases filled above, in the `pids` container:

```shell
PGPASSWORD=postgres pg_dump -U postgres -h postgres -p 5432 -F c -d lanl -f /data/lanl.dump
PGPASSWORD=postgres pg_dump -U postgres -h postgres -p 5432 -F c -d optc_net -f /data/optc_net.dump
```

`lanl.dump` is about 0.7 GB and `optc_net.dump` about 21 GB (dumping `optc_net` takes about 2 hours). For the dump of OpTC with the labels of LARES, build the database with `--no_label_propagation` before dumping it, and keep its file name `optc_net.dump` (dumps are restored in a database named after their file), e.g. in a `lares_labels/` folder. `scripts/load_dumps.sh` also creates the index of malicious events when restoring dumps made without it.

The official network flows of OpTC (`ecar-bro` folder of its Google Drive) can be downloaded with:

```shell
python dataset_preprocessing/optc/download_optc_net.py ${DATA_FOLDER}/optc_net
```

!!! tip "Adding a new dataset"
    To add a dataset that PIDSMaker does not support yet, see [Adding custom datasets](../guide/datasets.md#adding-custom-datasets).
