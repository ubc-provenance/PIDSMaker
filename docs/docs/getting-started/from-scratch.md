---
title: Building datasets from scratch
description: Build the PIDSMaker datasets yourself, from the official DARPA TC and OpTC files for provenance datasets, and from the preprocessed files of LARES for the LANL and OpTC network datasets.
---

# Building datasets from scratch

PIDSMaker [downloads](installation.md#2-download-the-datasets) preprocessed versions of all its datasets. This guide builds them yourself, for instance to change the parsing: the **provenance datasets** from the official DARPA files, and the **network datasets** (LANL and the network flows of OpTC) from the preprocessed files of [LARES](https://github.com/TristanBilot/lares). The scripts write Parquet datasets with the tables and columns of the downloaded ones, without any database server.
{ .lead }

!!! warning "Plan for time and disk space"
    Downloading, uncompressing and parsing raw files takes hours and a lot of storage, especially for E5 datasets and OpTC flows: the raw files take much more space than the datasets, and the scripts write temporary files (unsorted events) about as large as the dataset. The events are sorted with DuckDB, which uses up to 80% of the RAM and spills the rest to disk (about 85 GB of RAM for the 1.9 billion events of `optc_net`, built in 42 minutes). Running experiments takes more: a single run can produce more than 15 GB of artifacts on E3 datasets, and much more on E5.

The rows of a dataset, and their order, are those of the database that the previous versions of the scripts loaded into Postgres. The downloaded datasets come from databases restored from dumps, which store some rows of the node tables in another order: a dataset built from scratch can give slightly different features than the downloaded one.

The scripts are in the repository: [install PIDSMaker from source](installation.md#1-install-pidsmaker) (`pip install -e .`) and run them from its root.

Both kinds of datasets follow the same five steps. Pick your dataset in the tabs of each step.

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

    `DATASET` is one of `clearscope_e3`, `cadets_e3`, `theia_e3`, `clearscope_e5`, `cadets_e5` or `theia_e5`, and `DATA_FOLDER` is the **absolute** path of the output folder. The datasets built in [step 4](#4-build-the-dataset) are `CADETS_E3`, `THEIA_E3`, `CADETS_E5` and `THEIA_E5`, whose files are listed in `dataset_preprocessing/darpa_tc/filelist.py`.

=== "DARPA OpTC"

    ```shell
    python ./dataset_preprocessing/optc/download_dataset.py DATASET ${DATA_FOLDER}
    ```

    `DATASET` is one of `optc_h051`, `optc_h201` or `optc_h501`, and `DATA_FOLDER` is the **absolute** path of the output folder.

=== "Network (LANL, OpTC flows)"

    Download the archive of LARES, with the preprocessed files and labels of both datasets (14.35 GB, 18.44 GB once extracted):

    ```shell
    curl -L -o ${DATA_FOLDER}/lanl_optc_datasets.tar.gz "https://drive.usercontent.google.com/download?id=10fdDttLXP8zYwlrhqFzOe_kEvDGokfVY&export=download&confirm=t"
    ```

### Downloading the files manually

For DARPA TC, you can also pick the files by hand on Google Drive. All files go in `DATA_FOLDER`:

1. **Data**: all the `.gz` files of the dataset, from the [E3](https://drive.google.com/drive/folders/1fOCY3ERsEmXmvDekG-LUUSjfWs6TRdp-) or [E5](https://drive.google.com/drive/folders/1GVlHQwjJte3yz0n1a1y4H4TfSe8cu6WJ) folder.
2. **Java consumer**: `ta3-java-consumer.tar.gz`, which converts the Avro files, for [E3](https://drive.google.com/drive/folders/1kCRC5CPI8MvTKQFvPO4hWIRHeuUXLrr1) or [E5](https://drive.google.com/drive/folders/1YDxodpEmwu4VTlczsrLGkZMnh_o70lUh).
3. **Schema**: the `.avdl` and `.avsc` files, for [E3](https://drive.google.com/drive/folders/1gwm2gAlKHQnFvETgPA8kJXLLm3L-Z3H1) or [E5](https://drive.google.com/drive/folders/1fCdYCIMBCm7gmBpmqDTuoMhbOoIY6Wzq).

From the command line, [use gdown](https://stackoverflow.com/a/50670037/10183259) with the ID of each file, taken from its URL. If a download stops, press ++ctrl+c++ and re-run the same command with `--continue`.

For the network datasets, LANL can also be built from its official files, requested on the [LANL website](https://csr.lanl.gov/data/cyber1/) (authentications only, see [step 4](#4-build-the-dataset)), and the official network flows of OpTC (`ecar-bro` folder of its Google Drive) are downloaded with:

```shell
python dataset_preprocessing/optc/download_optc_net.py ${DATA_FOLDER}/optc_net
```

<span id="2-mount-the-raw-files-in-the-container"></span>

## 2. Get Java for DARPA TC

The raw files of DARPA TC are Avro files, converted to JSON by the Java consumer of DARPA, which needs **Java 8** and **Maven**. The [`pids` container](docker.md) has both: mount `DATA_FOLDER` in it (uncomment the `- /path/to/raw/data:/data` volume of `compose-pidsmaker.yml`, with your `DATA_FOLDER`), and run the next steps in the container with `DATA_FOLDER=/data`. Otherwise, install them on your machine (e.g. `sudo apt install openjdk-8-jdk maven`).

The other datasets need nothing more than PIDSMaker.

## 3. Extract the files

=== "DARPA TC (E3, E5)"

    ```shell
    ./dataset_preprocessing/darpa_tc/scripts/uncompress_darpa_files.sh ${DATA_FOLDER}/
    ```

    This may take several hours, depending on the dataset.

=== "DARPA OpTC"

    ```shell
    ./dataset_preprocessing/optc/extract_data.sh ${DATA_FOLDER}/
    ```

=== "Network (LANL, OpTC flows)"

    ```shell
    tar -xzf ${DATA_FOLDER}/lanl_optc_datasets.tar.gz -C ${DATA_FOLDER}/
    ```

<span id="4-fill-the-database"></span>

## 4. Build the dataset

Each script writes a dataset to `<datasets_dir>/<database>/`, where PIDSMaker reads it: `--datasets_dir`, else `$PIDSMAKER_DATASETS_DIR`, else `~/.cache/pidsmaker/datasets` (see [Where files are stored](installation.md#paths)), and the database of the dataset, e.g. `cadets_e3`. The dataset is written to `<database>.tmp` and renamed once complete. An existing dataset, e.g. a downloaded one, is only replaced with `--overwrite`.

=== "DARPA TC E3"

    ```shell
    python dataset_preprocessing/darpa_tc/create_database_e3.py orthrus DATASET --raw_dir ${DATA_FOLDER}
    ```

    `DATASET` is `CADETS_E3` or `THEIA_E3`.

=== "DARPA TC E5"

    ```shell
    python dataset_preprocessing/darpa_tc/create_database_e5.py orthrus DATASET --raw_dir ${DATA_FOLDER}
    ```

    `DATASET` is `CADETS_E5` or `THEIA_E5`.

=== "DARPA OpTC"

    ```shell
    python dataset_preprocessing/optc/create_database_optc.py orthrus DATASET --raw_dir ${DATA_FOLDER}
    ```

    `DATASET` is one of `optc_h051`, `optc_h201` or `optc_h501`.

=== "LANL"

    ```shell
    python dataset_preprocessing/lanl/create_database_lanl.py LANL --lares_dir ${DATA_FOLDER}/lanl_optc_datasets/LANL/preprocessed
    ```

    This builds the `lanl` dataset with the NTLM authentications of the [LANL "Comprehensive, Multi-Source Cyber-Security Events"](https://csr.lanl.gov/data/cyber1/) dataset, as used by Euler, Argus, Jbeil and LARES. Add `--with_flows` to also load its network flows.

    From the official files of LANL instead (authentications only):

    ```shell
    python dataset_preprocessing/lanl/create_database_lanl.py LANL --auth_file ${DATA_FOLDER}/auth.txt.gz --redteam_file ${DATA_FOLDER}/redteam.txt.gz
    ```

    LANL timestamps are seconds from an undisclosed origin. They are mapped to UTC dates such that the first red team event happens on 2015-01-03: by default, models are trained on 2015-01-01, validated on 2015-01-02 and tested from 2015-01-03 to 2015-01-30.

=== "OpTC flows"

    ```shell
    python dataset_preprocessing/optc/create_database_optc_net.py OPTC_NET --lares_dir ${DATA_FOLDER}/lanl_optc_datasets/OPTC/preprocessed
    ```

    This builds the `optc_net` dataset with the FLOW START and FLOW MESSAGE events of all the hosts of [DARPA OpTC](https://github.com/FiveDirections/OpTC-data) (about 1.9 billion events). Files are read by `--num_workers` processes (16 by default), each writing its events to temporary files, which are then sorted into the dataset. As in LARES and Argus, graphs are built from FLOW START events only: set `construction.operations=all` to also use FLOW MESSAGE events, or a comma-separated list of operations (e.g. `construction.operations=FLOW_START`) to select events of any dataset.

    **Labels.** A flow is recorded several times in OpTC (FLOW START and FLOW MESSAGE events, by the hosts at both ends), and LARES labels only some of these records: e.g. in the lateral movement of 2019-09-24, the FLOW MESSAGE events of 561 flows are malicious but the FLOW START of only 56 of them. By default, the script labels malicious all the records of a malicious flow (same 5-tuple, in either direction, in the same LARES file): 5,165 malicious events (1,015 FLOW START) instead of 2,370 (115 FLOW START). Add `--no_label_propagation` to keep the labels of LARES as they are. A dataset built without propagation can be relabelled in place with `--relabel`; graphs built before must then be built again (the labels of edges are stored in the graphs).

## 5. Check the result

Run a system on the new dataset:

<div class="pm-families" markdown>

=== "Provenance"

    ```shell
    pidsmaker orthrus DATASET
    ```

=== "Network"

    ```shell
    pidsmaker euler LANL
    ```

</div>

## Sharing a dataset

A dataset is a folder, `<datasets_dir>/<database>/`, with a `manifest.json` and the Parquet files of its tables: copy the folder to share it. To publish it on Hugging Face, as the datasets of PIDSMaker, upload the folder to a dataset repository (`hf upload-large-folder REPO <folder> --repo-type dataset`, with one folder per database at the root of the repository), and download it with `pidsmaker download DATASET --repo REPO`.

Databases of the [legacy Postgres backend](installation.md#postgres) are exported to Parquet datasets with `dataset_preprocessing/export_parquet.py` (see `dataset_preprocessing/README.md`).

!!! tip "Adding a new dataset"
    To add a dataset that PIDSMaker does not support yet, provenance or network, see [Adding custom datasets](../guide/datasets.md#adding-custom-datasets).
