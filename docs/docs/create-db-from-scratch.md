# Installing a dataset from scratch

PIDSMaker comes by default with pre-processed versions of DARPA datasets.
If you want to install them from scratch using the official public files, follow this guide.

### Download files

1. Create an empty folder `DATA_FOLDER` and make sure that you have enough space to download the raw data
    ```shell
    DATA_FOLDER=./data
    mkdir ${DATA_FOLDER}
    ```

2. Install gdown
    ```shell
    pip install gdown
    ```

3. Download dataset

    For DARPA TC datasets, run:

    ```shell
    ./dataset_preprocessing/darpa_tc/scripts/download_DATASET.sh ${DATA_FOLDER}
    ```
    where `DATASET` can be either `clearscope_e3`, `cadets_e3`, `theia_e3`, `clearscope_e5`, `cadets_e5` or `theia_e5` and `DATA_FOLDER` is the absolute path to the output folder where all raw files will be downloaded.

    Alternatively, you can [download the files manually](./download-files.md) by selecting download URLs from Google Drive.

    For DARPA OpTC datasets, run:
    ```shell
    python ./dataset_preprocessing/optc/download_dataset.py DATASET ${DATA_FOLDER}
    ```
    where `DATASET` can be either `optc_h051`, `optc_h201` or `optc_h501` and `DATA_FOLDER` is the absolute path to the output folder where all raw files will be downloaded.

    !!! note   
        Make sure `DATA_FOLDER` is empty before downloading and parsing raw data.   
        Remove all old files before downloading a new dataset.

### Install docker images
1. In ```compose-pidsmaker.yml```, uncomment ```- /path/to/raw/data:/data``` and set ```/path/to/raw/data``` as the DATA_FOLDER where you downloaded the downloaded dataset files (.gz), the java client (tar.gz) and the schema files (.avdl, .avsc)

2. Follow the [guidelines](./ten-minute-install.md) to build the docker image and open a shell of pidsmaker container

### Extract files (TC)
This part is only for DARPA TC datasets (i.e. `clearscope_e3`, `cadets_e3`, `theia_e3`, `clearscope_e5`, `cadets_e5` and `theia_e5`)

In the `pidscontainer`, uncompress the DARPA TC files by running:
```shell
./dataset_preprocessing/darpa_tc/scripts/uncompress_darpa_files.sh /data/
```

!!! note  
    This may take multiple hours depending on the dataset.

### Extract files (OpTC)
This part is only for DARPA OpTC dataset (i.e. `optc_h051`, `optc_h201` and `optc_h501`)

In the `pidscontainer`, extract the files by running:
```shell
./dataset_preprocessing/optc/extract_data.sh /data/
```


### Optional configurations
- Set optional configs before filling the database if needed. If using a specific postgres database instead of the postgres docker, update the connection config by setting `DATABASE_DEFAULT_CONFIG` within `pidsmaker/config/pipeline.py`.
- If using a specific postgres database instead of the postgres docker, copy [creating_database](../postgres/init-create-empty-databases.sh) to your database server and run it to create databases, and then copry [creating_tables](../postgres/init-create-databases.sh) to your server and run it to create tables.


### Fill the database (TC)

For TC datasets (`clearscope_e3`, `cadets_e3`, `theia_e3`, `clearscope_e5`, `cadets_e5` and `theia_e5`)

Still in the container's shell, fill the database for the corresponding dataset by running this command:

```shell
python dataset_preprocessing/darpa_tc/create_database_e5.py orthrus DATASET
```
where `DATASET` can be [`CLEARSCOPE_E5` | `CADETS_E5` | `THEIA_E5`]. 
Or 
```shell
python dataset_preprocessing/darpa_tc/create_database_e3.py orthrus DATASET
```
where `DATASET` can be [`CLEARSCOPE_E3` | `CADETS_E3` | `THEIA_E3`]

**Note:** Large storage capacity is needed to download, parse and save datasets and databases, as well as to run experiments. A single run can generate more than 15GB of artifact files on E3 datasets, and much more with larger E5 datasets.

### Fill the database (OpTC)
For OpTC dataset (`optc_h051`, `optc_h201` and `optc_h501`)

Still in the container's shell, fill the database for the corresponding dataset by running this command:
```shell
python dataset_preprocessing/optc/create_database_optc.py orthrus DATASET
```
where `DATASET` can be [`optc_h051` | `optc_h201` | `optc_h501`]

## Network datasets (LANL and OpTC flows)

Network datasets have hosts as nodes and authentications or network flows as edges. They are built from the preprocessed files of [LARES](https://github.com/TristanBilot/lares), which also provide the labels of malicious events used as ground truth.
Unlike provenance datasets, their tables follow the dataset's schema (`pidsmaker/config/schemas.py`): no SQL initialization is needed, as the scripts below create the database and its tables.
Any system runs on them: for example, `construction.edge_type=logon_type` makes the edge type predicted by Velox the logon type of LANL authentications, and `batching.edge_features=edge_type,edge_attrs` gives the edge attributes (ports, protocols, logon types, ...) to the GNN.

### Download files

In `DATA_FOLDER`, download and extract the LARES archive (14.35 GB, 18.44 GB once extracted):

```shell
curl -L -o ${DATA_FOLDER}/lanl_optc_datasets.tar.gz "https://drive.usercontent.google.com/download?id=10fdDttLXP8zYwlrhqFzOe_kEvDGokfVY&export=download&confirm=t"
tar -xzf ${DATA_FOLDER}/lanl_optc_datasets.tar.gz -C ${DATA_FOLDER}
```

### Fill the database (LANL)

In the `pidscontainer`, run:

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

In the `pidscontainer`, run:

```shell
python dataset_preprocessing/optc/create_database_optc_net.py OPTC_NET --lares_dir /data/lanl_optc_datasets/OPTC/preprocessed
```

This creates the `optc_net` database with the FLOW START and FLOW MESSAGE events of all the hosts of [DARPA OpTC](https://github.com/FiveDirections/OpTC-data) (about 1.9 billion events).

!!! note
    As in LARES and Argus, graphs are built from FLOW START events only. Set `construction.operations=all` to also use FLOW MESSAGE events, or a comma-separated list of operations (e.g. `construction.operations=FLOW_START`) to select events of any dataset.

The official network flows of OpTC (`ecar-bro` folder of its Google Drive) can be downloaded with:

```shell
python dataset_preprocessing/optc/download_optc_net.py ${DATA_FOLDER}/optc_net
```

## Verification
Your databases are now built and filled with data. 
If you are not already in the `pidsmaker` container, run:

```sh
docker compose -p postgres -f compose-postgres.yml up -d --build
docker compose -f compose-pidsmaker.yml up -d --build
docker compose exec pids bash
```

Then run inside the container:

```sh
python pidsmaker/main.py orthrus DATASET
```

To export your database as a dump file for sharing, do:
```sh
PGPASSWORD=postgres pg_dump -U postgres -h postgres -p 5432 -F c -d DATABASE -f DATABASE.dump
```
where `DATABASE` is the name of the database (e.g. `cadets_e3`, `lanl` or `optc_net`). Dumps are restored in a database named after their file, so keep the name of the database. The `pg_dump` client must have the same major version as the Postgres server.
