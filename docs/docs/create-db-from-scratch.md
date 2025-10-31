# Install a dataset from scratch

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
PGPASSWORD=postgres pg_dump -U postgres -h postgres -p 5432 -F c -d DATASET -f DATASET.dump
```
