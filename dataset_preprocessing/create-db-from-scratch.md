# Create database from scratch

### Download files

1. Create a folder `{data_folder}` and make sure that it has enough space to store the raw data

2. Install gdown
    ```shell
    pip install gdown
    ```

3. Download dataset

    #### For DARPA TC datasets (i.e. `clearscope_e3`, `cadets_e3`, `theia_e3`, `clearscope_e5`, `cadets_e5` and `theia_e5`), run:

    ```shell
    ./darpa_tc/scripts/download_{dataset}.sh {data_folder}
    ```
    where `{dataset}` can be either `clearscope_e3`, `cadets_e3`, `theia_e3`, `clearscope_e5`, `cadets_e5` or `theia_e5` and `{data_folder}` is the absolute path to the output folder where all raw files will be downloaded.

    Alternatively, you can [download the files manually](./download-files.md) by selecting download URLs from Google Drive.

    #### For DARPA OPTC dataset (i.e. `optc_h051`, `optc_h201` and `optc_h501`), run:
    ```shell
    python ./optc/download_dataset.py {dataset} {data_folder}
    ```
    where `{dataset}` can be either `optc_h051`, `optc_h201` or `optc_h501` and `{data_folder}` is the absolute path to the output folder where all raw files will be downloaded.

    > [!NOTE]   
    > Make sure `{data_folder}` is empty before downloading and parsing raw data.   
    > Remove all old files before downloading a new dataset.

### Install docker images
1. In ```compose-pidsmaker.yml```, uncomment ```- /path/to/raw/data:/data``` and set ```/path/to/raw/data``` as the data folder where you downloaded the downloaded dataset files (.gz), the java client (tar.gz) and the schema files (.avdl, .avsc)

2. Follow the [guidelines](../docs/docs/ten-minute-install.md) to build the docker image and open a shell of pidsmaker container

### Convert bin files to JSON (For TC datasets)

> [!NOTE]   
> This section is only for DARPA TC datasets (i.e. `clearscope_e3`, `cadets_e3`, `theia_e3`, `clearscope_e5`, `cadets_e5` and `theia_e5`)

> [!NOTE]  
> This may take multiple hours depending on the dataset.

In the container shell, convert the DARPA TC files 
```shell
./dataset_preprocessing/darpa_tc/scripts/uncompress_darpa_files.sh /data/
```

### Extract files (For OPTC datasets)
> [!NOTE]   
> This section is only for DARPA OPTC dataset (i.e. `optc_h051`, `optc_h201` and `optc_h501`)

In the container shell, run script to extract files:
```shell
./dataset_preprocessing/optc/extract_data.sh /data/
```


### Optional configurations
> [!NOTE] Set optional configs before filling the database if needed
- optionally, if using a specific postgres database instead of the postgres docker, update the connection config by setting `DATABASE_DEFAULT_CONFIG` within `pidsmaker/config/pipeline.py`.
- optionally, if using a specific postgres database instead of the postgres docker, copy [creating_database](../postgres/init-create-empty-databases.sh) to your database server and run it to create databases, and then copry [creating_tables](../postgres/init-create-databases.sh) to your server and run it to create tables


### Fill the database

#### For TC datasets (`clearscope_e3`, `cadets_e3`, `theia_e3`, `clearscope_e5`, `cadets_e5` and `theia_e5`)

Still in the container's shell, fill the database for the corresponding dataset by running this command:

```shell
python dataset_preprocessing/darpa_tc/create_database.py orthrus {dataset}
```
where `{dataset}` can be [`CLEARSCOPE_E5` | `CADETS_E5` | `THEIA_E5`]. 
Or 
```shell
python dataset_preprocessing/darpa_tc/create_database_e3.py orthrus {dataset}
```
where `{dataset}` can be [`CLEARSCOPE_E3` | `CADETS_E3` | `THEIA_E3`]

**Note:** Large storage capacity is needed to download, parse and save datasets and databases, as well as to run experiments. A single run can generate more than 15GB of artifact files on E3 datasets, and much more with larger E5 datasets.

#### For OPTC dataset (`optc_h051`, `optc_h201` and `optc_h501`)

Still in the container's shell, fill the database for the corresponding dataset by running this command:
```shell
python dataset_preprocessing/optc/parser_optc.py orthrus {dataset}
```
where `{dataset}` can be [`optc_h051` | `optc_h201` | `optc_h501`]
