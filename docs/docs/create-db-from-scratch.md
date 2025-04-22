# Create database from scratch (DARPA TC)

### Download files
You can download all required files directly by running:

```shell
pip install gdown
```
```shell
./settings/scripts/download_{dataset}.sh {data_folder}
```
where `{dataset}` can be either `clearscope_e3`, `cadets_e3`, `theia_e3`, `clearscope_e5`, `cadets_e5` or `theia_e5` and `{data_folder}` is the absolute path to the output folder where all raw files will be downloaded.

Alternatively, you can [download the files manually](./download-files.md) by selecting download URLs from Google Drive.

### Convert bin files to JSON

1. Follow the guidelines to build the docker image

2. In ```compose.yml```, set ```/path/of/data/folder``` as the data folder where you downloaded the downloaded dataset files (.gz), the java client (tar.gz) and the schema files (.avdl, .avsc)

3. Fire a shell in the container 
```
sudo docker compose exec pids bash
```

4. Convert the DARPA files 
```shell
./settings/scripts/uncompress_darpa_files.sh /data/
```

> [!NOTE]  
> This may take multiple hours depending on the dataset.

### Optional configurations
- optionally, if using a specific postgres database instead of the postgres docker, update the connection config by setting `DATABASE_DEFAULT_CONFIG` within `pidsmaker/config.py`.

- optionaly, if you want to change the output folder where generated files are stored, update accordingly the volume by uncommenting `./artifacts:/home/artifacts` in `compose.yml`.

### Fill the database

Still in the container's shell, fill the database for the corresponding dataset by running this command:

```shell
python scripts/python/create_database.py [CLEARSCOPE_E3 | CADETS_E3 | THEIA_E3 | CLEARSCOPE_E5 | CADETS_E5 | THEIA_E5]
```

**Note:** Large storage capacity is needed to download, parse and save datasets and databases, as well as to run experiments. A single run can generate more than 15GB of artifact files on E3 datasets, and much more with larger E5 datasets.