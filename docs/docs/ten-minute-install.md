# 10-min Docker Install with DARPA TC/OpTC Datasets 

## Download datasets

DARPA TC and OpTC are very large datasets that are significantly challenging to process. We provide our pre-processed versions of these datasets. We use a postgres database to store and load the data and provide the dumps to download.

Sizes for each database dump are as follow: **compressed** is the raw size of each dump, **uncompressed** is the size taken once loaded into the postgres table.

| Dataset       | Compressed (GB) | Uncompressed (GB) |
|---------------|------------------|-------------------|
| `CADETS_E3`     | 1.4              | 10              |
| `THEIA_E3`      | 1.1              | 12                |
| `CLEARSCOPE_E3` | 0.6              | 4.8               |
| `FIVEDIRECTIONS_E3`      | 3.2              | 22                |
| `TRACE_E3`      | 11              | 100                |
| `CADETS_E5`     | 36               | 276               |
| `THEIA_E5`      | 5.8              | 36                |
| `CLEARSCOPE_E5` | 6.2              | 49                |
| `FIVEDIRECTIONS_E5`      | 39              | 280                |
| `TRACE_E5`      | 91              | 710                |
| `OPTC_H201`     | 2                | 9               |
| `OPTC_H_501`    | 1.5              | 6.7               |
| `OPTC_H051`     | 1.7              | 7.7               |


**Steps:**

Datasets can be downloaded in two ways.

### Option 1: From the Google drive interface

Go [here](https://drive.google.com/drive/folders/1hqfz8__zVqb3QzBuOI2SxrW4lLIdYqFr) and download the database dumps associated with the datasets you want.

### Option 2: From CLI

On most servers, it is more practical to download from CLI.
To do so, you must first get a Google API authorization token (as explained [here](https://stackoverflow.com/a/67550427/10183259)):
    
- First, go to [OAuth 2.0 Playground](https://developers.google.com/oauthplayground/)
- In the `Select the Scope` box, paste `https://www.googleapis.com/auth/drive.readonly`
- Log into your Google account
- Click `Authorize APIs` and then `Exchange authorization code for tokens`
- Copy the **access_token**
    
Then use this script to download the datasets directly from CLI:

```sh
# Example 1: Comma-separated datasets
./download_datasets.sh cadets_e3,optc_h201,clearscope_e5 YOUR_ACCESS_TOKEN

# Example 2: All datasets
./download_datasets.sh all YOUR_ACCESS_TOKEN
```

Alternatively, here are the [guidelines](./create-db-from-scratch.md) to manually create the databases from the official DARPA TC files.

## Docker Install

1. If not installed, install Docker following the [steps from the official site](https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository) and [avoid using sudo](https://docs.docker.com/engine/install/linux-postinstall/#manage-docker-as-a-non-root-user).

2. Then, install dependencies for CUDA support with Docker:

```shell
# Add the NVIDIA package repository
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Update and install
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Restart services
sudo systemctl restart docker
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

## Load databases
We create two containers: one that runs the postgres database, the other runs the Python env and the pipeline.

### 1. Set your paths in .env

```sh
cp .env.local .env
```

 In `.env`, set `INPUT_DIR` to the `data` folder path. Optionally, set `ARTIFACTS_DIR` to a path where all generated files will go (multiple GBs).


### 2. Build  and start the database container up:

```sh
docker compose -p postgres -f compose-postgres.yml up -d --build
```
Note: each time you modify variables in `.env`, update env variables using `source .env` prior to running `docker compose`.
    
### 3. Get a shell into the postgres container

```sh
docker compose -p postgres exec postgres bash
```

### 4. Load database dumps

If you have enough space to uncompress all datasets you have downloaded locally in the `data` folder, run this script:

```sh
./scripts/load_dumps.sh
```

If you have limited space and want to load databases one by one, do:

```sh
pg_restore -U postgres -h localhost -p 5432 -d DATASET /data/DATASET.dump
```

!!! note
    If you want to parse raw data and create database from scratch, please follow the [guideline](./create-db-from-scratch.md) instead of running the above two commands.

Once databases are loaded, we won't need to touch this container anymore:

```sh
exit
```

## Get into the PIDSMaker container

It is within the `pids` container that coding and experiments take place.

### 1. VSCode Devcontainer approach


For VSCode users, we recommend using the [dev container](https://code.visualstudio.com/docs/devcontainers/create-dev-container) extension to directly open VSCode in the container. To do so, simply install the extension, then ctrl+shift+P and <i>Dev Containers: Open Folder in Container</i>.


### 2. Manual approach

The other alternative is to load the container manually and open a shell directly in your terminal.

```sh
docker compose -f compose-pidsmaker.yml up -d --build
docker compose exec pids bash
```

It's in this container that the python env is installed and where the framework will be used.

## Weights & Biases interface

W&B is used as the default interface to visualize and historize experiments, we **highly** encourage to use it. You can [create an account](https://wandb.ai/site/) if not already done. Log into your account from CLI by pasting your API key, obtained via your W&B dashboard:

```sh
wandb login
```

Then you can push the logs and results of experiments to the interface using the `--wandb` arg or when calling `./run.sh`.
