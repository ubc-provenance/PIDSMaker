# 10-min Docker Install with DARPA TC/OpTC Datasets 

## Download datasets

DARPA TC and OpTC are very large datasets that are significantly challenging to process. We provide our pre-processed versions of these datasets. We use a postgres database to store and load the data and provide the dumps to download.

Sizes for each database dump are as follow: **compressed** is the raw size of each dump, **uncompressed** is the size taken once loaded into the postgres table.

| Dataset       | Compressed (GB) | Uncompressed (GB) |
|---------------|------------------|-------------------|
| `CLEARSCOPE_E3` | 0.6              | 4.8               |
| `CADETS_E3`     | 1.4              | 10.1              |
| `THEIA_E3`      | 1.1              | 12                |
| `CLEARSCOPE_E5` | 6.2              | 49                |
| `CADETS_E5`     | 36               | 276               |
| `THEIA_E5`      | 5.8              | 36                |
| `OPTC_H051`     | 1.7              | 7.7               |
| `OPTC_H_501`    | 1.5              | 6.7               |
| `OPTC_H201`     | 2                | 9.1               |

**Steps:**

1. First [download the archive(s)](https://drive.google.com/drive/folders/1hqfz8__zVqb3QzBuOI2SxrW4lLIdYqFr) into a new `data` folder. We provide archives containing multiple datasets if their size is small, or provide the dump directly for larger datasets.
    On CLI, you can use `curl` with an authorization token (as explained [here](https://stackoverflow.com/a/67550427/10183259)):
    
    - Go to OAuth 2.0 Playground https://developers.google.com/oauthplayground/
    - In the `Select the Scope` box, paste `https://www.googleapis.com/auth/drive.readonly`
    - Click `Authorize APIs` and then `Exchange authorization code for tokens`
    - Copy the **Access token**
    - Run in terminal
    
    **Note**: Each call to curl downloads only a part of each file. You should call the same command multiple times to download the archvives at 100%

    ```sh
    mkdir data && cd data
    
    # optc_and_cadets_theia_clearscope_e3.tar
    curl -H "Authorization: Bearer ACCESS_TOKEN" -C - https://www.googleapis.com/drive/v3/files/1i7CkK20p21aBp3HGw46o-Uy31JpPC_Yx?alt=media -o optc_and_cadets_theia_clearscope_e3.tar

    # theia_clearscope_e5.tar
    curl -H "Authorization: Bearer ACCESS_TOKEN" -C - https://www.googleapis.com/drive/v3/files/1DfolzEa3PVz_6fGZUNEUm0sBP42LB7_1?alt=media -o theia_clearscope_e5.tar

    # cadets_e5.dump
    curl -H "Authorization: Bearer ACCESS_TOKEN" -C - https://www.googleapis.com/drive/v3/files/1Xiq7w0Ofz4jZG2PVFuNqi_i0fm28kRcT?alt=media -o cadets_e5.dump
    ```

2. Then uncompress the archives (this won't increase space)
    ```
    tar -xvf optc_and_cadets_theia_clearscope_e3.tar
    tar -xvf theia_clearscope_e5.tar
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

1. Set your paths in .env
    ```sh
    cd ..
    cp .env.local .env
    ```
    
    - In `.env`, set `INPUT_DIR` to the `data` folder path. Optionally, set `ARTIFACTS_DIR` to the folder where all generated files will go (multiple GBs).
    
    - Run the following command and set accordingly `HOST_UID`, `HOST_GID` and `USER_NAME` in `.env`.
        ```sh
        echo "HOST_UID=$(id -u)" && echo "HOST_GID=$(id -g)" && echo "USER_NAME=$(whoami)"
        ```

    - Then run:
        ```sh
        source .env
        ```

    - Then create the output artifacts folder if it doesn't exist yet and ensure it is owned by your user.
        ```sh
        mkdir ${ARTIFACTS_DIR} || chown ${USER_NAME} -R ${ARTIFACTS_DIR}
        ```

2. Build  and start the database container up:
    ```sh
    docker compose -p postgres -f compose-postgres.yml up -d --build
    ```
    Note: each time you modify variables in `.env`, update env variables using `source .env` prior to running `docker compose`.
    
3. In a terminal, get a shell into this container:
    ```sh
    docker compose -p postgres exec postgres bash
    ```
4. If you have enough space to uncompress all datasets locally (135 GB), run this script to load all databases:
    ```sh
    ./scripts/load_dumps.sh
    ```
    If you have limited space and want to load databases one by one, do:
    ```sh
    pg_restore -U postgres -h localhost -p 5432 -d DATASET /data/DATASET.dump
    ```
    !!! note
        If you want to parse raw data and create database from scratch, please follow the [guideline](./create-db-from-scratch.md) instead of running the above two commands.
6. Once databases are loaded, we won't need to touch this container anymore:
    ```sh
    exit
    ```

## Get into the PIDSMaker container

It is within the `pids` container that coding and experiments take place.

1. For VSCode users, we recommend using the [dev container](https://code.visualstudio.com/docs/devcontainers/create-dev-container) extension to directly open VSCode in the container. To do so, simply install the extension, then ctrl+shift+P and <i>Dev Containers: Open Folder in Container</i>.

2. The other alternative is to load the container manually and open a shell directly in your terminal.
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
