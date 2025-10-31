# Install Framework using Singularity/Apptainer

For quick installation on environments where Docker is not available (such as HPC clusters), you can use Singularity/Apptainer. This guide assumes Singularity/Apptainer is already installed on your system.

## Setup Process

### 1. Database Setup
The Makefile in `./scripts/apptainer/Makefile` provides easy environment setup:

```bash
cd ./scripts/apptainer
make full-setup
```

This command will:
- Download and run a PostgreSQL container through Singularity/Apptainer
- Load database dumps by executing the `load_dumps.sh` script

### 2. Container Management
Once the database is ready:
- Stop the container: `make down`
- Start it again: `make up`

### 3. Dependencies Installation
Install all required dependencies using conda:

```bash
conda env create -f ./scripts/apptainer/environment.yml
conda activate pids
```

## Running the Framework

Once both the database and conda environment are ready, run the framework with:

```bash
python pidsmaker/main.py SYSTEM DATASET --artifact_dir ./artifacts/ --database_host localhost
```
