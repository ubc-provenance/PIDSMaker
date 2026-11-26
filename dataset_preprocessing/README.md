# Dataset preprocessing

PIDSMaker already provides preprocessed versions of the DARPA TC and OpTC datasets, follow guidelines [here](https://ubc-provenance.github.io/PIDSMaker/ten-minute-install/).

If you want to install a dataset by yourself, we provide our preprocessing scripts to install those datasets. 
Follow the guidelines [here](https://ubc-provenance.github.io/PIDSMaker/create-db-from-scratch/).

This includes the network datasets (LANL and the network flows of OpTC), built from the preprocessed files of [LARES](https://github.com/TristanBilot/lares).

## Building a dataset

The scripts write a Parquet dataset (no Postgres server is needed) to
`<datasets_dir>/<database>/`, where `<datasets_dir>` is `--datasets_dir`, or
`$PIDSMAKER_DATASETS_DIR`, or `~/.cache/pidsmaker/datasets` by default, and `<database>` the
database of the dataset in `pidsmaker/config/config.py` (e.g. `cadets_e3`). The dataset is read
with `--database_backend parquet` (or `PIDSMAKER_DATABASE_BACKEND=parquet`). An existing dataset
is only replaced with `--overwrite`. The dataset is written to `<database>.tmp`, renamed when
complete.

```shell
# DARPA TC E3 (CADETS_E3, THEIA_E3) and E5 (CADETS_E5, THEIA_E5): the files listed in
# darpa_tc/filelist.py, in --raw_dir (default: /data/)
python dataset_preprocessing/darpa_tc/create_database_e3.py orthrus CADETS_E3 --raw_dir /data/
python dataset_preprocessing/darpa_tc/create_database_e5.py orthrus CADETS_E5 --raw_dir /data/

# DARPA OpTC (optc_h051, optc_h201, optc_h501): all the files of --raw_dir (default: /data/)
python dataset_preprocessing/optc/create_database_optc.py orthrus optc_h051 --raw_dir /data/

# LANL, from the files of LARES (add --with_flows for the network flows), or from the raw files
python dataset_preprocessing/lanl/create_database_lanl.py LANL --lares_dir /data/lanl_optc_datasets/LANL/preprocessed
python dataset_preprocessing/lanl/create_database_lanl.py LANL --auth_file /data/auth.txt.gz --redteam_file /data/redteam.txt.gz

# Network flows of OpTC, from the files of LARES (read by --num_workers processes)
python dataset_preprocessing/optc/create_database_optc_net.py OPTC_NET --lares_dir /data/lanl_optc_datasets/OPTC/preprocessed
```

The datasets are those that the previous versions of these scripts loaded into Postgres, as
exported by `export_parquet.py` (same tables, columns, values and order of rows, see
`pidsmaker/utils/dataset_writer.py`). Databases still in Postgres are exported with:

```shell
python dataset_preprocessing/export_parquet.py clearscope_e3 cadets_e3 --out_dir ~/.cache/pidsmaker/datasets
```
