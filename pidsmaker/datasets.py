"""Downloads the datasets of PIDSMaker from Hugging Face.

    pidsmaker download CADETS_E3 THEIA_E3      # or: pidsmaker download all
    pidsmaker download --list                  # available datasets, sizes, downloaded or not

    from pidsmaker.datasets import download
    download("CADETS_E3")

Datasets are Parquet files (see `pidsmaker/utils/database.py`), stored in
`<datasets_dir>/<database>/`, with `datasets_dir` the `--datasets_dir` argument, else
$PIDSMAKER_DATASETS_DIR, else ~/.cache/pidsmaker/datasets. They are downloaded from a dataset
repository of Hugging Face with one folder per database (`<database>/manifest.json` and
`<database>/<table>/part-*.parquet`): $PIDSMAKER_HF_REPO, else `ubc-provenance/pidsmaker-datasets`.
A run downloads its dataset when it first opens it (`init_database_connection`), if it is
missing, unless $PIDSMAKER_AUTO_DOWNLOAD is `0`.
"""

import argparse
import json
import os
import sys

from pidsmaker.config.config import DATASET_DEFAULT_CONFIG
from pidsmaker.utils.database import DEFAULT_DATASETS_DIR

DEFAULT_HF_REPO = "ubc-provenance/pidsmaker-datasets"

# Size on disk of each database (GB), shown before downloads
DATABASE_SIZES_GB = {
    "PROVENANCE_BENIGN": 0.002,
    "atlasv2_edr": 0.16,
    "cadets_e3": 1.3,
    "cadets_e5": 32.9,
    "carbanakv2_edr": 1.1,
    "clearscope_e3": 0.51,
    "clearscope_e5": 5.0,
    "fivedirections_e3": 2.3,
    "fivedirections_e5": 30.3,
    "lanl": 0.40,
    "optc_051": 1.3,
    "optc_201": 1.6,
    "optc_501": 1.2,
    "optc_net": 11.8,
    "theia_e3": 0.71,
    "theia_e5": 3.9,
    "trace_e3": 9.4,
    "trace_e5": 74.3,
}


def hf_repo():
    return os.environ.get("PIDSMAKER_HF_REPO") or DEFAULT_HF_REPO


def get_datasets_dir(datasets_dir=None):
    return datasets_dir or os.environ.get("PIDSMAKER_DATASETS_DIR") or DEFAULT_DATASETS_DIR


def dataset_names():
    """Names of the datasets (as passed to the framework, e.g. CADETS_E3)."""
    return list(DATASET_DEFAULT_CONFIG.keys())


def resolve_database(name):
    """Database of a dataset, from its name (e.g. CADETS_E3 or cadets_e3, case-insensitive) or
    the name of its database (e.g. optc_051 for optc_h051)."""
    for dataset, config in DATASET_DEFAULT_CONFIG.items():
        if name.lower() in (dataset.lower(), config["database"].lower()):
            return config["database"]
    raise ValueError(f"Unknown dataset {name}. Available datasets: {', '.join(dataset_names())}")


def missing_files(database, datasets_dir=None):
    """Files of a downloaded dataset that are missing (all, if its manifest is missing)."""
    dataset_dir = os.path.join(get_datasets_dir(datasets_dir), database)
    manifest_path = os.path.join(dataset_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        return [manifest_path]
    with open(manifest_path) as f:
        manifest = json.load(f)
    paths = [
        os.path.join(dataset_dir, table, name)
        for table, info in manifest["tables"].items()
        for name in info["files"]
    ]
    return [p for p in paths if not os.path.exists(p)]


def is_downloaded(name, datasets_dir=None):
    return not missing_files(resolve_database(name), datasets_dir)


def download(*names, datasets_dir=None, repo_id=None, revision=None, force=False):
    """Downloads datasets (names of datasets or databases, or "all") and returns their folders.

    Datasets already downloaded are skipped, unless `force`. Interrupted downloads resume."""
    from huggingface_hub import snapshot_download

    if len(names) == 1 and isinstance(names[0], (list, tuple)):
        names = tuple(names[0])
    if any(n.lower() == "all" for n in names):
        names = dataset_names()
    databases = list(dict.fromkeys(resolve_database(n) for n in names))
    datasets_dir = get_datasets_dir(datasets_dir)
    repo_id = repo_id or hf_repo()

    dirs = []
    for database in databases:
        dataset_dir = os.path.join(datasets_dir, database)
        dirs.append(dataset_dir)
        if not force and not missing_files(database, datasets_dir):
            print(f"{database}: already downloaded in {dataset_dir}", flush=True)
            continue
        size = DATABASE_SIZES_GB.get(database)
        size = f" ({size:g} GB)" if size is not None else ""
        print(f"{database}: downloading{size} from {repo_id} to {dataset_dir}", flush=True)
        os.makedirs(datasets_dir, exist_ok=True)
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            allow_patterns=[f"{database}/*"],
            local_dir=datasets_dir,
            force_download=force,
        )
        missing = missing_files(database, datasets_dir)
        if missing:
            raise RuntimeError(
                f"{database}: {len(missing)} files are missing after the download from "
                f"{repo_id}, e.g. {missing[0]}"
            )
    return dirs


def ensure_downloaded(cfg):
    """Downloads the dataset of `cfg` if it is missing, unless $PIDSMAKER_AUTO_DOWNLOAD is 0.
    Processes that download it at the same time wait for each other (file locks of
    huggingface_hub)."""
    from pidsmaker.utils.database import PARQUET, database_backend, database_name

    if database_backend(cfg) != PARQUET:
        return
    database = database_name(cfg)
    datasets_dir = get_datasets_dir(getattr(cfg.database, "datasets_dir", None))
    if not missing_files(database, datasets_dir):
        return
    if os.environ.get("PIDSMAKER_AUTO_DOWNLOAD", "1") == "0":
        raise FileNotFoundError(
            f"Dataset {database} not found in {datasets_dir}: download it with "
            f"`pidsmaker download {cfg.dataset.name} --datasets_dir {datasets_dir}`"
        )
    download(database, datasets_dir=datasets_dir)


def list_datasets(datasets_dir=None):
    datasets_dir = get_datasets_dir(datasets_dir)
    print(f"Datasets in {datasets_dir} (Hugging Face repository: {hf_repo()})\n")
    print(f"{'Dataset':20s} {'Database':20s} {'Size (GB)':>10s}  Downloaded")
    for dataset, config in DATASET_DEFAULT_CONFIG.items():
        database = config["database"]
        size = DATABASE_SIZES_GB.get(database)
        size = f"{size:g}" if size is not None else "?"
        done = "yes" if not missing_files(database, datasets_dir) else ""
        print(f"{dataset:20s} {database:20s} {size:>10s}  {done}")


def cli(argv=None):
    parser = argparse.ArgumentParser(
        prog="pidsmaker download",
        description="Downloads datasets from Hugging Face. Interrupted downloads resume.",
    )
    parser.add_argument("datasets", nargs="*", help="Datasets (e.g. CADETS_E3 THEIA_E3), or `all`")
    parser.add_argument(
        "--list", action="store_true", help="Lists the datasets, their sizes and which are here"
    )
    parser.add_argument(
        "--datasets_dir",
        default=None,
        help="Where datasets are stored. Default: $PIDSMAKER_DATASETS_DIR, "
        f"else {DEFAULT_DATASETS_DIR}",
    )
    parser.add_argument(
        "--repo",
        default=None,
        help=f"Hugging Face dataset repository. Default: $PIDSMAKER_HF_REPO, else {DEFAULT_HF_REPO}",
    )
    parser.add_argument("--revision", default=None, help="Branch, tag or commit of the repository")
    parser.add_argument("--force", action="store_true", help="Downloads again, even if present")
    args = parser.parse_args(argv)

    if args.list:
        list_datasets(args.datasets_dir)
        return 0
    if not args.datasets:
        parser.print_help()
        return 1
    try:
        download(
            *args.datasets,
            datasets_dir=args.datasets_dir,
            repo_id=args.repo,
            revision=args.revision,
            force=args.force,
        )
    except ValueError as e:
        print(e, file=sys.stderr)
        return 1
    return 0
