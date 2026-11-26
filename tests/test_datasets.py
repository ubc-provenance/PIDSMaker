"""Downloads of datasets (pidsmaker/datasets.py), from a fake Hugging Face repository, and the
`pidsmaker` command."""

import fnmatch
import json
import os
import shutil

import pytest

import pidsmaker.datasets as datasets
from pidsmaker.cli import main as cli_main
from pidsmaker.config.pipeline import ROOT_PROJECT_PATH

pytestmark = pytest.mark.no_database


def write_fake_dataset(dataset_dir):
    """A dataset with a manifest and two tables."""
    files = {
        "event_table": ["part-2018-04-10.parquet", "part-2018-04-11.parquet"],
        "file_node_table": ["part-00000.parquet"],
    }
    for table, names in files.items():
        os.makedirs(os.path.join(dataset_dir, table))
        for name in names:
            with open(os.path.join(dataset_dir, table, name), "w") as f:
                f.write(name)
    manifest = {"tables": {t: {"files": names} for t, names in files.items()}}
    with open(os.path.join(dataset_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f)


@pytest.fixture
def fake_hub(tmp_path, monkeypatch):
    """A local repository with cadets_e3 and optc_051, served by a fake `snapshot_download`
    (same arguments and layout as huggingface_hub). Records its calls."""
    repo = tmp_path / "repo"
    for database in ["cadets_e3", "optc_051"]:
        write_fake_dataset(str(repo / database))
    calls = []

    def snapshot_download(repo_id, repo_type, revision, allow_patterns, local_dir, force_download):
        assert repo_type == "dataset"
        calls.append({"repo_id": repo_id, "revision": revision, "patterns": allow_patterns})
        for root, _, files in os.walk(repo):
            for f in files:
                rel = os.path.relpath(os.path.join(root, f), repo)
                if any(fnmatch.fnmatch(rel, p) for p in allow_patterns):
                    os.makedirs(os.path.join(local_dir, os.path.dirname(rel)), exist_ok=True)
                    shutil.copy(os.path.join(root, f), os.path.join(local_dir, rel))
        return local_dir

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    monkeypatch.setattr(
        datasets, "repo_databases", lambda repo_id, revision: sorted(os.listdir(repo))
    )
    monkeypatch.delenv("PIDSMAKER_HF_REPO", raising=False)
    monkeypatch.delenv("PIDSMAKER_AUTO_DOWNLOAD", raising=False)
    return {"repo": repo, "calls": calls, "dir": str(tmp_path / "datasets")}


def test_resolve_database():
    assert datasets.resolve_database("CADETS_E3") == "cadets_e3"
    assert datasets.resolve_database("cadets_e3") == "cadets_e3"
    assert datasets.resolve_database("optc_h051") == "optc_051"
    assert datasets.resolve_database("OPTC_051") == "optc_051"
    with pytest.raises(ValueError, match="Unknown dataset"):
        datasets.resolve_database("cadets_e4")


def test_every_database_has_a_size():
    # Except the sample of the tests, which replaces the database of CLEARSCOPE_E3 (conftest.py)
    databases = {
        c["database"]
        for c in datasets.DATASET_DEFAULT_CONFIG.values()
        if c["database"] != "clearscope_e3_sample"
    }
    assert databases <= set(datasets.DATABASE_SIZES_GB)
    assert len(datasets.DATABASE_SIZES_GB) == len(datasets.DATASET_DEFAULT_CONFIG)


def test_download(fake_hub):
    dirs = datasets.download("CADETS_E3", "cadets_e3", "optc_h051", datasets_dir=fake_hub["dir"])
    # Each database once, into <datasets_dir>/<database>, only its own files
    assert dirs == [os.path.join(fake_hub["dir"], d) for d in ["cadets_e3", "optc_051"]]
    assert [c["patterns"] for c in fake_hub["calls"]] == [["cadets_e3/*"], ["optc_051/*"]]
    assert fake_hub["calls"][0]["repo_id"] == datasets.DEFAULT_HF_REPO
    assert sorted(os.listdir(fake_hub["dir"])) == ["cadets_e3", "optc_051"]
    assert datasets.is_downloaded("CADETS_E3", fake_hub["dir"])

    # Downloaded datasets are skipped, unless forced
    datasets.download("CADETS_E3", datasets_dir=fake_hub["dir"])
    assert len(fake_hub["calls"]) == 2
    datasets.download("CADETS_E3", datasets_dir=fake_hub["dir"], force=True)
    assert len(fake_hub["calls"]) == 3

    # A missing file (e.g. interrupted download) downloads the dataset again
    os.remove(os.path.join(fake_hub["dir"], "cadets_e3", "event_table", "part-2018-04-11.parquet"))
    assert not datasets.is_downloaded("CADETS_E3", fake_hub["dir"])
    datasets.download("CADETS_E3", datasets_dir=fake_hub["dir"])
    assert len(fake_hub["calls"]) == 4
    assert datasets.is_downloaded("CADETS_E3", fake_hub["dir"])


def test_download_all(fake_hub):
    """`all` downloads the datasets of the repository."""
    datasets.download("all", datasets_dir=fake_hub["dir"])
    assert [c["patterns"] for c in fake_hub["calls"]] == [["cadets_e3/*"], ["optc_051/*"]]


def test_download_dataset_not_in_repository(fake_hub):
    with pytest.raises(RuntimeError, match="theia_e3 is not in the repository"):
        datasets.download("THEIA_E3", datasets_dir=fake_hub["dir"])


def test_download_repo_from_env(fake_hub, monkeypatch):
    monkeypatch.setenv("PIDSMAKER_HF_REPO", "someone/mirror")
    datasets.download("CADETS_E3", datasets_dir=fake_hub["dir"], revision="v1")
    assert fake_hub["calls"][0]["repo_id"] == "someone/mirror"
    assert fake_hub["calls"][0]["revision"] == "v1"


def test_download_incomplete_repository(fake_hub):
    os.remove(fake_hub["repo"] / "cadets_e3" / "file_node_table" / "part-00000.parquet")
    with pytest.raises(RuntimeError, match="1 files are missing"):
        datasets.download("CADETS_E3", datasets_dir=fake_hub["dir"])


def make_cfg(datasets_dir):
    from pidsmaker.config import get_runtime_required_args, get_yml_cfg

    args, _ = get_runtime_required_args(
        return_unknown_args=True,
        args=[
            "orthrus",
            "CADETS_E3",
            "--datasets_dir",
            datasets_dir,
        ],
    )
    return get_yml_cfg(args)


def test_connection_downloads_missing_dataset(fake_hub, monkeypatch):
    from pidsmaker.utils.database import init_database_connection

    cfg = make_cfg(fake_hub["dir"])
    monkeypatch.setenv("PIDSMAKER_AUTO_DOWNLOAD", "0")
    with pytest.raises(FileNotFoundError, match="pidsmaker download CADETS_E3"):
        init_database_connection(cfg)
    assert fake_hub["calls"] == []

    monkeypatch.setenv("PIDSMAKER_AUTO_DOWNLOAD", "1")
    datasets.ensure_downloaded(cfg)
    assert len(fake_hub["calls"]) == 1
    assert datasets.is_downloaded("CADETS_E3", fake_hub["dir"])
    datasets.ensure_downloaded(cfg)
    assert len(fake_hub["calls"]) == 1


def test_cli(fake_hub, capsys):
    assert cli_main([]) == 0
    assert "pidsmaker download" in capsys.readouterr().out
    assert cli_main(["download", "--list", "--datasets_dir", fake_hub["dir"]]) == 0
    assert "CADETS_E3" in capsys.readouterr().out
    assert cli_main(["download", "cadets_e4", "--datasets_dir", fake_hub["dir"]]) == 1
    assert cli_main(["download", "CADETS_E3", "--datasets_dir", fake_hub["dir"]]) == 0
    assert datasets.is_downloaded("CADETS_E3", fake_hub["dir"])


def test_resources_found():
    """config/ and Ground_Truth/ are found, from the repository or the installed wheel."""
    assert os.path.isfile(os.path.join(ROOT_PROJECT_PATH, "config", "orthrus.yml"))
    assert os.path.isdir(os.path.join(ROOT_PROJECT_PATH, "Ground_Truth", "orthrus", "E3-CADETS"))
