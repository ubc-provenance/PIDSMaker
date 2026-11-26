# First, before the test modules import torch: sets the CPU threads (see pidsmaker/__init__.py)
import pidsmaker  # noqa: F401

# isort: split
import os
import shutil

import pytest

# The sample of CLEARSCOPE_E3, a dataset in the folder of fixtures (see fixtures/make_clearscope_e3_sample.py)
FIXTURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
SAMPLE_DATABASE = "clearscope_e3_sample"
SAMPLE_DATASET = "CLEARSCOPE_E3"


def pytest_addoption(parser):
    parser.addoption("--device", action="store", default="cuda")
    parser.addoption(
        "--db",
        choices=["sample", "full"],
        default="sample",
        help="`sample` (default) runs the CLEARSCOPE_E3 tests on the small sample of "
        "tests/fixtures. `full` uses the full CLEARSCOPE_E3 dataset, read from "
        "$PIDSMAKER_DATASETS_DIR (else ~/.cache/pidsmaker/datasets), downloaded if missing.",
    )
    parser.addoption(
        "--artifact_dir",
        default=None,
        help="Directory where the tests write their artifacts (default: a temporary directory). "
        "It is removed before and after the tests.",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "no_database: the test does not need the database (when all the selected "
        "tests have this marker, the sample database is not loaded)"
    )


def worker_suffix():
    """Suffix of the per-worker resources with pytest-xdist (empty without xdist)."""
    worker = os.environ.get("PYTEST_XDIST_WORKER")
    return f"_{worker}" if worker else ""


@pytest.fixture(scope="session", autouse=True)
def clearscope_e3_database(request):
    """With `--db sample`, CLEARSCOPE_E3 is read from the sample of tests/fixtures."""
    if request.config.getoption("db") == "full" or all(
        item.get_closest_marker("no_database") for item in request.session.items
    ):
        yield
        return

    from pidsmaker.config import DATASET_DEFAULT_CONFIG

    os.environ["PIDSMAKER_DATASETS_DIR"] = FIXTURES_DIR
    dataset = DATASET_DEFAULT_CONFIG[SAMPLE_DATASET]
    original = dict(dataset)
    dataset["database"] = dataset["database_all_file"] = SAMPLE_DATABASE
    yield
    dataset.update(original)


@pytest.fixture(scope="session")
def artifact_dir(request, tmp_path_factory):
    """Artifact directory of the tests, specific to each pytest-xdist worker."""
    path = request.config.getoption("artifact_dir")
    if path:
        path = os.path.join(path, f"tests{worker_suffix()}") + "/"
    else:
        path = str(tmp_path_factory.mktemp("artifacts")) + "/"
    shutil.rmtree(path, ignore_errors=True)
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture(scope="session")
def device(request):
    name_value = request.config.option.device
    if name_value is None:
        pytest.skip()
    return name_value
