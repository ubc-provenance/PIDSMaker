# First, before the test modules import torch: sets the CPU threads (see pidsmaker/__init__.py)
import pidsmaker  # noqa: F401

# isort: split
import gzip
import os
import shutil

import psycopg2
import pytest

SAMPLE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "fixtures", "clearscope_e3_sample"
)
SAMPLE_TABLES = ["event_table", "subject_node_table", "file_node_table", "netflow_node_table"]
SAMPLE_DATASET = "CLEARSCOPE_E3"


def pytest_addoption(parser):
    parser.addoption("--device", action="store", default="cuda")
    parser.addoption(
        "--db",
        choices=["sample", "full"],
        default="sample",
        help="`sample` (default) runs the CLEARSCOPE_E3 tests on the small sample of tests/fixtures, loaded "
        "into a temporary database. `full` uses the full CLEARSCOPE_E3 database.",
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


def database_params():
    """Connection parameters of the database server, as used by the pipeline."""
    from pidsmaker.config import get_runtime_required_args

    args, _ = get_runtime_required_args(return_unknown_args=True, args=["tests", SAMPLE_DATASET])
    return dict(
        host=args.database_host,
        user=args.database_user,
        password=args.database_password,
        port=args.database_port,
    )


def admin_execute(params, *queries):
    connect = psycopg2.connect(database="postgres", **params)
    connect.autocommit = True
    with connect.cursor() as cur:
        for query in queries:
            cur.execute(query)
    connect.close()


def load_sample_database(name, params):
    """(Re)creates the database `name` with the tables of the CLEARSCOPE_E3 sample."""
    admin_execute(
        params,
        f'DROP DATABASE IF EXISTS "{name}";',
        f'CREATE DATABASE "{name}" TEMPLATE template0;',
    )
    connect = psycopg2.connect(database=name, **params)
    with connect, connect.cursor() as cur:
        with open(os.path.join(SAMPLE_DIR, "schema.sql")) as f:
            cur.execute(f.read())
        for table in SAMPLE_TABLES:
            with gzip.open(os.path.join(SAMPLE_DIR, f"{table}.csv.gz"), "rt") as f:
                columns = f.readline().strip()
                cur.copy_expert(f"COPY {table} ({columns}) FROM STDIN WITH (FORMAT csv);", f)
        cur.execute("ANALYZE;")
    connect.close()


@pytest.fixture(scope="session", autouse=True)
def clearscope_e3_database(request):
    """With `--db sample`, CLEARSCOPE_E3 is read from a database loaded with the sample."""
    if request.config.getoption("db") == "full" or all(
        item.get_closest_marker("no_database") for item in request.session.items
    ):
        yield
        return

    from pidsmaker.config import DATASET_DEFAULT_CONFIG

    # Specific to the process: pytest-xdist workers and concurrent test sessions each have their own
    name = f"pidsmaker_tests_{SAMPLE_DATASET.lower()}_{os.getpid()}"
    params = database_params()
    load_sample_database(name, params)
    dataset = DATASET_DEFAULT_CONFIG[SAMPLE_DATASET]
    original = dict(dataset)
    dataset["database"] = dataset["database_all_file"] = name
    yield
    dataset.update(original)
    admin_execute(params, f'DROP DATABASE IF EXISTS "{name}";')


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
