"""The `pidsmaker` command (installed with the package).

- `pidsmaker <model> <dataset> [args]`: runs the pipeline, as pidsmaker/main.py, with the
  environment of reproducible runs (`REPRODUCIBLE_ENV`). With `--background`, the run is
  detached from the terminal (it survives logging out) and its output goes to a log file.
- `pidsmaker download <dataset> ... | all`: downloads datasets (see pidsmaker/datasets.py)
"""

import os
import subprocess
import sys
import time

USAGE = """usage: pidsmaker <model> <dataset> [args]     run a system on a dataset
       pidsmaker <model> <dataset> [args] --background
                                              same, in the background, with a log file
       pidsmaker download <dataset> ...       download datasets (or `all`), --list to list them

Run `pidsmaker <model> <dataset> --help` or `pidsmaker download --help` for the arguments."""

# Set for the runs of the pipeline when they are not set already:
# - a fixed hash seed, so that sets of strings are iterated in the same order in every run.
#   Python reads it at startup: the command restarts itself to set it;
# - the workspace that cuBLAS needs for deterministic algorithms (`training.deterministic`).
REPRODUCIBLE_ENV = {"PYTHONHASHSEED": "0", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}


def log_path(argv):
    """Log file of a run in the background, in the working directory."""
    name = "_".join(["pidsmaker"] + [a for a in argv[:2] if not a.startswith("-")])
    path = os.path.abspath(f"{name}_{time.strftime('%Y%m%d-%H%M%S')}.log")
    base, n = path[: -len(".log")], 1
    while os.path.exists(path):
        path, n = f"{base}_{n}.log", n + 1
    return path


def run_pipeline(argv):
    background = "--background" in argv
    argv = [a for a in argv if a != "--background"]
    missing = {k: v for k, v in REPRODUCIBLE_ENV.items() if k not in os.environ}
    env = {**os.environ, **missing}
    command = [sys.executable, "-m", "pidsmaker.cli", *argv]

    if background:
        log = log_path(argv)
        with open(log, "ab") as out:
            proc = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=subprocess.STDOUT,
                env={**env, "PYTHONUNBUFFERED": "1"},
                start_new_session=True,
            )
        print(f"Running in the background (PID {proc.pid}). Log: {log}")
        return 0

    if missing:
        os.execve(sys.executable, command, env)

    from pidsmaker.main import cli

    cli(argv)
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(USAGE)
        return 0
    if argv[0] == "download":
        from pidsmaker.datasets import cli

        return cli(argv[1:])
    return run_pipeline(argv)


if __name__ == "__main__":
    sys.exit(main())
