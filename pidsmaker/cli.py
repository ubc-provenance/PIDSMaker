"""The `pidsmaker` command (installed with the package).

- `pidsmaker <model> <dataset> [args]`: runs the pipeline, as pidsmaker/main.py
- `pidsmaker download <dataset> ... | all`: downloads datasets (see pidsmaker/datasets.py)
"""

import sys

USAGE = """usage: pidsmaker <model> <dataset> [args]     run a system on a dataset
       pidsmaker download <dataset> ...       download datasets (or `all`), --list to list them

Run `pidsmaker <model> <dataset> --help` or `pidsmaker download --help` for the arguments."""


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(USAGE)
        return 0
    if argv[0] == "download":
        from pidsmaker.datasets import cli

        return cli(argv[1:])

    from pidsmaker.main import cli

    cli(argv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
