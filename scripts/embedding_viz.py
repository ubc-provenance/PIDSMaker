#!/usr/bin/env python3
"""Backward-compatible launcher for the standalone embedding-viz exporter.

The implementation now lives in ``pidsmaker.vizgen.exporter``. Prefer:

    python -m pidsmaker.vizgen.web.export <model> <dataset> [options]
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pidsmaker.vizgen.exporter import main

if __name__ == "__main__":
    main()
