"""Standalone embedding-viz data exporter (independent of the pipeline).

Generates the ``embedding_viz_<dataset>_<suffix>_points.json`` /
``*_adj.json`` files that the web viewer reads, for a chosen run in the
artifacts folder — without running the PIDSMaker pipeline. The web server
(``pidsmaker.vizgen.web.viz_server``) then serves whatever exists in
artifacts; this is how new data gets produced when you don't want viz wired
into a pipeline run.

Usage::

    # latest evaluated run for the dataset
    python -m pidsmaker.vizgen.web.export <model> <dataset> --embeddings both

    # a specific run from the artifacts folder
    python -m pidsmaker.vizgen.web.export <model> <dataset> \
        --run /home/artifacts/evaluation/evaluation/<hash>/CADETS_E3

Requires that run's artifacts (trained model, scores, preprocessed graphs)
to be present — the exporter loads the model to compute encoder embeddings.
"""

# Print a line BEFORE importing the heavy GPU libraries (torch/cuML). Loading and
# initialising them can take several seconds, and if the GPU is busy the CUDA init
# can stall — so this first line reassures the live console that generation has
# actually started instead of leaving it blank.
print("[export] starting up — loading libraries (torch / cuML) and the GPU…", flush=True)

# ``pidsmaker.utils.utils`` runs ``nltk.download("punkt")`` at import time. Even when
# punkt is already installed, that call hits NLTK's server to refresh its package
# index — and on a host with slow/blocked outbound HTTPS the SSL handshake hangs
# indefinitely, so generation stalls at import with no further logs (looks like the
# regenerate button "does nothing"). We only need punkt for tokenisation and it ships
# in the environment, so cap that one download to a few seconds and swallow failures;
# the timeout is restored immediately after so DB/network I/O in generation is
# unaffected. (utils.py is upstream/original code we don't modify — we harden the
# call from the viz entry point that triggers it.)
import socket as _socket  # noqa: E402

import nltk as _nltk  # noqa: E402

_orig_nltk_download = _nltk.download


def _nltk_download_with_timeout(*args, **kwargs):
    prev = _socket.getdefaulttimeout()
    _socket.setdefaulttimeout(8)
    try:
        return _orig_nltk_download(*args, **kwargs)
    except Exception as exc:  # network unreachable / handshake timeout / etc.
        print(f"[export] skipping nltk.download (using local data): {exc}", flush=True)
        return False
    finally:
        _socket.setdefaulttimeout(prev)


_nltk.download = _nltk_download_with_timeout

from pidsmaker.vizgen.exporter import main  # noqa: E402  (after the startup print)

if __name__ == "__main__":
    main()
