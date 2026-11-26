"""Process tags of R-Caid (Goyal et al., IEEE S&P 2024, Section 4.3).

R-Caid annotates each process with a tag, "the process name" extracted from "the file path to
its binary executable", and trains its GNN to predict it. So that the tag does not appear in the
node features, the process name is truncated from the path of the process before computing its
features.

Process labels are made of the attributes of `construction.node_label_features.subject`
joined by spaces, starting with the path of the executable (e.g. `path, cmd_line`):
`/bin/dash /bin/sh -e script` (THEIA) or `None sleep` (CADETS, whose processes have no path).
"""

import os

_NULL = ("", "None", "none", "null")


def _path_and_rest(label):
    label = (label or "").strip()
    if not label:
        return "", ""
    path, _, rest = label.partition(" ")
    return path, rest.strip()


def process_tag(label):
    """Name of the executable of a process: the basename of its path, or, without path, of the
    first word of the rest of its label (e.g. its command line)."""
    path, rest = _path_and_rest(label)
    if path in _NULL:
        path = rest.split(" ")[0] if rest else ""
    if path in _NULL:
        return ""
    return os.path.basename(path.replace("\\", "/").rstrip("/")) or path


def truncate_process_name(label):
    """Features label of a process: the directory of its executable, without its name nor the
    rest of its label (which usually repeats the name, e.g. in the command line)."""
    path, _ = _path_and_rest(label)
    if path in _NULL:
        return ""
    path = path.replace("\\", "/").rstrip("/")
    return path.rsplit("/", 1)[0] + "/" if "/" in path else ""
