"""
Noise filtering for provenance graph edges in T5 corpus construction.

Provenance graphs contain many low-information edges that dominate the
training signal and prevent meaningful clustering of entity embeddings.
This module identifies and filters out noisy edges so that each entity's
neighborhood reflects its actual functional role rather than generic
system behavior shared by all processes.

Categories of noise:
  1. Shared library loads: every process reads libc, libm, ld.so.cache, etc.
  2. System pseudo-files: /sys, /proc, /dev pseudo-filesystem reads
  3. Common config files: /etc/passwd, /etc/nsswitch.conf, /etc/group, etc.
  4. Locale/timezone data: loaded by nearly all userspace processes
  5. Null/empty labels: (null) or empty node labels carry no information
  6. Container runtime: runc, containerd, docker infrastructure commands
  7. Udev/device manager: /run/udev/data, device major:minor artifacts
  8. Binary/garbage: non-printable characters in labels
  9. Temporary lock files: .# atomics with random suffixes
 10. Numeric-only device IDs: PCI bus addresses, SCSI targets, loop devices

The filter operates on (center_node, neighbor_node, edge_type) triples
using the indexid2msg mapping to resolve node types and labels.
"""

import re
from functools import lru_cache


# ── Binary / non-printable character detection ──────────────────────────
_NON_PRINTABLE_RE = re.compile(r"[\x00-\x08\x0e-\x1f\x7f-\xff]")

# ── Shared library patterns ──────────────────────────────────────────────
# Matches /lib/**/*.so*, /usr/lib/**/*.so*, and versioned variants
_SHARED_LIB_RE = re.compile(
    r"^/(?:usr/)?lib(?:64)?/.*\.so(?:\.\d+)*$"
)

# ── Dynamic linker cache ─────────────────────────────────────────────────
_LINKER_FILES = frozenset({
    "/etc/ld.so.cache",
    "/etc/ld.so.preload",
    "/etc/ld.so.conf",
    "/etc/ld.so.nohwcap",
})

# ── Locale / timezone files ──────────────────────────────────────────────
_LOCALE_PREFIXES = (
    "/usr/lib/locale/",
    "/usr/share/locale/",
    "/usr/share/i18n/",
    "/usr/share/zoneinfo/",
    "/etc/localtime",
    "/etc/timezone",
)

# ── Common config files read by almost every process ─────────────────────
_COMMON_CONFIG_FILES = frozenset({
    "/etc/passwd",
    "/etc/group",
    "/etc/shadow",
    "/etc/nsswitch.conf",
    "/etc/host.conf",
    "/etc/hosts",
    "/etc/resolv.conf",
    "/etc/gai.conf",
    "/etc/ssl/openssl.cnf",
    "/etc/machine-id",
    "/etc/hostname",
})

# ── System pseudo-filesystem top-level entries ───────────────────────────
_SYSFS_TOPLEVEL = frozenset({
    "/sys", "/proc", "/dev",
    # sysfs sub-mounts that appear as standalone nodes
    "/class", "/block", "/devices", "/virtual", "/system", "/bus",
    "/module", "/firmware", "/kernel", "/power", "/fs",
})

# ── /dev pseudo-devices (noise) vs real devices (keep) ───────────────────
_DEV_NOISE_RE = re.compile(
    r"^/dev/(?:"
    r"null|zero|random|urandom|"       # standard pseudo-devices
    r"fd/\d+|"                         # file descriptors
    r"pts/\d+|"                        # pseudo-terminals
    r"tty\d*|"                         # terminals
    r"std(?:in|out|err)|"              # standard streams
    r"char/[\d:.]+(?:\.tmp[^\s]*)?|"   # character device nodes (udev artifacts)
    r"block/[\d:.]+(?:\.tmp[^\s]*)?|"  # block device nodes (udev artifacts)
    r"loop\d+|"                        # loop devices
    r"scap\d*|"                        # scap driver devices
    r"disk/by-id/.*"                   # disk symlinks (udev-generated)
    r")$"
)

# ── /proc filesystem entries ─────────────────────────────────────────────
_PROC_NOISE_RE = re.compile(
    r"^/proc(?:/\d+)?(?:/|$)"  # /proc, /proc/123, /proc/123/...
)

# ── Udev / device manager runtime files ──────────────────────────────────
_UDEV_NOISE_PREFIXES = (
    "/run/udev/",
    "/run/systemd/netif/",
)

# ── Containerd / Docker runtime paths ────────────────────────────────────
_CONTAINER_FILE_PREFIXES = (
    "/run/containerd/",
    "/run/docker/",
    "/var/run/docker/",
    "/var/run/containerd/",
    "/tmp/runc-process",
)

# ── Temporary lock files (.#<random>) ────────────────────────────────────
_LOCK_FILE_RE = re.compile(r"/\.#[^/]+$")

# ── Numeric-only device/bus identifiers ──────────────────────────────────
# PCI addresses (/0000:01:00.0), SCSI targets (/target0:3:111),
# major:minor (/8:2), loop devices (/loop10)
_NUMERIC_DEVICE_RE = re.compile(
    r"^/(?:"
    r"\d{4}:[0-9a-f]{2}:[0-9a-f]{2}\.\d|"  # PCI bus address
    r"target\d+:\d+:\d+|"                    # SCSI target
    r"\d+:\d+(?::\d+)*|"                     # major:minor[:lun]
    r"loop\d+"                                # loop device
    r")$"
)

# ── Null / empty / placeholder labels ────────────────────────────────────
_NULL_LABELS = frozenset({
    "(null)", "null", "(none)", "", " ",
})

# ── Short path-only labels with no semantic content ──────────────────────
_SHORT_NOISE_RE = re.compile(
    r"^/[a-zA-Z0-9_-]{1,4}$"  # /id, /11, /bdi, etc.
)

# ── Systemd transient/unit scope files ───────────────────────────────────
_SYSTEMD_TRANSIENT_RE = re.compile(
    r"^/run/user/\d+/systemd/(?:transient|units)/"
)


# ── Container runtime process patterns ───────────────────────────────────
_CONTAINER_PROC_RE = re.compile(
    r"(?:"
    r"runc:\[\d+:(?:INIT|CHILD)\]|"            # runc init/child
    r"^(?:\[NO_CMD\]\s*)?/usr/bin/runc\b|"     # runc binary
    r"--root\s+/var/run/docker/runtime-runc/|"  # runc exec commands
    r"containerd-shim"                           # containerd shim
    r")",
    re.IGNORECASE,
)

# ── Hex-encoded process names ────────────────────────────────────────────
# e.g., "53616E64626F7820466F726B6564" = "Sandbox Forked"
_HEX_LABEL_RE = re.compile(r"^[0-9a-f]{20,}$")

# ── Generic placeholder process names ────────────────────────────────────
_PLACEHOLDER_PROC_LABELS = frozenset({
    "(spawn)", "(imedated)", "(forked)",
})


@lru_cache(maxsize=65536)
def _has_binary_garbage(label: str) -> bool:
    """Check if a label contains non-printable / non-UTF8 characters."""
    return bool(_NON_PRINTABLE_RE.search(label))


@lru_cache(maxsize=65536)
def _is_noisy_file(label: str) -> bool:
    """Check if a FILE node label represents system noise."""
    label_lower = label.lower().strip()

    # Null/empty labels
    if label_lower in _NULL_LABELS:
        return True

    # Binary garbage
    if _has_binary_garbage(label):
        return True

    # Short meaningless path stubs
    if _SHORT_NOISE_RE.match(label_lower):
        return True

    # Shared libraries
    if _SHARED_LIB_RE.match(label_lower):
        return True

    # Dynamic linker files
    if label_lower in _LINKER_FILES:
        return True

    # Locale/timezone data
    for prefix in _LOCALE_PREFIXES:
        if label_lower.startswith(prefix):
            return True

    # Common config files
    if label_lower in _COMMON_CONFIG_FILES:
        return True

    # sysfs top-level pseudo-entries
    if label_lower in _SYSFS_TOPLEVEL:
        return True

    # /dev pseudo-devices
    if _DEV_NOISE_RE.match(label_lower):
        return True

    # /proc filesystem
    if _PROC_NOISE_RE.match(label_lower):
        return True

    # /sys/** deep paths (all of sysfs is noise for embedding purposes)
    if label_lower.startswith("/sys/"):
        return True

    # Udev / device manager runtime paths
    for prefix in _UDEV_NOISE_PREFIXES:
        if label_lower.startswith(prefix):
            return True

    # Containerd / Docker runtime paths
    for prefix in _CONTAINER_FILE_PREFIXES:
        if label_lower.startswith(prefix):
            return True

    # Temporary lock files (.#<random>)
    if _LOCK_FILE_RE.search(label_lower):
        return True

    # Numeric-only device/bus identifiers
    if _NUMERIC_DEVICE_RE.match(label_lower):
        return True

    # Systemd transient unit scopes
    if _SYSTEMD_TRANSIENT_RE.match(label_lower):
        return True

    return False


@lru_cache(maxsize=65536)
def _is_noisy_process(label: str) -> bool:
    """Check if a PROC node label is too generic to be informative."""
    label_lower = label.lower().strip()

    # Null/empty labels
    if label_lower in _NULL_LABELS:
        return True

    # Binary garbage in command line
    if _has_binary_garbage(label):
        return True

    # Container runtime processes (runc, containerd-shim)
    if _CONTAINER_PROC_RE.search(label):
        return True

    # Strip [NO_CMD] prefix for remaining checks
    bare = label_lower
    if bare.startswith("[no_cmd]"):
        bare = bare[len("[no_cmd]"):].strip()

    # Hex-encoded process names (e.g., "53616E64626F7820466F726B6564")
    if _HEX_LABEL_RE.match(bare):
        return True

    # Generic placeholder names
    if bare in _PLACEHOLDER_PROC_LABELS:
        return True

    return False


def is_noisy_edge(center_node, neighbor_node, edge_type, indexid2msg):
    """Determine whether a (center, neighbor, edge_type) triple is noise.

    An edge is noisy if the neighbor is a hub entity that provides no
    discriminative information about the center entity's functional role.

    Args:
        center_node: Node ID of the center entity
        neighbor_node: Node ID of the neighbor
        edge_type: Edge label string (e.g., "EVENT_READ", "EVENT_WRITE")
        indexid2msg: Dict mapping node IDs to (node_type, label_str)

    Returns:
        True if the edge should be filtered out
    """
    # Resolve neighbor identity
    if neighbor_node not in indexid2msg:
        return False  # Unknown node — keep to be safe

    neigh_type, neigh_label = indexid2msg[neighbor_node]

    # Filter noisy FILE neighbors
    if neigh_type == "file":
        if _is_noisy_file(neigh_label):
            return True

    # Filter noisy PROC neighbors
    if neigh_type == "subject":
        if _is_noisy_process(neigh_label):
            return True

    # Also filter if the CENTER is a noisy entity — its edges to others
    # are equally uninformative (e.g., libc.so.6 → [every process])
    if center_node in indexid2msg:
        center_type, center_label = indexid2msg[center_node]
        if center_type == "file" and _is_noisy_file(center_label):
            return True
        if center_type == "subject" and _is_noisy_process(center_label):
            return True

    return False


def filter_edges(center_node, neighbors, indexid2msg):
    """Filter a list of (neighbor_node, edge_type, timestamp) triples.

    Args:
        center_node: Node ID of the center entity
        neighbors: List of (neighbor_node, edge_type, timestamp) tuples
        indexid2msg: Dict mapping node IDs to (node_type, label_str)

    Returns:
        Filtered list of (neighbor_node, edge_type, timestamp) tuples
    """
    return [
        (n, e, t) for n, e, t in neighbors
        if not is_noisy_edge(center_node, n, e, indexid2msg)
    ]


def filter_walk(context_edges, context_nodes, indexid2msg, entity_node=None):
    """Filter a walk/neighborhood by removing noisy (edge, node) pairs.

    For walks with multiple hops, removes individual (edge, node) steps
    that are noisy. If entity_node is provided and is itself noisy,
    returns empty lists (skip entire example).

    Args:
        context_edges: List of edge type strings
        context_nodes: List of neighbor node IDs
        indexid2msg: Dict mapping node IDs to (node_type, label_str)
        entity_node: Optional center entity node ID

    Returns:
        (filtered_edges, filtered_nodes) — may be empty if all filtered
    """
    # If the center entity itself is noisy, skip entirely
    if entity_node is not None and entity_node in indexid2msg:
        center_type, center_label = indexid2msg[entity_node]
        if center_type == "file" and _is_noisy_file(center_label):
            return [], []
        if center_type == "subject" and _is_noisy_process(center_label):
            return [], []

    filtered_edges = []
    filtered_nodes = []

    for i, nid in enumerate(context_nodes):
        edge = context_edges[i] if i < len(context_edges) else None
        if nid in indexid2msg:
            ntype, nlabel = indexid2msg[nid]
            if ntype == "file" and _is_noisy_file(nlabel):
                continue
            if ntype == "subject" and _is_noisy_process(nlabel):
                continue
        if edge is not None:
            filtered_edges.append(edge)
        filtered_nodes.append(nid)

    return filtered_edges, filtered_nodes
