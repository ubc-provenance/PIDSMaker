"""Cleaning of the raw provenance data during graph construction.

Configured in `construction.cleaning`; `used_methods` is a comma-separated list of:

- `remove_null_attributes`: null-like attribute values (`None`, `NA`, `null`, `<UNNAMED fd:3>`,
  IPs that do not parse, ports outside 1-65535, ...) are removed from node labels instead of
  becoming literal tokens such as "None".
- `normalize_labels`: whitespace, duplicate slashes, Windows path prefixes and quotes are
  normalized, and long labels are truncated (see the options in `normalize_labels`).
- `remove_duplicate_events`: events with the same source, destination, type and timestamp are
  kept once. Event UUIDs are not used, as some datasets (e.g. CADETS) reuse them across events.
- `remove_self_loops`: events whose source is their destination are removed.
- `remove_noisy_netflows`: netflow nodes whose remote IP belongs to `ip_categories` (e.g.
  multicast name resolution or broadcast) are removed, along with their events.
- `dedup_window_edges`: only the first event of each (source, destination, type) is kept in
  each time window graph.

The default `none` builds the same graphs as without cleaning. Methods that remove nodes may
remove attack nodes: `cleaning_report.json` (in the construction folder) reports how many
nodes of the ground truth are lost, for information only.
"""

import ipaddress
import json
import os
import re
from collections import Counter
from operator import itemgetter

from pidsmaker.config.config import CLEANING_METHODS, IP_CATEGORIES  # noqa: F401
from pidsmaker.config.pipeline import get_cleaning_methods_from_cfg
from pidsmaker.utils.utils import log

# Values that stand for a missing attribute in the databases of the supported datasets
NULL_VALUES = {"", "none", "null", "nan", "na", "n/a", "unknown", "<unknown>", "-"}
_UNNAMED = re.compile(r"^<unnamed.*>$", re.IGNORECASE)  # ClearScope sockets, e.g. <UNNAMED fd:32>
IP_ATTRIBUTES = {"local_ip", "remote_ip"}
PORT_ATTRIBUTES = {"local_port", "remote_port"}

_WHITESPACES = re.compile(r"\s+")
_SLASHES = re.compile(r"(?<!:)/{2,}")  # keeps the "//" of URLs (http://...)
_BACKSLASHES = re.compile(r"\\{2,}")
# Windows NT object prefixes, e.g. \\?\C:\..., //?/C:/..., \??\C:\...
_PATH_PREFIXES = re.compile(r"(?<![^\s\"'])(?:\\\\\?\\|//\?/|\\\?\?\\)")
_QUOTES = "\"'"


def is_null_value(value) -> bool:
    if value is None:
        return True
    value = str(value).strip()
    return value.lower() in NULL_VALUES or bool(_UNNAMED.match(value))


def parse_ip(value):
    """Returns the IP address of `value`, or None if it is not a valid IP."""
    try:
        return ipaddress.ip_address(str(value).strip().split("%")[0])  # drops the zone of IPv6
    except ValueError:
        return None


def is_valid_port(value) -> bool:
    try:
        return 1 <= int(str(value).strip()) <= 65535
    except ValueError:
        return False


def ip_category(ip) -> str:
    """Category of an IP address among `IP_CATEGORIES`, or "unicast"."""
    if ip.is_unspecified:
        return "unspecified"
    if ip.is_loopback:
        return "loopback"
    if ip.is_multicast:
        return "multicast"
    if ip.version == 4 and ip == ipaddress.IPv4Address("255.255.255.255"):
        return "broadcast"
    if ip.is_link_local:
        return "link_local"
    return "unicast"


class Cleaner:
    """Applies the cleaning methods of `construction.cleaning` and counts their effects."""

    def __init__(self, cfg):
        self.methods = set(get_cleaning_methods_from_cfg(cfg))
        cleaning = cfg.construction.cleaning

        if "normalize_labels" in self.methods:
            args = cleaning.normalize_labels
            self.collapse_whitespace = args.collapse_whitespace
            self.collapse_slashes = args.collapse_slashes
            self.strip_path_prefixes = args.strip_path_prefixes
            self.strip_quotes = args.strip_quotes
            self.lowercase = args.lowercase
            self.max_length = args.max_length

        if "remove_noisy_netflows" in self.methods:
            categories = str(cleaning.remove_noisy_netflows.ip_categories)
            self.noisy_ip_categories = {c.strip() for c in categories.split(",") if c.strip()}

        self.stats = Counter()
        self.removed_nodes = set()
        self._seen_nodes = set()  # nodes of the events before cleaning
        self._graph_nodes = set()  # nodes of the built graphs
        self._last_timestamp = None
        self._events_at_timestamp = set()

    def __bool__(self):
        return len(self.methods) > 0

    # ---------------------------------------------------------------- nodes
    def clean_attributes(self, node_type: str, attrs: dict) -> dict:
        """Replaces the null-like attribute values by empty strings."""
        if "remove_null_attributes" not in self.methods:
            return attrs
        cleaned = {}
        for key, value in attrs.items():
            if key == "type":
                cleaned[key] = value
                continue
            is_null = is_null_value(value)
            if not is_null and key in IP_ATTRIBUTES:
                is_null = parse_ip(value) is None
            elif not is_null and key in PORT_ATTRIBUTES:
                is_null = not is_valid_port(value)
            if is_null:
                self.stats[f"null_attributes.{node_type}.{key}"] += 1
            cleaned[key] = "" if is_null else str(value).strip()
        return cleaned

    def is_noisy_netflow(self, attrs: dict) -> bool:
        if "remove_noisy_netflows" not in self.methods:
            return False
        ip = parse_ip(attrs.get("remote_ip", ""))
        if ip is not None and ip_category(ip) in self.noisy_ip_categories:
            self.stats[f"noisy_netflows.{ip_category(ip)}"] += 1
            return True
        return False

    def normalize_label(self, label: str) -> str:
        if "normalize_labels" not in self.methods:
            return label
        original = label
        if self.strip_path_prefixes:
            label = _PATH_PREFIXES.sub("", label)
        if self.collapse_slashes:
            label = _SLASHES.sub("/", label)
            label = _BACKSLASHES.sub(r"\\", label)
        if self.strip_quotes:
            # e.g. "C:\Program Files\app.exe" -arg and C:\Program Files\app.exe -arg are the same
            label = "".join(c for c in label if c not in _QUOTES)
        if self.collapse_whitespace:
            label = _WHITESPACES.sub(" ", label).strip()
        if self.lowercase:
            label = label.lower()
        if self.max_length > 0 and len(label) > self.max_length:
            label = label[: self.max_length].rstrip()
            self.stats["normalized_labels.truncated"] += 1
        if label != original:
            self.stats["normalized_labels.changed"] += 1
        return label

    # --------------------------------------------------------------- events
    def filter_events(self, rows):
        """Rows (with src, operation, dst and timestamp at indices 1, 2, 4 and 6) kept by
        `keep_event`, in order. Same result and statistics as calling it on each row."""
        if "remove_duplicate_events" in self.methods:  # sequential state: row by row
            keep_event = self.keep_event
            return [row for row in rows if keep_event(row[1], row[2], row[4], row[6])]
        if not rows:
            return []
        self.stats["events.input"] += len(rows)
        self._seen_nodes.update(map(itemgetter(1), rows))
        self._seen_nodes.update(map(itemgetter(4), rows))
        if "remove_self_loops" in self.methods:
            kept = [row for row in rows if row[1] != row[4]]
            removed = len(rows) - len(kept)
            if removed:
                self.stats["events.removed.self_loops"] += removed
            return kept
        return list(rows)

    def keep_event(self, src_index_id, operation, dst_index_id, timestamp) -> bool:
        """Whether an event is kept. Events must be given in timestamp order."""
        self.stats["events.input"] += 1
        self._seen_nodes.add(src_index_id)
        self._seen_nodes.add(dst_index_id)

        if "remove_self_loops" in self.methods and src_index_id == dst_index_id:
            self.stats["events.removed.self_loops"] += 1
            return False

        if "remove_duplicate_events" in self.methods:
            # Duplicates share their timestamp, so only the events of the current one are kept
            if timestamp != self._last_timestamp:
                self._last_timestamp = timestamp
                self._events_at_timestamp.clear()
            key = (src_index_id, operation, dst_index_id)
            if key in self._events_at_timestamp:
                self.stats["events.removed.duplicates"] += 1
                return False
            self._events_at_timestamp.add(key)

        return True

    def dedup_window(self, events: list, key_fn) -> list:
        """Keeps the first event of each `key_fn(event)` in a time window."""
        if "dedup_window_edges" not in self.methods:
            return events
        seen, kept = set(), []
        for event in events:
            key = key_fn(event)
            if key not in seen:
                seen.add(key)
                kept.append(event)
        self.stats["events.removed.window_duplicates"] += len(events) - len(kept)
        return kept

    def add_graph_nodes(self, nodes):
        self._graph_nodes.update(nodes)

    # --------------------------------------------------------------- report
    def save_report(self, cfg):
        """Logs the effect of each method and saves it to `cleaning_report.json`."""
        lost_nodes = self._seen_nodes - self._graph_nodes
        report = {
            "used_methods": sorted(self.methods),
            "stats": dict(sorted(self.stats.items())),
            "nodes_in_events": len(self._seen_nodes),
            "nodes_lost": len(lost_nodes),
        }
        # The ground truth is only read to report the attack nodes that cleaning removes
        try:
            from pidsmaker.utils.labelling import get_ground_truth

            gt_nodes, _, _ = get_ground_truth(cfg)
            gt_nodes = {str(n) for n in gt_nodes} & self._seen_nodes
            report["ground_truth_nodes_in_events"] = len(gt_nodes)
            report["ground_truth_nodes_lost"] = len(gt_nodes & lost_nodes)
        except Exception as e:  # e.g. datasets without ground truth
            report["ground_truth_nodes_lost"] = f"unavailable ({type(e).__name__}: {e})"

        log("Cleaning report:")
        for key, value in report.items():
            log(f"  {key}: {value}")
        gt_lost = report.get("ground_truth_nodes_lost")
        if isinstance(gt_lost, int) and gt_lost > 0:
            log(
                f"WARNING: cleaning removed {gt_lost} attack nodes of the ground truth, "
                "which are not evaluated."
            )
        with open(os.path.join(cfg.construction._task_path, "cleaning_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        return report
