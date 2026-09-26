"""
Functional class labeling for provenance graph entities.

Maps every entity (process, file, socket) to a coarse functional class that
captures *what it is* independent of the specific binary/path/port.  Two
nginx workers, Apache, and Caddy all get class "webserver".  This is used
by behavior_signatures.py to build behavior signature labels of the form
(direction, event_type, neighbor_entity_type, neighbor_class).

Design:
  - PROC classes: reuse canonical_tokens.PROC_CATEGORIES but strip the
    bracket/prefix notation → plain lowercase strings.
  - FILE classes: reuse canonical_tokens.FILE_DIR_CATEGORIES similarly,
    plus extension-based fallback.
  - SOCK classes: well-known port → named class, else range bucket.

All functions accept a raw label string (as stored in indexid2msg) and
return a plain string class name, or "unknown" if nothing matches.
"""

import functools

import os
import re
from typing import Optional

from .canonical_tokens import (
    PROC_CATEGORIES,
    FILE_DIR_CATEGORIES,
    _FILE_DIR_SORTED,
    match_proc_category,
)
from .tokenizer_bpe import (
    WELL_KNOWN_PORTS,
    RE_IP,
    RE_IPV6,
)


# ═══════════════════════════════════════════════════════════════════════════
# PROC CLASSES
# ═══════════════════════════════════════════════════════════════════════════
# Convert [CAT_WEBSERVER] → "webserver", [CAT_DB_CLIENT] → "db_client", etc.

# Merge classes that have the same behavioral profile.
# Applied after stripping the CAT_/FCAT_ prefix.
_CLASS_MERGE = {
    # ── Port merges (reduce singleton socket classes) ─────────────────
    "port_imap": "port_mail",       # IMAP/SMTP/POP3 all serve mail
    "port_imaps": "port_mail",
    "port_smtp": "port_mail",
    "port_pop3": "port_mail",
    "port_ldaps": "port_ldap",      # LDAP + LDAPS
    "port_ftp_data": "port_ftp",    # FTP control + data
    "port_rdp": "port_ssh",         # Remote access protocols
    "port_telnet": "port_ssh",
    "port_dhcp_c": "port_dhcp",     # DHCP client + server
    "port_mssql": "port_database",  # Database protocols
    "port_oracle": "port_database",
    "port_syslog": "port_low",      # Fold rare low-port services into port_low
    "port_ntp": "port_low",
    "unix_socket": "port_low",
    "port_rpc": "port_low",         # Windows RPC — confused with netbios, fold into port_low
    "port_http_alt": "port_http",   # 8080 → http
    "port_https_alt": "port_https", # 8443 → https

    # ── Fallback merges (from path/extension classifiers, not canonical_tokens) ──
    "binary": "executable",     # from _PROC_BIN_DIRS path fallback
    "desktop": "executable",    # from _PROC_BASENAME_CLASSES
    "script": "executable",     # from _FILE_EXT_CLASSES (.sh, .bat, .ps1)
    "socket_file": "tmp",       # from _FILE_EXT_CLASSES (.sock)
    "package": "archive",       # from _FILE_EXT_CLASSES (.deb, .rpm)
}


def _cat_token_to_class(token: str) -> str:
    """Convert a canonical category token to a plain class name."""
    # [CAT_WEBSERVER] → webserver, [FCAT_LOG_WEB] → log_web
    s = token.strip("[]")
    if s.startswith("CAT_"):
        cls = s[4:].lower()
    elif s.startswith("FCAT_"):
        cls = s[5:].lower()
    else:
        cls = s.lower()
    return _CLASS_MERGE.get(cls, cls)


# Pre-compute the set of unique PROC class names
_PROC_CLASS_SET = {_cat_token_to_class(v) for v in PROC_CATEGORIES.values()}

# Pre-compute the set of unique FILE class names
_FILE_CLASS_SET = {_cat_token_to_class(v) for v in FILE_DIR_CATEGORIES.values()}


# Known binary directories — if a process path starts with one of these
# and the basename isn't in PROC_CATEGORIES, classify as "binary".
_PROC_BIN_DIRS = (
    "/usr/bin/", "/usr/sbin/", "/usr/local/bin/", "/usr/local/sbin/",
    "/bin/", "/sbin/",
    "/system/bin/",          # Android
    "/system/xbin/",         # Android
    "/vendor/bin/",          # Android
)

# Bare command names that map to a SPECIFIC class (not just "binary").
# Commands not listed here will fall through to the generic "binary" catch-all.
# Only add entries here when the class is semantically meaningful and different
# from "binary" — don't enumerate every Unix command.
_PROC_BASENAME_CLASSES = {
    # Browsers (not in PROC_CATEGORIES)
    "firefox": "browser", "chromium-browse": "browser",
    "chromium-browser": "browser", "chrome": "browser",
    "chrome_crashpad": "browser", "geckodriver": "browser",
    # Mail subsystems (postfix internals not in PROC_CATEGORIES)
    "trivial-rewrite": "mail", "anvil": "mail",
    # Terminal multiplexers (behaviorally like shells)
    "screen": "shell", "tmux": "shell",
    # Desktop apps
    "telepathy-indicator": "desktop", "unity-2d-panel": "desktop",
}


def _extract_proc_basename(label_str: str) -> str:
    """Extract the binary basename from a process label.

    Handles paths (/usr/bin/foo → foo), .exe suffix (cmd.exe → cmd),
    quoted paths with spaces ("C:/Program Files/foo.exe" → foo),
    and the "subject" type prefix.
    """
    s = label_str
    # Strip "subject" prefix
    if s.startswith("subject "):
        s = s[8:]

    # Handle quoted paths (e.g. "//?/C:/Program Files (x86)/Firefox/firefox.exe" -args)
    s = s.strip()
    if s.startswith('"'):
        end_quote = s.find('"', 1)
        if end_quote > 0:
            first = s[1:end_quote]
        else:
            first = s[1:]
    else:
        first = s.split()[0] if s else ""

    first = first.replace("\\", "/")
    # Strip Windows device path prefix //?/
    if first.startswith("//?/"):
        first = first[4:]
    if "/" in first:
        first = first.rsplit("/", 1)[-1]
    if first.lower().endswith(".exe"):
        first = first[:-4]
    return first


def classify_proc(label_str: str) -> str:
    """Classify a process entity by its command line / binary name.

    Priority:
    1. canonical_tokens.PROC_CATEGORIES (functional roles like webserver, shell)
    2. _PROC_BASENAME_CLASSES (coreutils, sysadmin tools, browsers, etc.)
    3. Path-based fallback (known bin directories → "binary")
    4. Pattern-based fallback (Windows .exe, Android packages, apt-*/dpkg-*)

    Returns a plain class name or "unknown" if no match.
    """
    cat_token = match_proc_category(label_str)
    if cat_token is not None:
        return _cat_token_to_class(cat_token)

    # Basename lookup for common commands not in canonical_tokens
    basename = _extract_proc_basename(label_str)
    if basename in _PROC_BASENAME_CLASSES:
        return _PROC_BASENAME_CLASSES[basename]

    # Prefix-based matching
    if basename.startswith(("apt-", "dpkg-")):
        return "pkgmgr"

    # Truncated daemon names: "sshd:" → ssh, "systemd-journal" → init
    # Strip trailing ":" and check known prefixes against PROC_CATEGORIES
    basename_clean = basename.rstrip(":")
    if basename_clean in PROC_CATEGORIES:
        return _cat_token_to_class(PROC_CATEGORIES[basename_clean])
    # systemd-* variants (systemd-journald, systemd-networkd, systemd-resolved, etc.)
    if basename.startswith("systemd-") or basename.startswith("systemd:"):
        return "init"
    # Android system:* processes
    if basename.startswith("system:"):
        return "android_sys"

    # Versioned interpreters: python2.7, python3.6, perl5.30, ruby2.7, etc.
    # PROC_CATEGORIES has "python", "python3", "perl", "ruby" but not versioned variants.
    import re as _re
    if _re.match(r'^(python|perl|ruby|php|lua|node)\d', basename):
        base = _re.match(r'^(python|perl|ruby|php|lua|node)', basename).group(1)
        if base in PROC_CATEGORIES:
            return _cat_token_to_class(PROC_CATEGORIES[base])

    # Path-based fallback: processes from known binary directories
    parts = label_str.split()
    if parts and parts[0] == "subject":
        parts = parts[1:]
    if parts:
        first = parts[0].strip('"').replace("\\", "/")
        for bin_dir in _PROC_BIN_DIRS:
            if first.startswith(bin_dir):
                return "binary"

    # Windows .exe processes
    if ".exe" in label_str.lower():
        return "win_system"

    # Android package-style processes (com.android.*, android.*)
    if basename.startswith(("android.", "com.android.", "com.google.")):
        return "android_sys"

    # Firefox/chromium snap processes
    if "firefox" in basename.lower() or "chromium" in basename.lower():
        return "browser"

    # Any bare alphabetic command name is a binary that was executed.
    # Excludes flag-style args (-c, +%s), numbers, and garbage labels.
    if basename and basename[0].isalpha() and all(c.isalnum() or c in "-_." for c in basename):
        return "binary"

    return "unknown"


# ═══════════════════════════════════════════════════════════════════════════
# FILE CLASSES
# ═══════════════════════════════════════════════════════════════════════════

# Extensions that identify file content type, used as fallback when no
# path-prefix rule matches.  Path context wins for most files (a .tmp in
# browser data is browser_data, not tmp), but these catch files in
# unrecognized directories.
_FILE_EXT_CLASSES = {
    # Source code
    ".c": "source", ".h": "source", ".cpp": "source", ".hpp": "source",
    ".cc": "source", ".py": "source", ".js": "source", ".ts": "source",
    ".rs": "source", ".go": "source", ".java": "source", ".rb": "source",
    ".pl": "source", ".m": "source", ".swift": "source",
    ".sh": "script", ".bash": "script", ".bat": "script", ".ps1": "script",
    ".cmd": "script", ".vbs": "script", ".wsf": "script",
    # Web
    ".html": "source", ".htm": "source", ".css": "source",
    ".xhtml": "source", ".xul": "source", ".jsx": "source", ".tsx": "source",
    ".php": "source", ".asp": "source", ".aspx": "source", ".jsp": "source",
    # Compiled / bytecode
    ".o": "object", ".obj": "object", ".a": "library", ".so": "library",
    ".dylib": "library", ".dll": "library", ".class": "bytecode",
    ".pyc": "bytecode", ".pyo": "bytecode",
    # Binaries / executables
    ".exe": "binary", ".msi": "binary", ".apk": "binary",
    ".bin": "binary", ".elf": "binary",
    # Config
    ".conf": "config", ".cfg": "config", ".ini": "config",
    ".yaml": "config", ".yml": "config", ".toml": "config",
    ".properties": "config", ".reg": "config",
    # Data
    ".json": "data", ".xml": "data", ".csv": "data", ".tsv": "data",
    ".sql": "data", ".dat": "data", ".plist": "data",
    ".db": "database_file", ".sqlite": "database_file",
    # Logs
    ".log": "log", ".evtx": "log", ".etl": "log",
    # Archives
    ".tar": "archive", ".gz": "archive", ".zip": "archive",
    ".bz2": "archive", ".xz": "archive", ".7z": "archive",
    ".rar": "archive", ".cab": "archive",
    ".deb": "package", ".rpm": "package",
    # Security
    ".pem": "certificate", ".key": "key_file", ".crt": "certificate",
    ".cert": "certificate", ".pub": "pubkey",
    # Media
    ".png": "media", ".jpg": "media", ".jpeg": "media",
    ".gif": "media", ".svg": "media", ".ico": "media", ".bmp": "media",
    ".mp4": "media", ".mp3": "media", ".wav": "media", ".webm": "media",
    ".ttf": "media", ".otf": "media", ".woff": "media", ".woff2": "media",
    # Documents
    ".pdf": "document", ".doc": "document", ".docx": "document",
    ".xls": "document", ".xlsx": "document",
    ".ppt": "document", ".pptx": "document", ".rtf": "document",
    ".odt": "document", ".ods": "document", ".odp": "document",
    ".txt": "document", ".md": "document", ".rst": "document",
    # Windows-specific
    ".mui": "library", ".sys": "library", ".drv": "library",
    ".ocx": "library", ".cpl": "library",
    ".lnk": "data", ".pf": "data", ".manifest": "config",
    # Runtime
    ".sock": "socket_file", ".pid": "runtime", ".lock": "runtime",
    ".tmp": "tmp",
}


def classify_file(label_str: str) -> str:
    """Classify a file entity by its path.

    Extension first — a .docx is a document whether it's in /home/, /tmp/,
    or /var/www/. Path only matters for extensionless files and dotfiles.

    Returns a plain class name or "unknown".
    """
    parts = label_str.split()
    if parts and parts[0] == "file":
        parts = parts[1:]
    if not parts:
        return "unknown"

    path = parts[0].replace("\\", "/")

    # Extension always wins (defines what the file IS)
    _, ext = os.path.splitext(path)
    ext_lower = ext.lower()
    if ext_lower in _FILE_EXT_CLASSES:
        return _FILE_EXT_CLASSES[ext_lower]

    # Path-prefix fallback for extensionless files and dotfiles.
    # Find ALL matching patterns, pick the most specific:
    #   1. Longest pattern wins (more specific path context)
    #   2. At equal length, the one appearing DEEPER in the path wins
    #      e.g. for "/home/user/.git/refs", "/home/" matches at pos 0
    #      but "/.git/" matches at pos 37 — the deeper match is more specific.
    best_match = None
    best_len = -1
    best_pos = -1
    for pattern, category in _FILE_DIR_SORTED:
        if pattern in path:
            plen = len(pattern)
            pos = path.rfind(pattern)
            if plen > best_len or (plen == best_len and pos > best_pos):
                best_match = category
                best_len = plen
                best_pos = pos
    if best_match is not None:
        return _cat_token_to_class(best_match)

    return "unknown"


# ═══════════════════════════════════════════════════════════════════════════
# SOCK CLASSES
# ═══════════════════════════════════════════════════════════════════════════

# Map well-known port tokens to plain class names
# [PORT_SSH] → "port_ssh", [PORT_HTTP] → "port_http", etc.
_SOCK_PORT_CLASSES = {
    port: token.strip("[]").lower()
    for port, token in WELL_KNOWN_PORTS.items()
}

# Additional well-known ports not in tokenizer
_SOCK_PORT_CLASSES.update({
    123: "port_ntp",
    135: "port_rpc",
    137: "port_netbios", 138: "port_netbios", 139: "port_netbios",
    389: "port_ldap", 636: "port_ldaps",
    514: "port_syslog",
    1433: "port_mssql", 1521: "port_oracle",
    3389: "port_rdp",
    5985: "port_winrm", 5986: "port_winrm",
    6379: "port_redis",
    9200: "port_elasticsearch",
    27017: "port_mongodb",
})


def classify_sock(label_str: str) -> str:
    """Classify a socket entity by its IP/port tuple.

    Parses the label for port numbers and classifies by well-known port.
    Falls back to range buckets: port_low (1-1023), port_reg (1024-49151),
    port_eph (49152-65535).

    Returns a plain class name or "unknown".
    """
    parts = label_str.split()
    if parts and parts[0] == "netflow":
        parts = parts[1:]

    # Collect all port-like integers
    ports = []
    for token in parts:
        # Handle IP:port format
        if ":" in token and not token.startswith("0000:"):
            for sub in token.split(":"):
                try:
                    p = int(sub)
                    if 1 <= p <= 65535:
                        ports.append(p)
                except ValueError:
                    pass
        elif "->" in token:
            for endpoint in token.split("->"):
                for sub in endpoint.rsplit(":", 1):
                    try:
                        p = int(sub)
                        if 1 <= p <= 65535:
                            ports.append(p)
                    except ValueError:
                        pass
        else:
            try:
                p = int(token)
                if 1 <= p <= 65535:
                    ports.append(p)
            except ValueError:
                pass

    # Classify by the well-known port (the service side).
    # In a socket like "10.0.0.1 58425 192.168.1.1 80", port 80 is the
    # service — port 58425 is ephemeral.  If multiple well-known ports
    # exist (shouldn't happen), pick the lowest (most specific service).
    known = sorted([p for p in ports if p in _SOCK_PORT_CLASSES])
    if known:
        return _SOCK_PORT_CLASSES[known[0]]

    # Range buckets based on the lowest port (most likely the service side)
    if ports:
        min_port = min(ports)
        if min_port <= 1023:
            return "port_low"
        elif min_port <= 49151:
            return "port_reg"
        else:
            return "port_eph"

    # Unix domain socket or unrecognizable
    if any("/" in t for t in parts):
        return "unix_socket"

    return "unknown"


# ═══════════════════════════════════════════════════════════════════════════
# UNIFIED INTERFACE
# ═══════════════════════════════════════════════════════════════════════════

@functools.lru_cache(maxsize=None)
def classify_entity(node_type: str, label_str: str) -> str:
    """Classify any entity by its type and label.

    Args:
        node_type: "subject", "file", or "netflow"
        label_str: the raw label string from indexid2msg

    Returns:
        A plain string class name (e.g. "webserver", "log_system", "port_ssh").
    """
    if node_type == "subject":
        cls = classify_proc(label_str)
    elif node_type == "file":
        cls = classify_file(label_str)
    elif node_type == "netflow":
        cls = classify_sock(label_str)
    else:
        cls = "unknown"
    return _CLASS_MERGE.get(cls, cls)


def get_all_entity_classes():
    """Return the complete set of known entity class names.

    Used to build the behavior signature label vocabulary.
    """
    classes = set()
    classes.update(_PROC_CLASS_SET)
    classes.update(_FILE_CLASS_SET)
    classes.update({_cat_token_to_class(f"[{v.strip('[]')}]") for v in _SOCK_PORT_CLASSES.values()})
    # Extension-based file classes
    classes.update(set(_FILE_EXT_CLASSES.values()))
    # Range buckets + special
    classes.update({"port_low", "port_reg", "port_eph", "unix_socket", "unknown"})
    return sorted(classes)
