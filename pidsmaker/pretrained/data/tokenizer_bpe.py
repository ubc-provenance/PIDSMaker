"""
Improved provenance graph tokenizer: domain-specific normalization + BPE.

Key improvements over a plain whitespace tokenizer:
1. '/' is embedded in path component tokens: /usr/lib -> [ROOT][/USR][/LIB].
   For unknown dirs the '/' is prepended to the first BPE word (/traceback ...),
   so BPE fragments of one component never look like a new component.
2. Intra-component separators emitted as distinct tokens: x86_64-linux -> x86[SEP_UNDERSCORE]64[SEP_DASH]linux
3. BPE subword encoding - handles rare words by decomposing into known subwords
4. Entropy-based random string detection - hashes, UUIDs, random names -> canonical tokens
5. Semantic port categories - not one token per port number
6. No [DOT] tokens in IPs - octets alone carry the information
7. Flags preserve their content: -sh -> [FLAG] sh, --no-verify -> [FLAG] no[SEP_DASH]verify
8. Versioned shared objects: libfoo.so.1.2 -> libfoo [EXT_so] 1 2
"""

import os
import re
from collections import Counter
from typing import Dict, List, Optional, Set, Tuple

import torch

from pidsmaker.utils.dataset_utils import get_rel2id
from pidsmaker.utils.utils import log
from .canonical_tokens import (
    ALL_CATEGORY_TOKENS,
    match_file_category,
    match_proc_category,
)


# ── Label sanitization ───────────────────────────────────────────────────────
# Keep only printable ASCII (space through tilde). Provenance labels are file
# paths, command lines, and IP addresses — all ASCII. Everything else is noise:
# control chars, Latin-1 supplement, obscure Unicode punctuation, zero-width
# chars, CJK descriptors, BOM, replacement chars, etc.
_BINARY_GARBAGE_RE = re.compile(r"[^\x20-\x7e]+")

# ── Special tokens ───────────────────────────────────────────────────────────

PAD = "[PAD]"
MASK = "[MASK]"
CLS = "[CLS]"
SEP = "[SEP]"
UNK = "[UNK]"
FORWARD = "[FORWARD]"
BACKWARD = "[BACKWARD]"
FORWARD_NEIGH = "[FORWARD_NEIGH]"
BACKWARD_NEIGH = "[BACKWARD_NEIGH]"
SPECIAL_TOKENS = [PAD, MASK, CLS, SEP, UNK, FORWARD, BACKWARD, FORWARD_NEIGH, BACKWARD_NEIGH]

NODE_TYPE_TOKENS = {
    "subject": "[PROC]",
    "file": "[FILE]",
    "netflow": "[SOCK]",
}

# ── Provenance graph schema (DARPA TC) ───────────────────────────────────────
# Empirically derived from 2.7M walks across CADETS E3/E5, THEIA E3/E5, TRACE E3.

ENTITY_TYPES = {"[PROC]", "[FILE]", "[SOCK]"}

EVENT_TYPES = {
    "EVENT_CLONE",
    "EVENT_CONNECT",
    "EVENT_EXECUTE",
    "EVENT_OPEN",
    "EVENT_READ",
    "EVENT_RECVFROM",
    "EVENT_RECVMSG",
    "EVENT_SENDMSG",
    "EVENT_SENDTO",
    "EVENT_WRITE",
    # Windows registry edge types (consistent vocabulary)
    "REG_READ",
    "REG_WRITE",
    "REG_CREATEKEY",
    "REG_DELETEKEY",
    "REG_DELETEVALUE",
}

# (src_entity_token, event_name, dst_entity_token)
VALID_TRIPLETS = {
    ("[FILE]", "EVENT_EXECUTE", "[PROC]"),
    ("[FILE]", "EVENT_OPEN",    "[PROC]"),
    ("[FILE]", "EVENT_READ",    "[PROC]"),
    ("[FILE]", "EVENT_RECVFROM","[PROC]"),
    ("[FILE]", "EVENT_RECVMSG", "[PROC]"),
    ("[PROC]", "EVENT_CLONE",   "[PROC]"),
    ("[PROC]", "EVENT_CONNECT", "[FILE]"),
    ("[PROC]", "EVENT_CONNECT", "[SOCK]"),
    ("[PROC]", "EVENT_EXECUTE", "[PROC]"),
    ("[PROC]", "EVENT_SENDMSG", "[FILE]"),
    ("[PROC]", "EVENT_SENDMSG", "[SOCK]"),
    ("[PROC]", "EVENT_SENDTO",  "[FILE]"),
    ("[PROC]", "EVENT_SENDTO",  "[SOCK]"),
    ("[PROC]", "EVENT_WRITE",   "[FILE]"),
    ("[PROC]", "EVENT_WRITE",   "[SOCK]"),
    ("[SOCK]", "EVENT_READ",    "[PROC]"),
    ("[SOCK]", "EVENT_RECVFROM","[PROC]"),
    ("[SOCK]", "EVENT_RECVMSG", "[PROC]"),
    # Windows registry triplets (registry paths are [FILE] nodes)
    ("[FILE]", "REG_READ",       "[PROC]"),
    ("[PROC]", "REG_WRITE",      "[FILE]"),
    ("[PROC]", "REG_CREATEKEY",  "[FILE]"),
    ("[PROC]", "REG_DELETEKEY",  "[FILE]"),
    ("[PROC]", "REG_DELETEVALUE","[FILE]"),
}

# ── Domain tokens ────────────────────────────────────────────────────────────

HASH = "[HASH]"
GUID = "[GUID]"
RANDOM = "[RANDOM]"
TIMESTAMP = "[TIMESTAMP]"
VER = "[VER]"
NUM = "[NUM]"
ENV_VAR = "[ENV]"
FLAG = "[FLAG]"
ROOT = "[ROOT]"
WIN_ROOT = "[WIN_ROOT]"
NO_CMD = "[NO_CMD]"
NO_PATH = "[NO_PATH]"
NO_IP = "[NO_IP]"

# Intra-component separator tokens
SEP_DASH = "[SEP_DASH]"            # '-' separator within a name/component
SEP_UNDERSCORE = "[SEP_UNDERSCORE]"  # '_' separator within a name/component
SEP_DOT = "[SEP_DOT]"              # '.' separator within a name (non-extension context)
SEP_PLUS = "[SEP_PLUS]"            # '+' separator within a name (e.g. lost+found)
SEP_AMP = "[SEP_AMP]"              # '&' separator between URL query parameters

# Port categories: well-known ports get named tokens, others get range tokens
WELL_KNOWN_PORTS = {
    20: "[PORT_FTP_DATA]", 21: "[PORT_FTP]", 22: "[PORT_SSH]", 23: "[PORT_TELNET]",
    25: "[PORT_SMTP]", 53: "[PORT_DNS]", 67: "[PORT_DHCP]", 68: "[PORT_DHCP_C]",
    80: "[PORT_HTTP]", 110: "[PORT_POP3]", 123: "[PORT_NTP]", 143: "[PORT_IMAP]",
    389: "[PORT_LDAP]", 443: "[PORT_HTTPS]", 445: "[PORT_SMB]", 993: "[PORT_IMAPS]",
    3306: "[PORT_MYSQL]", 3389: "[PORT_RDP]", 5432: "[PORT_POSTGRES]",
    8080: "[PORT_HTTP_ALT]", 8443: "[PORT_HTTPS_ALT]",
}
PORT_LOW = "[PORT_LOW]"
PORT_REG = "[PORT_REG]"
PORT_EPH = "[PORT_EPH]"
IPV6 = "[IPV6]"

# IP address category tokens (used when normalize_netflow_ips is enabled)
PRIVATE_IP = "[PRIVATE_IP]"
PUBLIC_IP = "[PUBLIC_IP]"
LOCALHOST_IP = "[LOCALHOST_IP]"

# Well-known Unix directories
UNIX_DIRS = {
    "etc": "[ETC]", "bin": "[BIN]", "sbin": "[SBIN]", "usr": "[USR]",
    "var": "[VAR]", "tmp": "[TMP]", "dev": "[DEV]", "proc": "[PROC_DIR]",
    "sys": "[SYS]", "lib": "[LIB]", "lib64": "[LIB]", "home": "[HOME]",
    "root": "[ROOT_DIR]", "boot": "[BOOT]", "opt": "[OPT]", "mnt": "[MNT]",
    "media": "[MEDIA]", "run": "[RUN]", "srv": "[SRV]",
}

# Well-known Windows directories
WIN_DIRS = {
    "windows": "[WINDOWS]", "system32": "[SYS32]", "syswow64": "[SYSWOW64]",
    "programdata": "[PROGDATA]", "users": "[USERS]", "appdata": "[APPDATA]",
    "device": "[DEVICE]", "harddiskvolume1": "[DISK_VOL]",
    "harddiskvolume2": "[DISK_VOL]", "harddiskvolume3": "[DISK_VOL]",
    "program files": "[PROGFILES]", "program files (x86)": "[PROGFILES_X86]",
    "winsxs": "[WINSXS]", "temp": "[TMP]", "microsoft": "[MSFT]",
    "assembly": "[ASSEMBLY]", "microsoft.net": "[DOTNET]",
    "local": "[LOCAL]", "roaming": "[ROAMING]",
}

# Windows registry hive tokens (registry paths treated as file nodes)
REG_HKLM = "[REG_HKLM]"
REG_HKCU = "[REG_HKCU]"
REG_HKCR = "[REG_HKCR]"
REG_SOFTWARE = "[REG_SOFTWARE]"
REG_SYSTEM = "[REG_SYSTEM]"
REG_SERVICES = "[REG_SERVICES]"

# Registry hive prefix mapping
REG_HIVE_MAP = {
    "registry\\machine": REG_HKLM,
    "hkey_local_machine": REG_HKLM,
    "registry\\user": REG_HKCU,
    "hkey_current_user": REG_HKCU,
    "hkey_classes_root": REG_HKCR,
}
REG_SUBKEY_MAP = {
    "software": REG_SOFTWARE,
    "system": REG_SYSTEM,
    "services": REG_SERVICES,
}

# Path-context variants: / is embedded in the token so BPE fragments of one
# component (e.g. /trac + eback) are unambiguous vs separate components.
# Used only for absolute paths; e.g. /usr/lib -> [/USR] [/LIB].
UNIX_DIRS_PATH = {k: f"[/{v[1:-1]}]" for k, v in UNIX_DIRS.items()}
WIN_DIRS_PATH  = {k: f"[/{v[1:-1]}]" for k, v in WIN_DIRS.items()}

# Common file extensions that get dedicated tokens
KNOWN_EXTENSIONS = {
    # Binaries / libraries
    "exe", "dll", "so", "dylib", "bin", "msi", "jar", "war", "apk", "obj", "la",
    # Source code
    "c", "h", "cc", "hh", "cpp", "hpp", "py", "pyc", "js", "ts", "sh", "bash",
    "bat", "cmd", "ps1", "rb", "pl", "cs", "mm", "asm", "qml", "ui",
    # Build / project
    "pro", "pri", "prl", "prf", "cmake", "gn", "gni", "gyp", "in",
    # Config / data
    "conf", "cfg", "ini", "yaml", "yml", "json", "xml", "toml", "dtd",
    "idl", "proto", "mojom", "qdoc", "qrc", "qm", "qmlc",
    # Documents / text
    "log", "txt", "csv", "md", "html", "htm", "css", "svg",
    # Database
    "db", "sqlite", "sql",
    # Archives
    "zip", "tar", "gz", "bz2", "xz", "7z", "rar",
    # Security
    "pem", "key", "crt", "cert", "pub", "sha1",
    # System / runtime
    "sys", "ko", "o", "pid", "lock", "sock", "tmp", "cache", "pc",
    # Media
    "png", "jpg", "gif", "ico", "wav", "webp", "ttf",
    # Misc
    "dat", "data", "patch", "test", "tests", "ref", "out", "inc", "def",
    "pset", "stat", "xtb", "scxml", "tmpl", "ucm", "ent", "frag", "vert",
    "orth", "td", "xq", "d", "n", "new",
}

# ── Regex patterns ───────────────────────────────────────────────────────────

RE_HASH = re.compile(r"^[0-9a-fA-F]{7,}$")
RE_GUID = re.compile(
    r"^\{?[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}"
    r"-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\}?$"
)
RE_GUID_INNER = re.compile(
    r"\{?[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}"
    r"-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\}?"
)
RE_SID = re.compile(r"^S-\d+-\d+(-\d+)+$")
RE_VERSION = re.compile(r"^v?\d+(\.\d+){1,3}(-\w+)?$")
RE_TIMESTAMP = re.compile(r"^\d{4}[-_]\d{2}[-_]\d{2}")
RE_IP = re.compile(r"^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$")
RE_IPV6 = re.compile(r"^[0-9a-fA-F:]*::[0-9a-fA-F:]*$|^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$")
RE_ENV = re.compile(r"^(\$\{?[A-Za-z_]+\}?|%[A-Z_]+%)$")
RE_FLAG = re.compile(r"^--?[a-zA-Z][a-zA-Z0-9-]*$")
RE_EXT = re.compile(
    r"^(.+?)\.({})$".format(
        "|".join(re.escape(e) for e in sorted(KNOWN_EXTENSIONS, key=len, reverse=True))
    ),
    re.IGNORECASE,
)
RE_SEPARATORS = re.compile(r"[-_.]")
RE_SEPARATORS_CAPTURE = re.compile(r"([-_.+])")
RE_SO_VERSIONED = re.compile(r"^(.+)\.so(\.\d+)+$", re.IGNORECASE)

# Maps raw separator characters to their dedicated tokens
SEP_TOKEN_MAP = {"-": SEP_DASH, "_": SEP_UNDERSCORE, ".": SEP_DOT, "+": SEP_PLUS}

RE_QUERY_STRING = re.compile(r"^[^=&\s]+=.*&[^=&\s]+=.*$|^[^=&\s]*&[^=&\s]+=.*$")


# ── Tokenizer ────────────────────────────────────────────────────────────────


class ProvenanceTokenizerBPE:
    """Two-phase provenance graph tokenizer: domain normalization + BPE.

    Phase 1 (domain-specific):
      - Paths split into components without slash tokens
      - Well-known directories -> semantic tokens
      - Hashes, GUIDs, random strings -> canonical tokens
      - Ports -> semantic categories
      - IPs -> octets without dot tokens

    Phase 2 (BPE):
      - Remaining words decomposed into learned subwords
      - Guarantees no OOV (falls back to character level)

    Drop-in compatible with ProvenanceTokenizer.
    """

    def __init__(self, cfg, max_seq_len: int = 512, bpe_vocab_size: int = 8000, mode: str = "domain_bpe",
                 normalize_netflow_ips: bool = False, canonicalize_neighbors: bool = False,
                 strip_entity_type: bool = False):
        self.cfg = cfg
        self.max_seq_len = max_seq_len
        self.bpe_vocab_size = bpe_vocab_size
        self.mode = mode  # "domain_bpe" or "bpe_only"
        self.normalize_netflow_ips = normalize_netflow_ips
        self.canonicalize_neighbors = canonicalize_neighbors
        self.strip_entity_type = strip_entity_type
        self.expand_netflow_ips = False  # When True, emit [CATEGORY] + raw IP tokens
        self.mask_rate = 0.15

        # Edge tokens from dataset config
        rel2id = get_rel2id(cfg)
        self.edge_tokens = {k: f"[EDGE_{k}]" for k in rel2id if isinstance(k, str)}

        # BPE state
        self.merges: List[Tuple[str, str]] = []

        # Vocabulary
        self.token2id: Dict[str, int] = {}
        self.id2token: Dict[int, str] = {}
        self.vocab_size: int = 0

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def pad_id(self) -> int:
        return self.token2id[PAD]

    @property
    def mask_id(self) -> int:
        return self.token2id[MASK]

    @property
    def cls_id(self) -> int:
        return self.token2id[CLS]

    @property
    def sep_id(self) -> int:
        return self.token2id[SEP]

    # ── Phase 1: Domain-specific pre-tokenization ─────────────────────────

    @staticmethod
    def _is_random_string(s: str) -> bool:
        """Detect random/meaningless strings via character-type transition rate.

        Counts transitions between character types (upper/lower/digit).
        Random strings like 'pEja72mA' have many transitions; real words
        like 'Downloads' or 'python3' have very few.

        For short strings (4-7 chars), requires all 3 char types (upper +
        lower + digit) which is extremely rare in real words but common in
        random suffixes like 'lqDkW1', 'XSVaeB', 'j7x4Qs'.
        """
        if len(s) < 4:
            return False

        has_upper = any(c.isupper() for c in s)
        has_lower = any(c.islower() for c in s)
        has_digit = any(c.isdigit() for c in s)
        n_types = has_upper + has_lower + has_digit

        if n_types < 2:
            return False

        # Count transitions between character types
        transitions = 0
        for i in range(len(s) - 1):
            t1 = "U" if s[i].isupper() else ("D" if s[i].isdigit() else "L")
            t2 = "U" if s[i + 1].isupper() else ("D" if s[i + 1].isdigit() else "L")
            if t1 != t2:
                transitions += 1
        transition_rate = transitions / (len(s) - 1)

        # Short strings (4-7 chars): require all 3 char types (upper + lower
        # + digit). This is extremely rare in real words but common in random
        # suffixes like lqDkW1, j7x4Qs, Bb1sqj.
        # We do NOT flag 2-type short strings (upper+lower only) because
        # CamelCase words like WebKit, NumPy, SciPy are statistically
        # indistinguishable from 2-type random strings at this length.
        if len(s) < 8:
            return n_types >= 3

        return transition_rate > 0.6

    @staticmethod
    def _classify_port(port_str: str) -> str:
        """Map a port number to a semantic category token."""
        try:
            port = int(port_str)
        except (ValueError, TypeError):
            return PORT_LOW
        if port in WELL_KNOWN_PORTS:
            return WELL_KNOWN_PORTS[port]
        if port <= 1023:
            return PORT_LOW
        if port <= 49151:
            return PORT_REG
        return PORT_EPH

    def _normalize_part(self, part: str) -> List[str]:
        """Normalize a sub-part from compound name or extension base split.

        Handles hashes, random strings, numbers, and camelCase.
        Unlike _pretokenize_segment, does NOT match directory names
        (e.g. 'dev' stays as 'dev', not [DEV]).
        """
        if not part:
            return []
        if RE_HASH.match(part):
            return [HASH]
        if part.isdigit():
            return [part] if int(part) <= 255 else [NUM]
        if self._is_random_string(part):
            return [RANDOM]
        if any(c.isupper() for c in part[1:]):
            camel = re.split(r"(?<=[a-z])(?=[A-Z])", part)
            if len(camel) > 1:
                return [p.lower() for p in camel]
        return [part.lower()]

    def _split_with_sep_tokens(self, s: str) -> List[str]:
        """Split a string by separators (-, _, ., +) emitting separator tokens between parts.

        Leading/trailing separators are dropped. Consecutive separators collapse to
        the last separator type encountered before a non-empty part.

        Examples:
            "x86_64-linux-gnu"  -> ["x86", SEP_UNDERSCORE, "64", SEP_DASH, "linux", SEP_DASH, "gnu"]
            "traceback.cpython" -> ["traceback", SEP_DOT, "cpython"]
            "__pycache__"       -> ["pycache"]   (leading/trailing underscores ignored)
            "lost+found"        -> ["lost", SEP_PLUS, "found"]
        """
        raw = RE_SEPARATORS_CAPTURE.split(s)
        result: List[str] = []
        current_sep: Optional[str] = None
        for piece in raw:
            if piece in SEP_TOKEN_MAP:
                current_sep = SEP_TOKEN_MAP[piece]
            elif piece:  # non-empty, non-separator content
                if result and current_sep is not None:
                    result.append(current_sep)
                result.extend(self._normalize_part(piece))
                current_sep = None
            # empty pieces from leading/trailing/consecutive separators: just skip
        return result

    def _pretokenize_query_string(self, qs: str) -> List[str]:
        """Normalize a URL query string into structured tokens.

        Splits on '&' into individual key=value pairs.  Keys are tokenized as
        words; values are normalized (numbers -> [NUM], hashes -> [HASH],
        random strings -> [RANDOM], semantic words -> BPE candidates).
        Pairs are separated by [SEP_AMP].

        Examples:
            "detail=detail&id=5457"     -> ["detail", "detail", SEP_AMP, "id", NUM]
            "para1=leftnav&type=show"   -> ["para1", "leftnav", SEP_AMP, "type", "show"]
            "calendar&date=7"           -> ["calendar", SEP_AMP, "date", "7"]
            "action=view&hash=a3f9..."  -> ["action", "view", SEP_AMP, "hash", HASH]
        """
        tokens = []
        pairs = qs.split("&")
        for i, pair in enumerate(pairs):
            if not pair:
                continue
            if i > 0:
                tokens.append(SEP_AMP)
            if "=" in pair:
                k, _, v = pair.partition("=")
                if k:
                    tokens.extend(self._split_with_sep_tokens(k) or [k.lower()])
                if v:
                    if v.isdigit():
                        tokens.append(v if int(v) <= 255 else NUM)
                    elif RE_HASH.match(v):
                        tokens.append(HASH)
                    elif self._is_random_string(v):
                        tokens.append(RANDOM)
                    else:
                        tokens.extend(self._split_with_sep_tokens(v) or [v.lower()])
            else:
                tokens.extend(self._split_with_sep_tokens(pair) or [pair.lower()])
        return tokens

    def _pretokenize_segment(self, seg: str) -> List[str]:
        """Pre-tokenize a single path component or word.

        Returns a mix of domain tokens (e.g. [ETC], [HASH]) and raw words
        that will later go through BPE.
        """
        if not seg:
            return []

        lower = seg.lower()

        # Well-known directories
        if lower in UNIX_DIRS:
            return [UNIX_DIRS[lower]]
        if lower in WIN_DIRS:
            return [WIN_DIRS[lower]]

        # Hash (MD5/SHA1/SHA256)
        if RE_HASH.match(seg):
            return [HASH]

        # GUID / UUID (standalone)
        if RE_GUID.match(seg):
            return [GUID]

        # Embedded GUID (e.g. "Processid:{E10F6C3A-...}")
        guid_search = RE_GUID_INNER.search(seg)
        if guid_search:
            tokens = []
            prefix = seg[: guid_search.start()].rstrip(":{")
            if prefix:
                tokens.extend(self._pretokenize_segment(prefix))
            tokens.append(GUID)
            suffix = seg[guid_search.end() :].lstrip("}")
            if suffix:
                tokens.extend(self._pretokenize_segment(suffix))
            return tokens

        # Windows SID
        if RE_SID.match(seg):
            return [RANDOM]

        # Version string
        if RE_VERSION.match(seg):
            return [VER]

        # Timestamp
        if RE_TIMESTAMP.match(seg):
            return [TIMESTAMP]

        # Environment variable
        if RE_ENV.match(seg):
            return [ENV_VAR]

        # Pure number: keep small numbers (IP octets, exit codes), collapse large
        if seg.isdigit():
            return [seg] if int(seg) <= 255 else [NUM]

        # File extension extraction (e.g. traceback.cpython-34.pyc -> base + [EXT_pyc])
        # Must come before _is_random_string so that compound names like
        # "build-a1b2c3d.o" are split into base + ext before entropy is measured.
        ext_match = RE_EXT.match(seg)  # match original case (regex is IGNORECASE)
        if ext_match:
            base, ext = ext_match.group(1), ext_match.group(2).lower()
            tokens = self._split_with_sep_tokens(base)
            tokens.append(f"[EXT_{ext}]")
            return tokens

        # Versioned shared object: libfoo.so.1.2 -> libfoo [EXT_so] 1 2
        so_match = RE_SO_VERSIONED.match(seg)
        if so_match:
            base = so_match.group(1)
            tokens = self._split_with_sep_tokens(base)
            tokens.append("[EXT_so]")
            remainder = seg[len(base) + 3:]  # skip ".so"
            for v in remainder.split("."):
                if v.isdigit():
                    tokens.append(v if int(v) <= 255 else NUM)
            return tokens

        # URL query string (e.g. "detail=detail&id=5457", "calendar&date=7")
        if "&" in seg and RE_QUERY_STRING.match(seg):
            return self._pretokenize_query_string(seg)

        # Compound name with separators (e.g. x86_64-linux-gnu, lost+found).
        # Must come before _is_random_string: separators inflate the char-type
        # transition rate, causing legitimate hash-bearing names like
        # "gk-0f1fc345" to be wrongly flagged as random.  Each sub-part is
        # evaluated independently by _normalize_part which has its own check.
        if RE_SEPARATORS_CAPTURE.search(seg):
            result = self._split_with_sep_tokens(seg)
            if result:
                return result

        # Random string (high entropy + mixed char types).
        # Reached only by bare words with no separators or known extension —
        # keeping it before camelCase prevents random strings like "pEja72mA"
        # from being wrongly split on case boundaries.
        if self._is_random_string(seg):
            return [RANDOM]

        # camelCase splitting (e.g. networkHistory -> network, history)
        if any(c.isupper() for c in seg[1:]):
            camel_parts = re.split(r"(?<=[a-z])(?=[A-Z])", seg)
            if len(camel_parts) > 1:
                return [p.lower() for p in camel_parts]

        # Single word -> BPE candidate
        return [lower]

    def _pretokenize_path_component(self, seg: str) -> List[str]:
        """Tokenize one path component for an absolute path.

        Well-known directories use a '/'-prefixed domain token ([/USR], [/LIB], …)
        so each component is a single, unambiguous token.  For all other segments
        the '/' is prepended to the first raw BPE word so that even if BPE splits
        the word later (e.g. /traceback -> /trac + eback), every split piece except
        the first lacks a leading '/' and is recognisable as a continuation.

        Domain tokens (HASH, GUID, VER, …) already carry enough identity and are
        left without a '/' prefix.
        """
        lower = seg.lower()
        if lower in UNIX_DIRS_PATH:
            return [UNIX_DIRS_PATH[lower]]
        if lower in WIN_DIRS_PATH:
            return [WIN_DIRS_PATH[lower]]

        seg_toks = self._pretokenize_segment(seg)
        if not seg_toks:
            return seg_toks

        first = seg_toks[0]
        if first.startswith("[") and first.endswith("]"):
            # Domain token (HASH, GUID, VER, EXT_*, …) – keep as-is
            return seg_toks
        return ["/" + first] + seg_toks[1:]

    def _pretokenize_path(self, path_str: str) -> List[str]:
        """Pre-tokenize a path: root marker + slash-prefixed component tokens.

        For absolute paths each component token embeds the leading '/':
          /usr/lib/traceback.cpython-34.pyc
          -> [ROOT] [/USR] [/LIB] /traceback [SEP_DOT] cpython [SEP_DASH] 34 [EXT_pyc]

        BPE fragments of one component are distinguishable from new components
        because only the first fragment starts with '/'.  Relative paths fall back
        to the same plain token representation as before.
        """
        tokens = []
        path = path_str.replace("\\", "/").strip().strip('"').strip("'")

        # Windows device path prefix: \\?\C:\... or //?/C:/...
        if re.match(r"^/+\?/+", path):
            path = re.sub(r"^/+\?/+", "", path)

        # Windows registry path detection (e.g. "registry/machine/software/...")
        path_lower = path.lower()
        for hive_prefix, hive_token in REG_HIVE_MAP.items():
            hive_slash = hive_prefix.replace("\\", "/")
            if path_lower.startswith(hive_slash):
                tokens.append(hive_token)
                path = path[len(hive_slash):].lstrip("/")
                # Tokenize remaining registry subkeys as path components
                for seg in path.split("/"):
                    if seg:
                        seg_lower = seg.lower()
                        if seg_lower in REG_SUBKEY_MAP:
                            tokens.append(REG_SUBKEY_MAP[seg_lower])
                        else:
                            tokens.extend(self._pretokenize_path_component(seg))
                return tokens

        is_absolute = False
        if len(path) >= 2 and path[0].isalpha() and path[1] == ":":
            tokens.append(WIN_ROOT)
            path = path[2:].lstrip("/")
            is_absolute = True
        elif path.startswith("/"):
            tokens.append(ROOT)
            path = path.lstrip("/")
            is_absolute = True

        for seg in path.split("/"):
            if seg:
                if is_absolute:
                    tokens.extend(self._pretokenize_path_component(seg))
                else:
                    tokens.extend(self._pretokenize_segment(seg))

        return tokens

    @staticmethod
    def _pretokenize_ip(ip_str: str) -> List[str]:
        """Pre-tokenize an IP address: octets only, no dots."""
        match = RE_IP.match(ip_str)
        if match:
            return [match.group(i) for i in range(1, 5)]
        return [ip_str.lower()]

    @staticmethod
    def _pretokenize_ipv6(ip_str: str) -> List[str]:
        """Pre-tokenize an IPv6 address: [IPV6] prefix + hex groups."""
        groups = [g for g in ip_str.split(":") if g]
        return [IPV6] + [g.lower() for g in groups]

    @staticmethod
    def _classify_ip(ip_str: str) -> str:
        """Classify an IP address as private, public, or localhost.

        Heuristics:
        - Localhost: 127.0.0.0/8, 0.0.0.0
        - Private (RFC 1918 + link-local + CGNAT):
            10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16,
            169.254.0.0/16 (link-local), 100.64.0.0/10 (CGNAT)
        - Everything else: public
        """
        match = RE_IP.match(ip_str)
        if not match:
            return PUBLIC_IP
        o1, o2 = int(match.group(1)), int(match.group(2))
        if o1 == 127 or (o1 == 0 and o2 == 0):
            return LOCALHOST_IP
        if o1 == 10:
            return PRIVATE_IP
        if o1 == 172 and 16 <= o2 <= 31:
            return PRIVATE_IP
        if o1 == 192 and o2 == 168:
            return PRIVATE_IP
        if o1 == 169 and o2 == 254:
            return PRIVATE_IP
        if o1 == 100 and 64 <= o2 <= 127:
            return PRIVATE_IP
        return PUBLIC_IP

    @staticmethod
    def _classify_ipv6(ip_str: str) -> str:
        """Classify an IPv6 address as private, public, or localhost.

        Heuristics:
        - Localhost: ::1
        - Private: fc00::/7 (unique local), fe80::/10 (link-local)
        - Everything else: public
        """
        # Normalize: expand :: and lowercase
        normalized = ip_str.strip().lower()
        # ::1 is loopback
        if normalized == "::1" or normalized == "0:0:0:0:0:0:0:1":
            return LOCALHOST_IP
        # fc00::/7 — unique local addresses (fc00:: and fd00::)
        if normalized.startswith("fc") or normalized.startswith("fd"):
            return PRIVATE_IP
        # fe80::/10 — link-local
        if normalized.startswith("fe80"):
            return PRIVATE_IP
        return PUBLIC_IP

    @staticmethod
    def _looks_like_path(s: str) -> bool:
        return "/" in s or "\\" in s or (len(s) >= 2 and s[1] == ":")

    def _tokenize_flag(self, flag_str: str) -> List[str]:
        """Tokenize a CLI flag, preserving its content after [FLAG].

        Emits [FLAG] followed by the flag name/letters so the model retains
        flag semantics rather than losing them in a single [FLAG] token.

        Examples:
            "-sh"         -> [[FLAG], "sh"]
            "-idrc"       -> [[FLAG], "idrc"]
            "--verbose"   -> [[FLAG], "verbose"]
            "--no-verify" -> [[FLAG], "no", SEP_DASH, "verify"]
        """
        content = flag_str.lstrip("-")
        tokens = [FLAG]
        if content:
            # Split long flags by '-', emitting SEP_DASH between parts.
            # Use _normalize_part (not _pretokenize_segment) to avoid mapping
            # flag words like "lib" to directory tokens like [LIB].
            sub_parts = content.split("-")
            for i, sp in enumerate(sub_parts):
                if sp:
                    if i > 0:
                        tokens.append(SEP_DASH)
                    tokens.extend(self._normalize_part(sp))
        return tokens

    def _pretokenize_node_simple(self, _node_type: str, label_str: str) -> List[str]:
        """Minimal pre-tokenization for bpe_only mode: just lowercase and split on whitespace."""
        parts = label_str.split()
        if parts and parts[0] in ("subject", "file", "netflow"):
            parts = parts[1:]
        return [p.lower() for p in parts if p and p != "None"]

    def _pretokenize_node(self, node_type: str, label_str: str) -> List[str]:
        """Pre-tokenize a node label based on its type."""
        parts = label_str.split()

        # Strip type prefix if present (e.g. "subject", "file", "netflow")
        if parts and parts[0] in ("subject", "file", "netflow"):
            parts = parts[1:]

        if not parts:
            return []

        # ── Netflow ──
        if node_type == "netflow":
            tokens = []
            normalize_ips = self.normalize_netflow_ips

            def _tokenize_ip(ip_str):
                """Tokenize an IP: category token if normalizing, octets otherwise.
                If expand_netflow_ips is set, emit both category + octets."""
                if normalize_ips:
                    cat = [self._classify_ip(ip_str)]
                    if self.expand_netflow_ips:
                        return cat + self._pretokenize_ip(ip_str)
                    return cat
                return self._pretokenize_ip(ip_str)

            for part in parts:
                if "->" in part:
                    # Format: "IP:port->IP:port"
                    for endpoint in part.split("->"):
                        ip_port = endpoint.rsplit(":", 1)
                        if len(ip_port) == 2 and RE_IP.match(ip_port[0]):
                            tokens.extend(_tokenize_ip(ip_port[0]))
                            tokens.append(self._classify_port(ip_port[1]))
                        else:
                            tokens.extend(self._pretokenize_segment(endpoint))
                elif RE_IP.match(part):
                    tokens.extend(_tokenize_ip(part))
                elif RE_IPV6.match(part):
                    if normalize_ips:
                        tokens.append(self._classify_ipv6(part))
                        if self.expand_netflow_ips:
                            tokens.extend(self._pretokenize_ipv6(part))
                    else:
                        tokens.extend(self._pretokenize_ipv6(part))
                elif part.isdigit():
                    tokens.append(self._classify_port(part))
                elif ":" in part:
                    ip_port = part.rsplit(":", 1)
                    if RE_IP.match(ip_port[0]):
                        tokens.extend(_tokenize_ip(ip_port[0]))
                        tokens.append(self._classify_port(ip_port[1]))
                    else:
                        tokens.extend(self._pretokenize_segment(part))
                else:
                    tokens.extend(self._pretokenize_segment(part))
            return tokens

        # ── File / Subject ──
        tokens = []
        for part in parts:
            if not part or part == "None":
                continue
            if self._looks_like_path(part):
                tokens.extend(self._pretokenize_path(part))
            elif RE_FLAG.match(part):
                tokens.extend(self._tokenize_flag(part))
            elif "&" in part and RE_QUERY_STRING.match(part):
                # Full URL query string (e.g. "detail=detail&id=5457&type=show")
                tokens.extend(self._pretokenize_query_string(part))
            elif "=" in part and not part.startswith("="):
                # key=value argument (e.g. --channel=123, --output=/tmp/f)
                k, _, v = part.partition("=")
                if RE_FLAG.match(k):
                    tokens.extend(self._tokenize_flag(k))
                else:
                    tokens.extend(self._pretokenize_segment(k))
                if v:
                    if self._looks_like_path(v):
                        tokens.extend(self._pretokenize_path(v))
                    else:
                        tokens.extend(self._pretokenize_segment(v))
            else:
                tokens.extend(self._pretokenize_segment(part))
        return tokens

    # ── Category token matching ──────────────────────────────────────────

    @staticmethod
    def _get_category_token(node_type: str, label_str: str) -> Optional[str]:
        """Return the canonical category token for a node, or None if no match.

        For processes: extracts binary name and looks up in PROC_CATEGORIES.
        For files: matches path against FILE_DIR_CATEGORIES (most-specific-first).
        For sockets: no additional category (already have IP/port categories).
        """
        if node_type == "subject":
            return match_proc_category(label_str)
        elif node_type == "file":
            return match_file_category(label_str)
        return None

    def _pretokenize_node_with_category(self, node_type: str, label_str: str) -> List[str]:
        """Pre-tokenize a node and prepend any matching category token.

        Used for encoder input (full detail + category signal) and for
        continue-pretrain decoder targets (same: full tokens + category).
        """
        tokens = self._pretokenize_node(node_type, label_str)
        cat = self._get_category_token(node_type, label_str)
        if cat:
            tokens = [cat] + tokens
        return tokens

    @staticmethod
    def _is_canonical_token(tok: str) -> bool:
        """Return True if tok is a token to keep in canonical neighbor targets.

        Kept: category tokens ([CAT_*], [FCAT_*]), port tokens ([PORT_*]),
        IP category tokens, [IPV6], file extension tokens ([EXT_*]),
        null-label tokens ([NO_CMD], [NO_PATH], [NO_IP]).

        Stripped: path structure ([ROOT], [/USR], [/LIB], …), separators
        ([SEP_*]), value tokens ([HASH], [GUID], [RANDOM], [NUM], [VER],
        [TIMESTAMP], [ENV], [FLAG]), and all other domain tokens.
        """
        if not (tok.startswith("[") and tok.endswith("]")):
            return False
        return (
            tok.startswith("[CAT_")
            or tok.startswith("[FCAT_")
            or tok.startswith("[PORT_")
            or tok.startswith("[EXT_")
            or tok in (PRIVATE_IP, PUBLIC_IP, LOCALHOST_IP, IPV6,
                       PORT_LOW, PORT_REG, PORT_EPH,
                       NO_CMD, NO_PATH, NO_IP)
        )

    def _canonicalize_pretokens(self, tokens: List[str], category_token: Optional[str] = None) -> List[str]:
        """Filter pre-tokens to keep only canonical tokens.

        Used for decoder reconstruction targets during main pretraining:
        strips dataset-specific words AND structural noise (paths, flags,
        separators), keeping only category tokens, port/IP categories,
        file extensions, and null-label markers.
        """
        canonical = []
        if category_token:
            canonical.append(category_token)
        for tok in tokens:
            if self._is_canonical_token(tok):
                canonical.append(tok)
        return canonical

    def tokenize_node_canonical(self, node_type: str, label_str: str) -> List[int]:
        """Tokenize a node keeping only special tokens + category token.

        Used for neighbor nodes in decoder targets during main pretraining.
        The node type prefix is always included, then the canonical tokens.
        """
        label_str = _BINARY_GARBAGE_RE.sub("", label_str).strip()

        ids = []

        # Node type prefix (skip when strip_entity_type is active)
        if not self.strip_entity_type:
            type_tok = NODE_TYPE_TOKENS.get(node_type)
            if type_tok and type_tok in self.token2id:
                ids.append(self.token2id[type_tok])

        # Get category token
        cat = self._get_category_token(node_type, label_str)

        # Pre-tokenize normally, then filter to special tokens only
        pretok_fn = self._pretokenize_node if self.mode == "domain_bpe" else self._pretokenize_node_simple
        raw_tokens = pretok_fn(node_type, label_str)
        canonical = self._canonicalize_pretokens(raw_tokens, cat)

        for tok in canonical:
            if tok in self.token2id:
                ids.append(self.token2id[tok])

        return ids

    # ── Phase 2: BPE ─────────────────────────────────────────────────────

    def _collect_bpe_words(self, corpus: List[List[str]]) -> Counter:
        """Count word frequencies for BPE training.

        Skips special tokens, words already registered in vocab, and words
        containing non-ASCII characters (non-ASCII base chars are still
        registered individually in _train_bpe, but we don't want BPE to learn
        merges of garbage Unicode sequences from malformed labels).
        """
        freq = Counter()
        for tokens in corpus:
            for tok in tokens:
                if tok in self.token2id:
                    continue
                if tok.startswith("[") and tok.endswith("]"):
                    continue
                if not tok.isascii():
                    continue
                freq[tok] += 1
        return freq

    def _train_bpe(self, corpus: List[List[str]]):
        """Train BPE merges on pre-tokenized corpus."""
        # Pre-seed all printable ASCII as base character tokens.
        # This ensures BPE can encode any word, even with chars not in training.
        for i in range(32, 127):
            c = chr(i)
            if c not in self.token2id:
                self.token2id[c] = len(self.token2id)

        word_freqs = self._collect_bpe_words(corpus)
        if not word_freqs:
            return

        # Initialize: split each word into characters
        word_splits = {w: list(w) for w in word_freqs}

        # Add any non-ASCII characters found in corpus
        for chars in word_splits.values():
            for c in chars:
                if c not in self.token2id:
                    self.token2id[c] = len(self.token2id)

        n_merges = max(0, self.bpe_vocab_size - len(self.token2id))
        self.merges = []

        for _ in range(n_merges):
            # Count pair frequencies
            pair_freq = Counter()
            for word, freq in word_freqs.items():
                chars = word_splits[word]
                for i in range(len(chars) - 1):
                    pair_freq[(chars[i], chars[i + 1])] += freq

            if not pair_freq:
                break

            best_pair, best_count = pair_freq.most_common(1)[0]
            if best_count < 2:
                break

            self.merges.append(best_pair)
            merged_tok = best_pair[0] + best_pair[1]
            if merged_tok not in self.token2id:
                self.token2id[merged_tok] = len(self.token2id)

            # Apply merge across all words
            a, b = best_pair
            new_splits = {}
            for word, chars in word_splits.items():
                new_chars = []
                i = 0
                while i < len(chars):
                    if i < len(chars) - 1 and chars[i] == a and chars[i + 1] == b:
                        new_chars.append(merged_tok)
                        i += 2
                    else:
                        new_chars.append(chars[i])
                        i += 1
                new_splits[word] = new_chars
            word_splits = new_splits

    def _apply_bpe(self, word: str) -> List[str]:
        """Apply learned BPE merges to a word -> list of subword tokens."""
        if not word:
            return []

        chars = list(word)
        for a, b in self.merges:
            i = 0
            while i < len(chars) - 1:
                if chars[i] == a and chars[i + 1] == b:
                    chars = chars[:i] + [a + b] + chars[i + 2:]
                else:
                    i += 1
        return chars

    def _token_to_id(self, tok: str) -> int:
        """Convert a token to its ID, using UNK for unknown."""
        return self.token2id.get(tok, self.token2id.get(UNK, 0))

    # ── Vocabulary management ─────────────────────────────────────────────

    def build_vocab(
        self,
        indexid2msg: Dict[str, list],
        nodes_to_include: Optional[Set] = None,
        walk_label_corpus: Optional[List[Tuple[str, str]]] = None,
    ):
        """Build vocabulary: special tokens + domain tokens + BPE from training data.

        Args:
            indexid2msg: node ID → (node_type, label_str) mapping (used as fallback).
            nodes_to_include: if set, only include these node IDs (only used with indexid2msg).
            walk_label_corpus: if provided, use this walk-frequency-weighted list of
                (node_type, label_str) pairs instead of iterating over indexid2msg.
                Each entry corresponds to one node occurrence in a unique (prelabel-deduped)
                walk, so common nodes in many distinct walks get higher frequency weight.
        """
        self.token2id = {}

        # Special tokens (fixed positions)
        for tok in SPECIAL_TOKENS:
            self.token2id[tok] = len(self.token2id)

        # Node type tokens
        for tok in NODE_TYPE_TOKENS.values():
            if tok not in self.token2id:
                self.token2id[tok] = len(self.token2id)

        # Edge type tokens
        for tok in sorted(self.edge_tokens.values()):
            if tok not in self.token2id:
                self.token2id[tok] = len(self.token2id)

        # Domain-specific tokens (skipped in bpe_only mode)
        if self.mode == "domain_bpe":
            domain_toks = [
                HASH, GUID, RANDOM, TIMESTAMP, VER, NUM, ENV_VAR, FLAG,
                ROOT, WIN_ROOT, NO_CMD, NO_PATH, NO_IP,
                SEP_DASH, SEP_UNDERSCORE, SEP_DOT, SEP_PLUS, SEP_AMP,
                PORT_LOW, PORT_REG, PORT_EPH, IPV6,
                PRIVATE_IP, PUBLIC_IP, LOCALHOST_IP,
            ]
            domain_toks.extend(sorted(WELL_KNOWN_PORTS.values()))
            domain_toks.extend(sorted(set(UNIX_DIRS.values())))
            domain_toks.extend(sorted(set(WIN_DIRS.values())))
            domain_toks.extend(sorted(set(UNIX_DIRS_PATH.values())))  # [/USR], [/LIB], …
            domain_toks.extend(sorted(set(WIN_DIRS_PATH.values())))   # [/WINDOWS], …
            domain_toks.extend(f"[EXT_{e}]" for e in sorted(KNOWN_EXTENSIONS))

            # Canonical category tokens (process + file categories)
            if self.canonicalize_neighbors:
                domain_toks.extend(ALL_CATEGORY_TOKENS)

            for tok in domain_toks:
                if tok not in self.token2id:
                    self.token2id[tok] = len(self.token2id)

        if self.canonicalize_neighbors and self.mode == "domain_bpe":
            pretok_fn = self._pretokenize_node_with_category
        elif self.mode == "domain_bpe":
            pretok_fn = self._pretokenize_node
        else:
            pretok_fn = self._pretokenize_node_simple

        # Pre-tokenize training corpus
        corpus = []
        if walk_label_corpus is not None:
            # Walk-frequency-weighted: each unique walk contributes one entry per node,
            # so labels that appear in many distinct walks get proportionally higher weight.
            for node_type, label_str in walk_label_corpus:
                label_str = _BINARY_GARBAGE_RE.sub("", label_str).strip()
                corpus.append(pretok_fn(node_type, label_str))
        else:
            # Fallback: one entry per unique entity in indexid2msg (entity-count weighted).
            for node_id, (node_type, label_str) in indexid2msg.items():
                if nodes_to_include is not None and node_id not in nodes_to_include:
                    continue
                label_str = _BINARY_GARBAGE_RE.sub("", label_str).strip()
                corpus.append(pretok_fn(node_type, label_str))

        # Pre-register frequently-seen words as direct vocab entries.
        # Cap at half the BPE budget to leave room for BPE merges.
        word_freq = Counter(
            tok for tokens in corpus for tok in tokens
            if not (tok.startswith("[") and tok.endswith("]"))
        )
        max_direct = self.bpe_vocab_size // 2
        for word, _count in word_freq.most_common():
            if len(self.token2id) >= max_direct + len(SPECIAL_TOKENS) + 50:
                break
            if _count >= 2 and word not in self.token2id:
                self.token2id[word] = len(self.token2id)

        pre_bpe_size = len(self.token2id)
        self.pre_bpe_size = pre_bpe_size

        # Train BPE on remaining rare words (also adds character + merge tokens)
        self._train_bpe(corpus)

        self.id2token = {v: k for k, v in self.token2id.items()}
        self.vocab_size = len(self.token2id)

        source = "walk-based" if walk_label_corpus is not None else "entity-based"
        log(f"Tokenizer ({source}): {len(corpus)} corpus entries, "
            f"{len(word_freq)} unique words, "
            f"pre-BPE vocab={pre_bpe_size}, "
            f"BPE merges={len(self.merges)}, "
            f"final vocab={self.vocab_size}")

    def save(self, path: str):
        """Save tokenizer state."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "token2id": self.token2id,
                "vocab_size": self.vocab_size,
                "max_seq_len": self.max_seq_len,
                "merges": self.merges,
                "mode": self.mode,
                "normalize_netflow_ips": self.normalize_netflow_ips,
                "canonicalize_neighbors": self.canonicalize_neighbors,
            },
            path,
        )

    def dump_vocab(self, path: str):
        """Write vocab to a human-readable text file, split by step."""
        pre_bpe_size = getattr(self, "pre_bpe_size", None)
        sorted_tokens = sorted(self.token2id.items(), key=lambda x: x[1])
        with open(path, "w") as f:
            f.write(f"# Vocabulary: {self.vocab_size} tokens\n")
            if pre_bpe_size is not None:
                f.write(f"# Step 1 (direct): tokens 0–{pre_bpe_size - 1}  ({pre_bpe_size} tokens)\n")
                f.write(f"# Step 2 (BPE):    tokens {pre_bpe_size}–{self.vocab_size - 1}  ({self.vocab_size - pre_bpe_size} tokens)\n")
            f.write("\n")
            current_section = None
            for token, idx in sorted_tokens:
                if pre_bpe_size is not None:
                    section = 1 if idx < pre_bpe_size else 2
                    if section != current_section:
                        current_section = section
                        if section == 1:
                            f.write("# ── Step 1: special + domain + frequent words ──\n")
                        else:
                            f.write("\n# ── Step 2: BPE subwords ──\n")
                f.write(f"{idx}\t{token}\n")

    def load(self, path: str):
        """Load tokenizer state."""
        state = torch.load(path)
        self.token2id = state["token2id"]
        self.vocab_size = state["vocab_size"]
        self.max_seq_len = state["max_seq_len"]
        self.merges = state["merges"]
        self.mode = state["mode"]
        self.normalize_netflow_ips = state["normalize_netflow_ips"]
        self.canonicalize_neighbors = state["canonicalize_neighbors"]
        self.id2token = {v: k for k, v in self.token2id.items()}

    # ── Tokenization ──────────────────────────────────────────────────────

    def tokenize_node(self, node_type: str, label_str: str) -> List[int]:
        """Tokenize a node: [type_prefix] + domain_tokens / BPE_subwords.

        When canonicalize_neighbors is enabled, a matching category token
        is prepended after the type prefix (for encoder input: full detail
        + category signal).
        """
        # Sanitize binary garbage before tokenization
        label_str = _BINARY_GARBAGE_RE.sub("", label_str).strip()

        ids = []

        # Node type prefix (skip when strip_entity_type is active)
        if (not hasattr(self, "strip_entity_type")) or (not self.strip_entity_type):
            type_tok = NODE_TYPE_TOKENS.get(node_type)
            if type_tok and type_tok in self.token2id:
                ids.append(self.token2id[type_tok])

        # Prepend category token when canonicalization is active
        if self.canonicalize_neighbors:
            cat = self._get_category_token(node_type, label_str)
            if cat and cat in self.token2id:
                ids.append(self.token2id[cat])

        # Pre-tokenize, then convert to IDs
        pretok_fn = self._pretokenize_node if self.mode == "domain_bpe" else self._pretokenize_node_simple
        for tok in pretok_fn(node_type, label_str):
            if tok in self.token2id:
                ids.append(self.token2id[tok])
            else:
                # BPE fallback for unknown words
                for sub in self._apply_bpe(tok):
                    ids.append(self._token_to_id(sub))

        return ids

    def tokenize_edge(self, edge_type: str) -> List[int]:
        """Tokenize an edge type into a single token ID."""
        tok = self.edge_tokens.get(edge_type)
        if tok and tok in self.token2id:
            return [self.token2id[tok]]
        return []

    @property
    def forward_id(self) -> int:
        """Token ID for [FORWARD] direction token."""
        return self.token2id[FORWARD]

    @property
    def backward_id(self) -> int:
        """Token ID for [BACKWARD] direction token."""
        return self.token2id[BACKWARD]

    @property
    def forward_neigh_id(self) -> int:
        """Token ID for [FORWARD_NEIGH] direction token."""
        return self.token2id[FORWARD_NEIGH]

    @property
    def backward_neigh_id(self) -> int:
        """Token ID for [BACKWARD_NEIGH] direction token."""
        return self.token2id[BACKWARD_NEIGH]

    def tokenize_context_segment(
        self,
        edge_types: List[str],
        node_ids: List[str],
        indexid2msg: Dict[str, list],
        canonical: Optional[bool] = None,
    ) -> List[int]:
        """Tokenize an edge-first walk segment for T5 decoder targets.

        Produces: [edge0_token] [node0_tokens] [edge1_token] [node1_tokens] ...

        Used for forward/backward context segments split from a walk at the
        entity position. Each segment starts with an edge (the edge connecting
        the entity to its neighbor) followed by the neighbor node, then the
        next edge, etc.

        Args:
            edge_types: list of edge type strings (same length as node_ids)
            node_ids: list of node IDs (one per edge)
            indexid2msg: node ID to (node_type, label_str) mapping
            canonical: if True, neighbor nodes are canonicalized (only special
                tokens + category tokens). If False, full tokenization with
                category tokens prepended. If None (default), uses
                self.canonicalize_neighbors.

        Returns:
            token_ids: flat list of token IDs (truncated to max_seq_len)
        """
        use_canonical = canonical if canonical is not None else self.canonicalize_neighbors
        token_ids = []
        for etype, nid in zip(edge_types, node_ids):
            token_ids.extend(self.tokenize_edge(etype))
            if nid in indexid2msg:
                ntype, nlabel = indexid2msg[nid]
                if use_canonical:
                    token_ids.extend(self.tokenize_node_canonical(ntype, nlabel))
                else:
                    token_ids.extend(self.tokenize_node(ntype, nlabel))
        return token_ids[:self.max_seq_len]

    def tokenize_walk(
        self,
        walk_nodes: List[str],
        walk_edge_types: List[str],
        indexid2msg: Dict[str, list],
    ) -> Tuple[List[int], List[Tuple[int, int]], List[Tuple[int, int]]]:
        """Tokenize a walk into a flat token sequence with node and edge boundaries.

        Returns:
            token_ids: flat list of token IDs
            node_boundaries: list of (start, end) for each node's tokens
            edge_boundaries: list of (start, end) for each edge's tokens.
                edge_boundaries[j] is the edge added after node_boundaries[j].
        """
        token_ids = []
        node_boundaries = []
        edge_boundaries = []

        for i, node_id in enumerate(walk_nodes):
            if node_id not in indexid2msg:
                continue

            node_type, label_str = indexid2msg[node_id]
            node_tokens = self.tokenize_node(node_type, label_str)
            if not node_tokens:
                continue

            start = len(token_ids)
            token_ids.extend(node_tokens)
            node_boundaries.append((start, len(token_ids)))

            if i < len(walk_edge_types):
                edge_toks = self.tokenize_edge(walk_edge_types[i])
                if edge_toks:
                    edge_start = len(token_ids)
                    token_ids.extend(edge_toks)
                    edge_boundaries.append((edge_start, len(token_ids)))

        if len(token_ids) > self.max_seq_len:
            token_ids = token_ids[: self.max_seq_len]
            node_boundaries = [
                (s, min(e, self.max_seq_len))
                for s, e in node_boundaries
                if s < self.max_seq_len
            ]
            edge_boundaries = [
                (s, min(e, self.max_seq_len))
                for s, e in edge_boundaries
                if s < self.max_seq_len
            ]

        return token_ids, node_boundaries, edge_boundaries

    # ── Masking ───────────────────────────────────────────────────────────

    def set_mask_rate(self, percent_done: float, fixed_rate: float = 0.7, min_rate: float = 0.15):
        """Polynomial annealing of mask rate (same as CyberGFM)."""
        anneal = 1 - (percent_done ** 2)
        self.mask_rate = max(anneal * fixed_rate, min_rate)

    def mask_nodes(
        self,
        token_ids: List[int],
        node_boundaries: List[Tuple[int, int]],
    ) -> Tuple[List[int], List[int], List[bool]]:
        """Node-level masking: 80% [MASK], 10% random, 10% keep."""
        seq_len = len(token_ids)
        masked_ids = list(token_ids)
        targets = []
        predict_mask = [False] * seq_len

        n_nodes = len(node_boundaries)
        if n_nodes == 0:
            return masked_ids, targets, predict_mask

        mask_flags = torch.rand(n_nodes) < self.mask_rate
        if not mask_flags.any():
            mask_flags[torch.randint(0, n_nodes, (1,))] = True

        for idx in range(n_nodes):
            if not mask_flags[idx]:
                continue

            start, end = node_boundaries[idx]
            for pos in range(start, min(end, seq_len)):
                predict_mask[pos] = True
                targets.append(token_ids[pos])

                r = torch.rand(1).item()
                if r < 0.8:
                    masked_ids[pos] = self.mask_id
                elif r < 0.9:
                    masked_ids[pos] = torch.randint(0, self.vocab_size, (1,)).item()
                # else: keep original

        return masked_ids, targets, predict_mask

    def mask_walk(
        self,
        token_ids: List[int],
        node_boundaries: List[Tuple[int, int]],
        edge_boundaries: List[Tuple[int, int]],
    ) -> Tuple[List[int], List[int], List[bool]]:
        """Structured masking: mask nodes and edges, never adjacent pairs.

        When a node is masked, its adjacent edges stay visible so the model
        can use edge types as context for node prediction. When an edge is
        masked, its adjacent nodes stay visible so the model can use node
        identities to predict the relationship.

        Edge j is adjacent to node j (before) and node j+1 (after).
        """
        seq_len = len(token_ids)
        masked_ids = list(token_ids)
        targets = []
        predict_mask = [False] * seq_len

        n_nodes = len(node_boundaries)
        n_edges = len(edge_boundaries)

        if n_nodes == 0:
            return masked_ids, targets, predict_mask

        # Step 1: Select nodes to mask
        node_mask_flags = torch.rand(n_nodes) < self.mask_rate

        # Step 2: Find protected edges (adjacent to any masked node)
        # edge j is adjacent to node j and node j+1
        protected_edges = set()
        for ni in range(n_nodes):
            if not node_mask_flags[ni]:
                continue
            if ni - 1 >= 0 and ni - 1 < n_edges:
                protected_edges.add(ni - 1)
            if ni < n_edges:
                protected_edges.add(ni)

        # Step 3: Select unprotected edges to mask
        edge_mask_flags = torch.zeros(n_edges, dtype=torch.bool)
        for ei in range(n_edges):
            if ei not in protected_edges and torch.rand(1).item() < self.mask_rate:
                edge_mask_flags[ei] = True

        # Ensure at least one position is masked
        if not node_mask_flags.any() and not edge_mask_flags.any():
            node_mask_flags[torch.randint(0, n_nodes, (1,))] = True

        # Step 4: Apply masking to selected nodes
        for idx in range(n_nodes):
            if not node_mask_flags[idx]:
                continue
            start, end = node_boundaries[idx]
            for pos in range(start, min(end, seq_len)):
                predict_mask[pos] = True
                targets.append(token_ids[pos])
                r = torch.rand(1).item()
                if r < 0.8:
                    masked_ids[pos] = self.mask_id
                elif r < 0.9:
                    masked_ids[pos] = torch.randint(0, self.vocab_size, (1,)).item()

        # Step 5: Apply masking to selected edges
        for idx in range(n_edges):
            if not edge_mask_flags[idx]:
                continue
            start, end = edge_boundaries[idx]
            for pos in range(start, min(end, seq_len)):
                predict_mask[pos] = True
                targets.append(token_ids[pos])
                r = torch.rand(1).item()
                if r < 0.8:
                    masked_ids[pos] = self.mask_id
                elif r < 0.9:
                    masked_ids[pos] = torch.randint(0, self.vocab_size, (1,)).item()

        return masked_ids, targets, predict_mask

    def batch_tokenize(
        self,
        walks: List[Tuple[List[str], List[str]]],
        indexid2msg: Dict[str, list],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize a batch of walks without masking -> padded tensors.

        Used by causal LM models (LLaMA) that don't need masking.

        Returns:
            input_ids: [B, L] padded token IDs
            attention_mask: [B, L] True for real tokens
        """
        batch_token_ids = []
        for walk_nodes, walk_edge_types in walks:
            token_ids, _, _ = self.tokenize_walk(
                walk_nodes, walk_edge_types, indexid2msg
            )
            if token_ids:
                batch_token_ids.append(token_ids)

        if not batch_token_ids:
            return torch.zeros(0, 1, dtype=torch.long), torch.zeros(0, 1, dtype=torch.bool)

        max_len = min(max(len(s) for s in batch_token_ids), self.max_seq_len)
        B = len(batch_token_ids)

        input_ids = torch.full((B, max_len), self.pad_id, dtype=torch.long)
        attention_mask = torch.zeros(B, max_len, dtype=torch.bool)

        for i in range(B):
            seq_len = min(len(batch_token_ids[i]), max_len)
            input_ids[i, :seq_len] = torch.tensor(batch_token_ids[i][:seq_len])
            attention_mask[i, :seq_len] = True

        return input_ids, attention_mask

    def batch_tokenize_and_mask(
        self,
        walks: List[Tuple[List[str], List[str]]],
        indexid2msg: Dict[str, list],
        mask_edges: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Tokenize and mask a batch of walks -> padded tensors.

        Args:
            mask_edges: if True, use structured masking (nodes + edges,
                never adjacent). If False, mask only nodes (original behavior).

        Returns:
            input_ids: [B, L] padded token IDs with masks applied
            attention_mask: [B, L] True for real tokens
            target_ids: [B, L] -100 for non-predicted, token_id for predicted
            predict_mask: [B, L] True at positions to predict
        """
        batch_masked = []
        batch_targets = []

        for walk_nodes, walk_edge_types in walks:
            token_ids, node_boundaries, edge_boundaries = self.tokenize_walk(
                walk_nodes, walk_edge_types, indexid2msg
            )
            if mask_edges and edge_boundaries:
                masked_ids, targets, predict_mask = self.mask_walk(
                    token_ids, node_boundaries, edge_boundaries
                )
            else:
                masked_ids, targets, predict_mask = self.mask_nodes(
                    token_ids, node_boundaries
                )
            batch_masked.append(masked_ids)
            batch_targets.append((targets, predict_mask))

        max_len = min(max((len(s) for s in batch_masked), default=1), self.max_seq_len)
        B = len(batch_masked)

        input_ids = torch.full((B, max_len), self.pad_id, dtype=torch.long)
        attention_mask = torch.zeros(B, max_len, dtype=torch.bool)
        target_tensor = torch.full((B, max_len), -100, dtype=torch.long)
        predict_tensor = torch.zeros(B, max_len, dtype=torch.bool)

        for i in range(B):
            seq_len = min(len(batch_masked[i]), max_len)
            input_ids[i, :seq_len] = torch.tensor(batch_masked[i][:seq_len])
            attention_mask[i, :seq_len] = True

            targets, p_mask = batch_targets[i]
            t_idx = 0
            for pos in range(seq_len):
                if p_mask[pos]:
                    target_tensor[i, pos] = targets[t_idx]
                    predict_tensor[i, pos] = True
                    t_idx += 1

        return input_ids, attention_mask, target_tensor, predict_tensor

    # ── Fine-tuning tokenization ──────────────────────────────────────────

    def tokenize_for_finetune(
        self,
        walk_nodes: List[str],
        walk_edge_types: List[str],
        dst_node_id: str,
        edge_type: str,
        indexid2msg: Dict[str, list],
        mode: str = "cls",
        dst_walk_nodes: List[str] = None,
        dst_walk_edge_types: List[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Tokenize a walk for fine-tuning (CLS or LP mode).

        CLS: src_walk + edge + dst_walk + [CLS] -> classify at [CLS] position
        LP:  src_walk + edge + [MASK]...         -> predict masked destination tokens

        dst_walk_nodes / dst_walk_edge_types: when provided (CLS mode only),
        the backward walk ending at dst is tokenized and appended after the
        edge token. The dst node itself is the last node in this walk, so its
        tokens are naturally the final context before [CLS]. When absent the
        behaviour falls back to bare dst identity tokens.

        Truncation is applied from the LEFT so that the target token(s) at the
        end of the sequence are always preserved.
        """
        src_ctx_ids, _, _ = self.tokenize_walk(walk_nodes, walk_edge_types, indexid2msg)
        src_ctx_ids.extend(self.tokenize_edge(edge_type))

        if mode == "cls":
            if dst_walk_nodes and len(dst_walk_nodes) > 1:
                dst_ctx_ids, _, _ = self.tokenize_walk(
                    dst_walk_nodes, dst_walk_edge_types, indexid2msg
                )
            else:
                # Fallback: bare dst identity tokens
                if dst_node_id in indexid2msg:
                    dst_type, dst_label = indexid2msg[dst_node_id]
                    dst_ctx_ids = self.tokenize_node(dst_type, dst_label)
                else:
                    dst_ctx_ids = []

            token_ids = src_ctx_ids + dst_ctx_ids + [self.cls_id]
            target_mask = [False] * (len(src_ctx_ids) + len(dst_ctx_ids)) + [True]

        elif mode == "lp":
            if dst_node_id in indexid2msg:
                dst_type, dst_label = indexid2msg[dst_node_id]
                dst_tokens = self.tokenize_node(dst_type, dst_label)
            else:
                dst_tokens = []

            token_ids = src_ctx_ids + [self.mask_id] * len(dst_tokens)
            target_mask = [False] * len(src_ctx_ids) + [True] * len(dst_tokens)

        else:
            raise ValueError(f"Unknown finetune mode: {mode}")

        # Truncate from the LEFT to always preserve the target at the end
        if len(token_ids) > self.max_seq_len:
            excess = len(token_ids) - self.max_seq_len
            token_ids = token_ids[excess:]
            target_mask = target_mask[excess:]

        return (
            torch.tensor(token_ids, dtype=torch.long),
            torch.ones(len(token_ids), dtype=torch.bool),
            torch.tensor(target_mask, dtype=torch.bool),
        )

    def tokenize_for_edge_scoring(
        self,
        walk_nodes: List[str],
        walk_edge_types: List[str],
        dst_node_id: str,
        edge_type: str,
        indexid2msg: Dict[str, list],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Tokenize walk + masked edge + visible destination for edge-type scoring.

        The edge token is masked; the model must predict it given the walk
        context and the (visible) destination node.

        Returns:
            input_ids: [L] token IDs with edge masked
            attention_mask: [L] all True
            edge_mask: [L] True at masked edge position(s)
            edge_target_ids: [L] true edge token IDs at masked positions, -100 elsewhere
        """
        token_ids, _, _ = self.tokenize_walk(walk_nodes, walk_edge_types, indexid2msg)

        edge_tokens = self.tokenize_edge(edge_type)
        edge_mask = [False] * len(token_ids)
        edge_target_ids = [-100] * len(token_ids)

        # Mask the edge token(s)
        for tok in edge_tokens:
            edge_mask.append(True)
            edge_target_ids.append(tok)
            token_ids.append(self.mask_id)

        # Append visible destination
        if dst_node_id in indexid2msg:
            dst_type, dst_label = indexid2msg[dst_node_id]
            dst_tokens = self.tokenize_node(dst_type, dst_label)
            token_ids.extend(dst_tokens)
            edge_mask.extend([False] * len(dst_tokens))
            edge_target_ids.extend([-100] * len(dst_tokens))

        # Truncate
        if len(token_ids) > self.max_seq_len:
            token_ids = token_ids[: self.max_seq_len]
            edge_mask = edge_mask[: self.max_seq_len]
            edge_target_ids = edge_target_ids[: self.max_seq_len]

        return (
            torch.tensor(token_ids, dtype=torch.long),
            torch.ones(len(token_ids), dtype=torch.bool),
            torch.tensor(edge_mask, dtype=torch.bool),
            torch.tensor(edge_target_ids, dtype=torch.long),
        )
