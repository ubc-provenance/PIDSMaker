"""Substitution of variable values in labels by placeholder tokens ([HASH], [GUID], [NUM], ...).

Values such as hashes, UUIDs, random file names or ephemeral ports differ between
otherwise identical entities. Replacing them with a placeholder token makes the
featurization generalize across such entities.

Each tokenizer sets which patterns are substituted in `substitutions`:

    featurization:
      tokenization:
        used_method: nltk
        nltk:
          substitutions: hash, guid, number   # or `none`

The BPE tokenizer applies the patterns during its domain-specific pre-tokenization.
The other tokenizers apply them to the words of a label (separated by whitespace and
path / argument separators) before tokenizing it, see `Substitutor`.
"""

import re
from typing import Iterable, List, Optional, Tuple

from pidsmaker.config.config import SUBSTITUTION_METHODS as _CONFIG_SUBSTITUTION_METHODS

from pidsmaker.config.schemas import label_format

# ── Placeholder tokens ───────────────────────────────────────────────────────

HASH = "[HASH]"
GUID = "[GUID]"
RANDOM = "[RANDOM]"
TIMESTAMP = "[TIMESTAMP]"
VER = "[VER]"
NUM = "[NUM]"
ENV_VAR = "[ENV]"

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

# IP address categories
PRIVATE_IP = "[PRIVATE_IP]"
PUBLIC_IP = "[PUBLIC_IP]"
LOCALHOST_IP = "[LOCALHOST_IP]"

# ── Patterns ─────────────────────────────────────────────────────────────────

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

# Numbers up to this value are kept (IP octets, exit codes, small ids)
MAX_KEPT_NUMBER = 255

# Name of each substitution -> what it replaces
SUBSTITUTIONS = {
    "hash": f"hexadecimal strings of 7+ characters (MD5, SHA, commit ids) -> {HASH}",
    "guid": f"GUIDs / UUIDs, also inside a word -> {GUID}",
    "random": f"random-looking strings (mixed case and digits) and Windows SIDs -> {RANDOM}",
    "version": f"version numbers (1.2.3, v2.0-rc1) -> {VER}",
    "timestamp": f"words starting with a date (2019-05-08...) -> {TIMESTAMP}",
    "env_var": f"environment variables ($HOME, ${{PATH}}, %APPDATA%) -> {ENV_VAR}",
    "number": f"numbers above {MAX_KEPT_NUMBER} -> {NUM}",
    "ip": f"IP addresses of netflow entities -> {PRIVATE_IP}, {PUBLIC_IP} or {LOCALHOST_IP}",
    "port": f"ports of netflow entities -> well-known port tokens ({WELL_KNOWN_PORTS[443]}, ...) or {PORT_LOW}, {PORT_REG}, {PORT_EPH}",
}
SUBSTITUTION_METHODS = list(SUBSTITUTIONS.keys())
assert SUBSTITUTION_METHODS == _CONFIG_SUBSTITUTION_METHODS, "Update SUBSTITUTION_METHODS in config.py"
ALL_SUBSTITUTION_TOKENS = [HASH, GUID, RANDOM, TIMESTAMP, VER, NUM, ENV_VAR]


def parse_substitutions(value) -> frozenset:
    """Parses the `substitutions` argument: a comma-separated list of names, or `none`."""
    if value is None:
        raise ValueError("Missing `substitutions` argument of the tokenization.")
    if isinstance(value, str):
        names = [v.strip() for v in value.split(",") if v.strip()]
    else:
        names = [str(v).strip() for v in value]
    if names == ["none"] or not names:
        return frozenset()
    invalid = [n for n in names if n not in SUBSTITUTIONS]
    if invalid or "none" in names:
        raise ValueError(
            f"Invalid substitutions {invalid or ['none']}: use `none` or a list among {SUBSTITUTION_METHODS}."
        )
    return frozenset(names)


def get_substitutions(cfg, method: str) -> frozenset:
    """Substitutions set in `featurization.tokenization.<method>.substitutions`."""
    return parse_substitutions(getattr(cfg.featurization.tokenization, method).substitutions)


# ── Classifiers ──────────────────────────────────────────────────────────────


def is_random_string(s: str) -> bool:
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


def classify_port(port_str: str) -> str:
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


def classify_ip(ip_str: str) -> str:
    """Classify an IPv4 address as private, public, or localhost.

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


def classify_ipv6(ip_str: str) -> str:
    """Classify an IPv6 address: ::1 is localhost, fc00::/7 and fe80::/10 are private."""
    normalized = ip_str.strip().lower()
    if normalized == "::1" or normalized == "0:0:0:0:0:0:0:1":
        return LOCALHOST_IP
    if normalized.startswith("fc") or normalized.startswith("fd"):
        return PRIVATE_IP
    if normalized.startswith("fe80"):
        return PRIVATE_IP
    return PUBLIC_IP


# ── Word-level substitution (tokenizers other than BPE) ──────────────────────

# Separators of the words of process and file labels (kept in the label)
_WORD_SEPARATORS = re.compile(r"([\s/\\:=,&\"'])")
_WHITESPACE = re.compile(r"(\s+)")


class Substitutor:
    """Replaces the words of a label that match the enabled patterns by placeholder tokens.

    Words are separated by whitespace and by `/ \\ : = , & " '` (only whitespace in
    netflow labels, whose words are IPs, ports or `ip:port`). A word is substituted
    only if it fully matches a pattern, in the order used by the BPE tokenizer.
    """

    def __init__(self, substitutions: Iterable[str] = ()):
        self.substitutions = frozenset(substitutions)

    def __bool__(self):
        return bool(self.substitutions)

    def substitute_word(self, word: str, node_type: Optional[str] = None) -> Optional[str]:
        """The placeholder token of `word`, or None if no enabled pattern matches."""
        subs = self.substitutions
        if label_format(node_type) == "address":
            if "ip" in subs:
                if RE_IP.match(word):
                    return classify_ip(word)
                if RE_IPV6.match(word):
                    return classify_ipv6(word)
            if word.isdigit():
                return classify_port(word) if "port" in subs else None

        if "hash" in subs and RE_HASH.match(word):
            return HASH
        if "guid" in subs and RE_GUID.match(word):
            return GUID
        if "random" in subs and RE_SID.match(word):
            return RANDOM
        if "version" in subs and RE_VERSION.match(word):
            return VER
        if "timestamp" in subs and RE_TIMESTAMP.match(word):
            return TIMESTAMP
        if "env_var" in subs and RE_ENV.match(word):
            return ENV_VAR
        if word.isdigit():
            return NUM if "number" in subs and int(word) > MAX_KEPT_NUMBER else None
        if "random" in subs and is_random_string(word):
            return RANDOM
        return None

    def substitute(self, label: str, node_type: Optional[str] = None) -> List[Tuple[str, bool]]:
        """Splits `label` into pieces `(text, is_placeholder)`; joining the texts gives the
        label with its substituted words replaced by placeholders."""
        if not self.substitutions:
            return [(label, False)]

        pieces = []
        separators = _WHITESPACE if label_format(node_type) == "address" else _WORD_SEPARATORS
        for word in separators.split(label):
            if not word:
                continue
            placeholder = None if separators.fullmatch(word) else self.substitute_word(word, node_type)
            if placeholder is None and label_format(node_type) == "address" and ":" in word:
                # ip:port
                ip, _, port = word.rpartition(":")
                ip_tok = self.substitute_word(ip, node_type) or ip
                port_tok = self.substitute_word(port, node_type) or port
                if (ip_tok, port_tok) != (ip, port):
                    pieces.extend([(ip_tok, ip_tok != ip), (":", False), (port_tok, port_tok != port)])
                    continue
            pieces.append((placeholder, True) if placeholder else (word, False))
        return pieces

    def substitute_text(self, label: str, node_type: Optional[str] = None) -> str:
        """`label` with its substituted words replaced by placeholders."""
        if not self.substitutions:
            return label
        return "".join(text for text, _ in self.substitute(label, node_type))
