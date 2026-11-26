"""Terminal output of the pipeline: timestamped lines, task headers, tables, progress bars.

Colors and progress bars are only used when the output is a terminal. Log files (`--background`,
nohup, redirections, CI) get plain text, with one line per step instead of a progress bar.
$NO_COLOR (https://no-color.org) disables colors, $FORCE_COLOR forces them (e.g. `| less -R`).

The terminal is detected at import, before `wandb.init` redirects the output: with `--wandb`,
the logs of the run on W&B keep the colors of the terminal, which its Logs tab renders.
"""

import os
import re
import shutil
import sys
from datetime import datetime


def _is_terminal(stream):
    try:
        return stream.isatty() and os.environ.get("TERM") != "dumb"
    except (AttributeError, ValueError):
        return False


def _supports(stream, text):
    try:
        text.encode(getattr(stream, "encoding", None) or "ascii")
        return True
    except (LookupError, UnicodeEncodeError):
        return False


INTERACTIVE = _is_terminal(sys.stdout)
COLOR = False if os.environ.get("NO_COLOR") else bool(os.environ.get("FORCE_COLOR")) or INTERACTIVE
UNICODE = _supports(sys.stdout, "✓▶─·↓")
if COLOR and os.name == "nt":
    os.system("")  # enables the ANSI escape codes in the Windows console

CHECK, ARROW, RULE, DOT, DOWN = ("✓", "▶", "─", "·", "↓") if UNICODE else ("+", ">", "-", "|", "v")

_CODES = {"bold": 1, "dim": 2, "red": 31, "green": 32, "yellow": 33, "blue": 34, "cyan": 36}


def style(text, *styles):
    """`text` with ANSI styles (see `_CODES`) when colors are enabled."""
    if not COLOR or not styles:
        return str(text)
    return f"\033[{';'.join(str(_CODES[s]) for s in styles)}m{text}\033[0m"


_state = {"indent": "", "day": None, "seen_steps": set(), "artifact_dir": None}
_HASH = re.compile(r"\b[0-9a-f]{64}\b")


def _prefix():
    now = datetime.now()
    if now.date() != _state["day"]:
        _state["day"] = now.date()
        print(style(f"{RULE * 3} {now:%A %Y-%m-%d} {RULE * 3}", "dim"), flush=True)
    return f"{style(f'{now:%H:%M:%S}', 'dim')}  {_state['indent']}"


def log(msg="", *args, shorten=True, **kwargs):
    """Prints `msg` after the time of day (the date is printed when it changes). Messages
    starting with "Warning" are yellow and those starting with "Error" are red.

    With `shorten`, the paths in the artifacts folder are shown relative to it, with task
    hashes shortened to 8 characters (see `short_path`)."""
    # Flushed, so that logs redirected to a file (e.g. with `pidsmaker --background`) are
    # written as they come, instead of minutes later when the output buffer is full
    kwargs.setdefault("flush", True)
    msg = str(msg)
    if not msg.strip():
        print(*args, **kwargs)
        return
    if shorten:
        msg = short_paths(msg)
    if msg.lower().startswith("warning"):
        msg = style(msg, "yellow")
    elif msg.lower().startswith("error"):
        msg = style(msg, "red", "bold")
    print(f"{_prefix()}{msg}", *args, **kwargs)


def warning(msg):
    log(f"Warning: {msg}")


def set_artifact_dir(path):
    _state["artifact_dir"] = os.path.join(os.path.abspath(path), "") if path else None


def short_hash(text):
    """`text` with its task hashes (64 hex characters) shortened to 8 characters."""
    return _HASH.sub(lambda m: m.group(0)[:8], str(text))


def short_path(path):
    """`path`, relative to the artifacts folder, with shortened task hashes."""
    path = str(path)
    root = _state["artifact_dir"]
    if root and path.startswith(root):
        path = path[len(root) :]
    return short_hash(path)


def short_paths(text):
    """`text` with its paths in the artifacts folder shortened (see `short_path`)."""
    root = _state["artifact_dir"]
    if not root or root not in text:
        return text
    return re.sub(re.escape(root) + r"\S*", lambda m: short_path(m.group(0)), text)


def format_duration(seconds):
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, seconds = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return f"{minutes}m {seconds:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def format_value(value):
    """Short and aligned-friendly text of a metric."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, int) or float(value).is_integer() and abs(value) >= 1:
        return f"{int(value):,}".replace(",", " ") if abs(value) >= 10_000 else str(int(value))
    if value != value:
        return "nan"
    if value == 0:
        return "0"
    if abs(value) < 1e-3 or abs(value) >= 1e6:
        return f"{value:.2e}"
    return f"{value:.4f}".rstrip("0").rstrip(".") if abs(value) < 1 else f"{value:.4g}"


def task_start(task, task_path):
    """Header of a task that runs; the lines logged until `task_end` are indented."""
    _state["indent"] = ""
    _state["seen_steps"] = set()
    print(flush=True)
    log(
        f"{style(f'{ARROW} {task}', 'bold', 'cyan')}  {style(short_hash(_task_id(task_path)), 'dim')}"
    )
    _state["indent"] = "  "


def task_end(task, seconds):
    _state["indent"] = ""
    log(
        f"{style(f'{CHECK} {task}', 'green')}  {style(f'done in {format_duration(seconds)}', 'dim')}"
    )


def task_cached(task, task_path):
    _state["indent"] = ""
    log(
        f"{style(f'{CHECK} {task}', 'green')}  "
        f"{style(f'already done  {short_hash(_task_id(task_path))}', 'dim')}"
    )


def _task_id(task_path):
    match = _HASH.search(str(task_path))
    return match.group(0) if match else os.path.basename(os.path.normpath(task_path))


def section(title):
    """Title of a group of tasks (e.g. an iteration of an uncertainty experiment)."""
    _state["indent"] = ""
    print(flush=True)
    log(style(f"{RULE * 2} {title} {RULE * (max(4, 56 - len(title)))}", "bold"))


def log_once(msg):
    """Logs `msg` once per task (e.g. the same message in each epoch)."""
    if msg in _state["seen_steps"]:
        return
    _state["seen_steps"].add(msg)
    log(msg)


def step(desc):
    """Logs that a step starts, once per task in log files (e.g. "Training" in each epoch is
    logged once). In a terminal, the progress bar of the step shows it instead."""
    log_once(f"{desc}...")


def progress(iterator, desc="", **kwargs):
    """Progress bar of a loop in a terminal, removed when the loop ends. In log files, the
    step is logged once instead (see `step`)."""
    if not INTERACTIVE:
        step(desc)
        return iterator
    from tqdm import tqdm

    return tqdm(
        iterator,
        desc=f"{_prefix()}{desc}",
        leave=False,
        dynamic_ncols=True,
        bar_format="{desc} {percentage:3.0f}% {bar:24} {n_fmt}/{total_fmt} "
        "[{elapsed}<{remaining}, {rate_fmt}]"
        if _has_len(iterator, kwargs)
        else "{desc} {n_fmt} [{elapsed}, {rate_fmt}]",
        **kwargs,
    )


def _has_len(iterator, kwargs):
    return kwargs.get("total") is not None or hasattr(iterator, "__len__")


def table(header, rows, align=None):
    """Logs `rows` as aligned columns, under a dim `header`. The first column is aligned to
    the left and the others to the right, unless `align` ("<" or ">" per column)."""
    rows = [[str(c) for c in row] for row in rows]
    columns = list(zip(header, *rows))
    widths = [max(len(str(c)) for c in col) for col in columns]
    align = align or ["<"] + [">"] * (len(header) - 1)

    def fmt(row):
        return "   ".join(f"{c:{a}{w}}" for c, a, w in zip(row, align, widths)).rstrip()

    log(style(fmt([str(h) for h in header]), "dim"))
    for row in rows:
        log(fmt(row))


def metrics(stats, max_columns=3):
    """Logs the scalar metrics of `stats` as `name value` pairs, on as many columns (up to
    `max_columns`) as the width of the terminal allows (120 characters in log files)."""
    items = [
        (k, format_value(v))
        for k, v in stats.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    ]
    if not items:
        return
    width = shutil.get_terminal_size((120, 24)).columns if INTERACTIVE else 120
    width -= 10 + len(_state["indent"])  # time of day
    for columns in range(max_columns, 0, -1):
        rows = [items[i : i + columns] for i in range(0, len(items), columns)]
        key_widths = [max(len(r[c][0]) for r in rows if c < len(r)) for c in range(columns)]
        value_widths = [max(len(r[c][1]) for r in rows if c < len(r)) for c in range(columns)]
        if sum(key_widths) + sum(value_widths) + 5 * columns <= width:
            break
    for row in rows:
        cells = [
            f"{style(k.ljust(key_widths[c]), 'dim')} {v.rjust(value_widths[c])}"
            for c, (k, v) in enumerate(row)
        ]
        log("    ".join(cells))
