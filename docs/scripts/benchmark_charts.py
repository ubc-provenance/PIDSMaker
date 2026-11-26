"""Generates the charts and tables of the benchmark page from docs/docs/assets/benchmark/results.csv.

    python docs/scripts/benchmark_charts.py

Writes inline SVG charts and Markdown tables to docs/scripts/benchmark/, included by
docs/docs/guide/benchmark.md with --8<--. Colors are CSS variables (--pm-ds-*) of extra.css,
so the charts follow the light and dark themes.
"""

import csv
import math
import os
from html import escape

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "..", "docs", "assets", "benchmark", "results.csv")
OUT = os.path.join(HERE, "benchmark")

# One chart per row of datasets: three panels fit the width of the page
DATASET_ROWS = [["CADETS_E3", "THEIA_E5", "CLEARSCOPE_E5"], ["TRACE_E3", "optc_h201", "LANL"]]
DATASETS = [d for row in DATASET_ROWS for d in row]
# Systems that don't run on a dataset (see "Running across families" on the systems page)
NOT_APPLICABLE = {("LANL", s) for s in ["rcaid", "flash", "threatrace", "nodlink", "prographer"]}
FAMILIES = [("prov", "Provenance (PIDS)"), ("net", "Network (NIDS)")]
PROV = {
    "velox",
    "orthrus",
    "ocrapt",
    "rcaid",
    "flash",
    "kairos",
    "magic",
    "nodlink",
    "threatrace",
    "shadewatcher",
    "prographer",
}
# In the order of the Systems page: by family, then most recent first
NAMES = {
    "velox": "Velox",
    "orthrus": "Orthrus",
    "ocrapt": "OCR-APT",
    "rcaid": "R-Caid",
    "flash": "Flash",
    "kairos": "Kairos",
    "magic": "MAGIC",
    "nodlink": "NodLink",
    "prographer": "ProGrapher",
    "threatrace": "ThreaTrace",
    "shadewatcher": "ShadeWatcher",
    "lares": "LARES",
    "argus": "Argus",
    "jbeil": "Jbeil",
    "euler": "Euler",
    "egraphsage": "E-GraphSAGE",
    "anomal_e": "Anomal-E",
}
STAGES = [
    "construction",
    "transformation",
    "featurization",
    "feat_inference",
    "batching",
    "training",
    "evaluation",
    "triage",
]
# Stages of the stacked bars: the 8 tasks of the pipeline, in 5 groups
GROUPS = [
    ("Construction", ["construction"]),
    ("Featurization", ["featurization", "feat_inference"]),
    ("Training", ["training"]),
    ("Evaluation", ["evaluation"]),
    ("Other", ["transformation", "batching", "triage"]),
]


def segments(r, key):
    """Minutes (time) or GB (disk) of each group of stages. "Other" also holds what no task
    accounts for: the start of the run (imports, config) for time, top-level files for disk."""
    prefix, total, scale = {"time": ("t", "wall_min", 1 / 60), "disk": ("d", "disk_gb", 1)}[key]
    segs = [sum((r[f"{prefix}_{t}"] or 0) * scale for t in tasks) for _, tasks in GROUPS]
    segs[-1] += max(0.0, r[total] - sum(segs))
    return segs


METRICS = {
    # key: (csv column, axis title, unit, number format)
    "time": ("wall_min", "Wall time, from scratch (minutes)", "min", "{:,.0f}"),
    "ram": ("ram_gb", "Peak RAM (GB)", "GB", "{:,.1f}"),
    "gpu": ("gpu_gb", "Peak GPU memory (GB)", "GB", "{:,.1f}"),
    "disk": ("disk_gb", "Disk usage of the outputs (GB)", "GB", "{:,.1f}"),
}


def load():
    rows = list(csv.DictReader(open(CSV)))
    for r in rows:
        for k in ["wall_min", "ram_gb", "gpu_gb", "disk_gb"] + [
            f"{p}_{s}" for p in "td" for s in STAGES
        ]:
            r[k] = float(r[k]) if r.get(k) not in (None, "") else None
    return rows


def fmt(v, f):
    if v is None:
        return "–"
    if v < 1 and "{:,.0f}" == f:
        return "<1"
    if v < 0.1:
        return "<0.1"
    return f.format(v)


def tip_attrs(tip):
    """Hover text of a mark, shown by docs/docs/javascripts/benchmark.js; its first line is the
    heading. Also the accessible name of the mark."""
    text = escape(tip).replace("\n", "&#10;")
    return f'data-tip="{text}" role="img" aria-label="{text}"'


def tick_label(v):
    return f"{v:,.0f}" if v >= 1 else f"{v:g}"


def linear_ticks(vmax):
    """Ticks from 0 with a step of 1, 2, 2.5 or 5 x 10^k, about 3 steps up to the axis end."""
    raw = vmax / 3
    e = 10 ** math.floor(math.log10(raw))
    step = min((m * e for m in (1, 2, 2.5, 5, 10) if m * e >= raw), default=10 * e)
    n = math.ceil(vmax / step - 1e-9)
    return [i * step for i in range(n + 1)]


def axis_cut(values):
    """End of the axis of a metric: its largest value or, when a few runs are far above the
    others (more than 1.25 times above), the 95th percentile rounded up to a tick, so that the
    bars of the other runs stay readable."""
    if not values:
        return 0
    values = sorted(values)
    cut = linear_ticks(values[min(len(values) - 1, int(0.95 * len(values)))])[-1]
    return cut if values[-1] > 1.25 * cut else values[-1]


def bar_end(x0, x1, y, h, r=4):
    """Horizontal bar, square at the baseline x0 and rounded at its data end x1."""
    w = x1 - x0
    r = min(r, w / 2, h / 2)
    return (
        f"M{x0:.1f} {y:.1f}h{w - r:.1f}a{r:.1f} {r:.1f} 0 0 1 {r:.1f} {r:.1f}v{h - 2 * r:.1f}"
        f"a{r:.1f} {r:.1f} 0 0 1 {-r:.1f} {r:.1f}h{-(w - r):.1f}Z"
    )


def bar_panels(rows, key, datasets, stacked=False):
    """One panel of horizontal bars per dataset, on a linear axis from zero, the same rows in
    every panel. All the panels of a metric share one scale, that of its largest value over all
    the datasets, so that bars can be compared across datasets.
    With `stacked`, bars are split into the groups of stages of GROUPS. Systems are in the order
    of the Systems page (most recent first in each family), the same in every chart.
    The axis is cut (see `axis_cut`) when a few runs are far above the others: their bars stop
    at the cut with a break mark, and their hover text gives their value."""
    col, title, unit, f = METRICS[key]
    # Every system has a row, also before its first run has finished
    data = {("prov" if s in PROV else "net", s): {} for s in NAMES}
    for r in rows:
        data[(r["family"], r["system"])][r["dataset"]] = r
    # Over all the datasets, not only those of this chart: both charts of a metric share the scale
    vmax_all = axis_cut([r[col] for r in rows if r[col]])

    W, left, gap, top, row_h, head_h, bar_h = 760, 104, 22, 66 + (28 if stacked else 0), 22, 26, 12
    panel_w = (W - left - gap * (len(datasets) - 1)) / len(datasets) - 4
    systems = []
    for fam, label in FAMILIES:
        systems.append((fam, label, [s for (fm, s) in data if fm == fam]))  # in the order of NAMES
    plot_bottom = top + sum(head_h + row_h * len(m) for _, _, m in systems)
    H = plot_bottom + 26

    parts = [
        f'<svg class="pm-bench-chart" viewBox="0 0 {W} {H}" role="img" '
        f'aria-label="{escape(title)} of each system, one panel per dataset: {", ".join(datasets)}" '
        f'preserveAspectRatio="xMinYMin meet">',
        f'<text class="pm-bench-axis" x="0" y="16">{escape(title)}</text>',
    ]
    if stacked:  # legend of the stages, between the title and the panels
        lx = 0
        for k, (g, _) in enumerate(GROUPS):
            parts.append(
                f'<rect class="pm-bench-bar pm-st-{k + 1}" x="{lx}" y="{top - 58}" width="10" height="10" rx="2"/>'
            )
            parts.append(f'<text class="pm-bench-legend" x="{lx + 15}" y="{top - 49}">{g}</text>')
            lx += 30 + 7.2 * len(g)
    for i, d in enumerate(datasets):
        x0 = left + i * (panel_w + gap)
        ticks = linear_ticks(vmax_all) if vmax_all else [0, 1]

        def X(v, x0=x0, vmax=ticks[-1]):
            return x0 + panel_w * v / vmax

        parts.append(f'<text class="pm-bench-panel" x="{x0:.1f}" y="{top - 21}">{d}</text>')
        for t in ticks:
            cls = "pm-bench-base" if t == 0 else "pm-bench-grid"
            parts.append(
                f'<line class="{cls}" x1="{X(t):.1f}" x2="{X(t):.1f}" y1="{top - 8}" y2="{plot_bottom}"/>'
            )
            anchor = "start" if t == 0 else ("end" if t == ticks[-1] else "middle")
            parts.append(
                f'<text class="pm-bench-tick" x="{X(t):.1f}" y="{plot_bottom + 16}" text-anchor="{anchor}">{tick_label(t)}</text>'
            )

    y = top
    for fam, label, members in systems:
        parts.append(
            f'<text class="pm-bench-family pm-bench-family--{fam}" x="0" y="{y + 14}">{label}</text>'
        )
        y += head_h
        for s in members:
            name = NAMES.get(s, s)
            cy = y + row_h / 2
            parts.append(
                f'<text class="pm-bench-label" x="{left - 10}" y="{cy + 4:.1f}" text-anchor="end">{name}</text>'
            )
            for i, d in enumerate(datasets):
                x0 = left + i * (panel_w + gap)
                vmax = linear_ticks(vmax_all)[-1] if vmax_all else 1
                r = data[(fam, s)].get(d)
                if not r or not r[col]:
                    status = (
                        r["status"] if r else ("n/a" if (d, s) in NOT_APPLICABLE else "pending")
                    )
                    parts.append(
                        f'<text class="pm-bench-note" x="{x0 + 4:.1f}" y="{cy + 4:.1f}">{escape(status)}</text>'
                    )
                    continue
                limit = x0 + panel_w  # end of a cut axis
                x1 = min(x0 + max(panel_w * r[col] / vmax, 2), limit)
                tip = f"{name} on {d}: {fmt(r[col], f)} {unit}"
                if not stacked:
                    bar = f'<path class="pm-bench-bar pm-single" d="{bar_end(x0, x1, cy - bar_h / 2, bar_h)}"/>'
                else:
                    segs = segments(r, key)
                    tip += "\n" + "\n".join(
                        f"{g}: {fmt(v, f)} {unit}" for (g, _), v in zip(GROUPS, segs) if v > 0
                    )
                    # Each segment is followed by a 2px gap; the last one has the rounded end
                    drawn = [
                        (k, panel_w * v / vmax)
                        for k, v in enumerate(segs)
                        if panel_w * v / vmax >= 0.5
                    ]
                    # Beyond the end of a cut axis, segments are clipped
                    clipped, xs = [], x0
                    for k, w in drawn:
                        if xs >= limit - 1:
                            break
                        clipped.append((k, min(w, limit - xs)))
                        xs += w
                    drawn = clipped
                    bar, xs = "", x0
                    for n, (k, w) in enumerate(drawn):
                        if n == len(drawn) - 1:
                            bar += f'<path class="pm-bench-bar pm-st-{k + 1}" d="{bar_end(xs, max(xs + w, xs + 1), cy - bar_h / 2, bar_h)}"/>'
                        elif w > 2.5:
                            bar += f'<rect class="pm-bench-bar pm-st-{k + 1}" x="{xs:.1f}" y="{cy - bar_h / 2:.1f}" width="{w - 2:.1f}" height="{bar_h}"/>'
                        xs += w
                if r[col] > vmax:  # cut by the axis: a break mark across the end of the bar
                    xb = limit - 12
                    bar += (
                        f'<path class="pm-bench-break" d="M{xb:.1f} {cy + 8:.1f}l4 -16'
                        f'M{xb + 4:.1f} {cy + 8:.1f}l4 -16"/>'
                    )
                parts.append(
                    f'<g class="pm-bench-hit" {tip_attrs(tip)}>'
                    f'<rect x="{x0:.1f}" y="{y:.1f}" width="{panel_w:.1f}" height="{row_h}" fill="transparent"/>'
                    f"{bar}</g>"
                )
            y += row_h
    parts.append("</svg>")
    return "".join(parts)


def tables(rows):
    """One Markdown table per dataset, systems in the order of the systems page."""
    out = {}
    for d in DATASETS:
        lines = [
            "| System | Family | Wall time | Peak RAM | Peak GPU memory | Disk |",
            "|---|---|--:|--:|--:|--:|",
        ]
        for s in NAMES:
            r = next((r for r in rows if r["dataset"] == d and r["system"] == s), None)
            fam = "Provenance" if s in PROV else "Network"
            if r is None:
                if (d, s) not in NOT_APPLICABLE:
                    lines.append(f"| {NAMES[s]} | {fam} | pending | | | |")
                continue
            if r["status"] != "ok":
                lines.append(f"| {NAMES[s]} | {fam} | {r['status']} | | | |")
                continue
            lines.append(
                f"| {NAMES[s]} | {fam} | {fmt(r['wall_min'], '{:,.0f}')} min | {fmt(r['ram_gb'], '{:,.1f}')} GB "
                f"| {fmt(r['gpu_gb'], '{:,.1f}')} GB | {fmt(r['disk_gb'], '{:,.1f}')} GB |"
            )
        out[d] = "\n".join(lines) + "\n"
    return out


def stage_tables(rows):
    """Minutes spent in each stage, one Markdown table per dataset."""
    short = {
        "construction": "Construction",
        "transformation": "Transformation",
        "featurization": "Featurization",
        "feat_inference": "Feat. inference",
        "batching": "Batching",
        "training": "Training",
        "evaluation": "Evaluation",
        "triage": "Triage",
    }
    out = {}
    for d in DATASETS:
        lines = [
            "| System | " + " | ".join(short[s] for s in STAGES) + " |",
            "|---|" + "--:|" * len(STAGES),
        ]
        for s in NAMES:
            r = next(
                (r for r in rows if r["dataset"] == d and r["system"] == s and r["status"] == "ok"),
                None,
            )
            if r is None:
                continue
            cells = [
                fmt(r[f"t_{st}"] / 60, "{:,.1f}") if r[f"t_{st}"] is not None else "–"
                for st in STAGES
            ]
            lines.append(f"| {NAMES[s]} | " + " | ".join(cells) + " |")
        out[d] = "\n".join(lines) + "\n"
    return out


def main():
    rows = load()
    os.makedirs(OUT, exist_ok=True)
    for key in METRICS:
        # Time and disk are split by stage, peak memory is a single value
        kind = "stacked" if key in ("time", "disk") else "bars"
        for n, datasets in enumerate(DATASET_ROWS, 1):
            with open(os.path.join(OUT, f"{key}_{kind}_{n}.svg"), "w") as fh:
                fh.write(bar_panels(rows, key, datasets, stacked=kind == "stacked") + "\n")

    for d, t in tables(rows).items():
        with open(os.path.join(OUT, f"table_{d.lower()}.md"), "w") as fh:
            fh.write(t)
    for d, t in stage_tables(rows).items():
        with open(os.path.join(OUT, f"stages_{d.lower()}.md"), "w") as fh:
            fh.write(t)
    print(f"Wrote the charts and tables of {len(rows)} runs to {OUT}")


if __name__ == "__main__":
    main()
