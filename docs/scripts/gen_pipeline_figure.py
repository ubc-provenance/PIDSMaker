#!/usr/bin/env python3
"""Generate the pipeline overview figure (docs/docs/img/pipeline.svg).

The boxes of the figure are read from `pidsmaker/config/config.py`, so the figure
follows the framework: a new encoder, objective, featurization method, pretrained
model... shows up after re-running this script.

Usage (from anywhere, no dependency other than the standard library):

    python docs/scripts/gen_pipeline_figure.py            # rewrite the figure
    python docs/scripts/gen_pipeline_figure.py --check    # exit 1 if the figure is stale

What to edit here when the config changes:
    LABELS            display name of a config key (fallback: "my_key" -> "My Key")
    HIDDEN            config keys that should not appear in the figure
    PRETRAINED_GROUPS how `featurization.pretrained.model_type` values are grouped
    build_stages()    which config entries feed each column
"""

import argparse
import importlib.util
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from xml.sax.saxutils import escape

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "pidsmaker" / "config" / "config.py"
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "docs" / "img" / "pipeline.svg"

# ---------------------------------------------------------------------------
# Content: how config keys are displayed
# ---------------------------------------------------------------------------

# Display names. A "context.key" entry wins over a plain "key" entry.
LABELS = {
    "none": "None",
    # transformation
    "undirected": "Undirected",
    "dag": "DAG",
    "rcaid_pseudo_graph": "R-CAID Pseudo",
    "synthetic_attack_naive": "Synthetic Attack",
    # featurization
    "word2vec": "Word2Vec",
    "doc2vec": "Doc2Vec",
    "fasttext": "FastText",
    "alacarte": "ALaCarte",
    "temporal_rw": "Temporal RW",
    "flash": "Flash",
    "hierarchical_hashing": "HFH",
    "magic": "MAGIC",
    "only_type": "Only Type",
    "only_ones": "Only Ones",
    "ocrapt_features": "OCR-APT",
    # featurization.pretrained.model_type
    "spider": "SPIDER",
    "gnn_distill": "GNN Distill",
    "behavior_cluster": "Behavior Cluster",
    "bert": "BERT",
    "roberta": "RoBERTa",
    "modernbert": "ModernBERT",
    "ropebert": "RoPE BERT",
    "logbert": "LogBERT",
    "llama": "Llama",
    "gpt2_pretrained": "GPT-2",
    "llama3_pretrained": "Llama 3.2",
    "opt_pretrained": "OPT",
    "graphmae": "GraphMAE",
    "gae": "GAE",
    "dgi": "DGI",
    "deepwalk": "DeepWalk",
    "node2vec": "node2vec",
    # batching
    "global_batching": "Global",
    "intra_graph_batching": "Intra-graph",
    "inter_graph_batching": "Inter-graph",
    # training.encoder
    "tgn": "TGN",
    "graph_attention": "Graph Attn",
    "sage": "SAGE",
    "gat": "GAT",
    "gin": "GIN",
    "rgcn": "RGCN",
    "rgcn_per_type": "RGCN per Type",
    "sum_aggregation": "Sum Aggr",
    "rcaid_gat": "R-CAID GAT",
    "magic_gat": "MAGIC GAT",
    "glstm": "GLSTM",
    "custom_mlp": "Custom MLP",
    "hetero_graph_transformer": "HGT",
    # training.decoder (decoders)
    "edge_mlp": "Edge MLP",
    "node_mlp": "Node MLP",
    "nodlink": "NodLink",
    "inner_product": "Inner Prod",
    # training.decoder (objectives)
    "predict_edge_type": "Edge Type",
    "predict_node_type": "Node Type",
    "predict_masked_struct": "Masked Struct",
    "predict_edge_supervised": "Supervised Edge",
    "detect_edge_few_shot": "Few-Shot Edge",
    "predict_edge_contrastive": "Contrastive",
    "reconstruct_node_features": "Node Feat Rec",
    "reconstruct_node_embeddings": "Node Emb Rec",
    "reconstruct_edge_embeddings": "Edge Emb Rec",
    "reconstruct_masked_features": "Mask Feat Rec",
    "one_class": "One-Class",
    # training.pretrained.finetune_mode
    "finetune.lp": "Link Pred",
    "finetune.mlm": "MLM",
    "finetune.cls": "CLS",
    "finetune.cls_attack": "CLS Attack",
    "finetune.edge_cls": "Edge CLS",
    "finetune.perplexity": "Perplexity",
    # evaluation
    "max_val_loss": "Max Val Loss",
    "mean_val_loss": "Mean Val Loss",
    "percentile": "Percentile",
    "threatrace": "ThreaTrace",
    "fixed_zero": "Fixed Zero",
    "ocrapt": "OCR-APT",
    "node_evaluation": "Node Level",
    "edge_evaluation": "Edge Level",
    "tw_evaluation": "Time Window",
    "node_tw_evaluation": "Node-TW Level",
    "queue_evaluation": "Queue Level",
    # triage
    "depimpact": "DepImpact",
    "ocrapt_subgraph": "OCR-APT",
}

# Config keys left out of the figure ("context.key" or plain "key").
HIDDEN = {
    "featurization.pretrained",  # shown through its model types instead
    "pretrained.gnn_distill",  # earlier variants of SPIDER's distillation
    "pretrained.behavior_cluster",
}

# Grouping of `featurization.pretrained.model_type`. A model type of the config
# that is not listed here is appended to the first group.
PRETRAINED_GROUPS = [
    (
        "Pretrained Transformer",
        [
            "spider",
            "gnn_distill",
            "behavior_cluster",
            "bert",
            "roberta",
            "modernbert",
            "ropebert",
            "logbert",
            "llama",
            "gpt2_pretrained",
            "llama3_pretrained",
            "opt_pretrained",
        ],
    ),
    ("Pretrained Graph", ["graphmae", "gae", "dgi", "deepwalk", "node2vec"]),
]

NODE_ENTITIES = {"subject": "Process", "file": "File", "netflow": "Socket"}
NODE_ATTRIBUTES = {"path": "Path", "cmd_line": "Cmd", "remote_ip": "IP", "remote_port": "Port"}

CAPTION = "Modular pipeline with on-disk caching and automatic restarting"

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

FONT = "Helvetica Neue, Helvetica, Arial, Liberation Sans, sans-serif"
NONE_FILL = "#E6E6E6"
PRETRAINED_FILL = "#A9D8B8"
PRETRAINED_STROKE = "#5E9E76"
ITEM_STROKE = "#9A9A9A"
PANEL_STROKE = "#999999"
DARK = "#4D4D4D"
GRAY = "#999999"

# stage -> (panel fill, header fill, [item fill of each kind of column])
PALETTE = {
    "construction": ("#DCEBF5", "#B4D2E6", ["#C8DEEE"]),
    "transformation": ("#FFFADC", "#F5EBB4", ["#F8F2C8"]),
    "featurization": ("#E1F5E1", "#C3E1C3", ["#D2EBD2"]),
    "batching": ("#FFEBF0", "#F5CDD7", ["#F8DCE4"]),
    "training": ("#F2F2F5", "#D7D7DC", ["#CDE6E6", "#F8EBD7", "#F5DEDE"]),
    "evaluation": ("#F5EBFA", "#DCC8EB", ["#E8D7F2", "#D7EBF2"]),
    "triage": ("#EBFAE1", "#CDE6BE", ["#DCEED0"]),
}

MARGIN = 24
PANEL_TOP = 190  # room above the panels for the Datasets / YAML / Metrics icons
PANEL_PAD_X = 16
PANEL_PAD_BOTTOM = 18
PANEL_GAP = 42  # horizontal space between two stages (holds the arrow)
COL_GAP = 14
HEADER_TOP = 22
HEADER_H = 54
HEADER_INSET = 14
GROUP_TITLE_H = 48  # vertical room taken by a group title
GROUP_GAP = 10  # extra space between two stacked groups
ITEM_H = 34
ITEM_PITCH = 41
ITEM_FONT = 16
TITLE_FONT = 15
HEADER_FONT = 17.5
MIN_COL_W = 126

# ---------------------------------------------------------------------------
# Reading the config
# ---------------------------------------------------------------------------


def load_config():
    """Imports config.py by path, so that the heavy `pidsmaker` package is not needed."""
    spec = importlib.util.spec_from_file_location("pidsmaker_config", CONFIG_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_fallback_labels = []


def label_of(key, context=None):
    for name in (f"{context}.{key}", key):
        if name in LABELS:
            return LABELS[name]
    _fallback_labels.append(f"{context}.{key}" if context else key)
    return key.replace("_", " ").title()


@dataclass
class Item:
    label: str
    fill: str
    stroke: str = ITEM_STROKE


@dataclass
class Group:
    title: str
    items: list


@dataclass
class Stage:
    key: str
    name: str
    columns: list  # list of columns, each one a list of Groups stacked vertically
    x: float = 0
    w: float = 0
    h: float = 0
    col_w: list = field(default_factory=list)


def make_items(keys, context, fill, stroke=ITEM_STROKE):
    """Config keys -> boxes. Hidden keys are dropped and `none` is moved last."""
    keys = [k for k in keys if k not in HIDDEN and f"{context}.{k}" not in HIDDEN]
    keys = [k for k in keys if k != "none"] + [k for k in keys if k == "none"]
    return [
        Item(label_of(k, context), NONE_FILL, ITEM_STROKE)
        if k == "none"
        else Item(label_of(k, context), fill, stroke)
        for k in keys
    ]


def arg_values(arg):
    """Options of an Arg: its `vals`, or else the quoted names of its description."""
    if arg.vals:
        return list(arg.vals)
    return re.findall(r"'(\w+)'", arg.desc or "")


def build_stages(cfg):
    tasks = cfg.TASK_ARGS
    stages = []

    # Construction: node attributes that can be used as textual features
    fill = PALETTE["construction"][2][0]
    features = [Item("Type", fill)]
    for entity, arg in tasks["construction"]["node_label_features"].items():
        for attr in arg.vals:
            if attr not in ("auto", "type"):
                entity_name = NODE_ENTITIES.get(entity, entity.title())
                attr_name = NODE_ATTRIBUTES.get(attr, attr.title())
                features.append(Item(f"{entity_name} {attr_name}", fill))
    stages.append(Stage("construction", "Construction", [[Group("Features", features)]]))

    # Transformation
    fill = PALETTE["transformation"][2][0]
    methods = tasks["transformation"]["used_methods"].vals
    stages.append(
        Stage(
            "transformation",
            "Transformation",
            [[Group("Graph Trans.", make_items(methods, "transformation", fill))]],
        )
    )

    # Featurization: text embeddings + pretrained encoders
    fill = PALETTE["featurization"][2][0]
    text_embed = make_items(cfg.FEATURIZATIONS_CFG.keys(), "featurization", fill)
    model_types = arg_values(cfg.FEATURIZATIONS_CFG["pretrained"]["model_type"])
    known = {k for _, keys in PRETRAINED_GROUPS for k in keys}
    pretrained = []
    for i, (title, keys) in enumerate(PRETRAINED_GROUPS):
        keys = [k for k in keys if k in model_types]
        if i == 0:
            keys += [k for k in model_types if k not in known]
        items = make_items(keys, "pretrained", PRETRAINED_FILL, PRETRAINED_STROKE)
        if items:
            pretrained.append(Group(title, items))
    stages.append(
        Stage("featurization", "Featurization", [[Group("Text Embed.", text_embed)], pretrained])
    )

    # Batching: one box per batching level
    fill = PALETTE["batching"][2][0]
    levels = [k for k, v in tasks["batching"].items() if isinstance(v, dict)]
    stages.append(
        Stage(
            "batching", "Batching", [[Group("Graph Batch.", make_items(levels, "batching", fill))]]
        )
    )

    # Training: encoder, decoder, objective + fine-tuning of pretrained encoders
    enc_fill, dec_fill, obj_fill = PALETTE["training"][2]
    finetune_modes = arg_values(tasks["training"]["pretrained"]["finetune_mode"])
    stages.append(
        Stage(
            "training",
            "Training",
            [
                [Group("Encoder", make_items(cfg.ENCODERS_CFG.keys(), "encoder", enc_fill))],
                [
                    Group("Decoder", make_items(cfg.DECODERS_CFG.keys(), "decoder", dec_fill)),
                    Group(
                        "Fine-tuning",
                        make_items(finetune_modes, "finetune", PRETRAINED_FILL, PRETRAINED_STROKE),
                    ),
                ],
                [Group("Objective", make_items(cfg.OBJECTIVES_CFG.keys(), "objective", obj_fill))],
            ],
        )
    )

    # Evaluation: thresholding methods + detection granularities
    thr_fill, det_fill = PALETTE["evaluation"][2]
    detections = [k for k, v in tasks["evaluation"].items() if isinstance(v, dict)]
    stages.append(
        Stage(
            "evaluation",
            "Evaluation",
            [
                [Group("Threshold", make_items(cfg.THRESHOLD_METHODS, "threshold", thr_fill))],
                [Group("Detection", make_items(detections, "detection", det_fill))],
            ],
        )
    )

    # Triage (optional stage: `none` is its default)
    fill = PALETTE["triage"][2][0]
    methods = list(tasks["triage"]["used_method"].vals) + ["none"]
    stages.append(
        Stage("triage", "Triage", [[Group("Tracing", make_items(methods, "triage", fill))]])
    )

    return stages


# ---------------------------------------------------------------------------
# SVG helpers
# ---------------------------------------------------------------------------

# Helvetica advance widths (1/1000 em), used to size the boxes around the labels.
_WIDTHS = {}
for chars, width in [
    ("ijl'", 222),
    (" !,./:;If|t", 278),
    ("()-r", 333),
    ('"', 355),
    ("*", 389),
    ("Jcksvxyz", 500),
    ("#$0123456789?L_abdeghnopqu", 556),
    ("+<=>", 584),
    ("FTZ", 611),
    ("&ABEKPSVXY", 667),
    ("CDHNRUw", 722),
    ("GOQ", 778),
    ("Mm", 833),
    ("%", 889),
    ("W", 944),
    ("@", 1015),
]:
    for c in chars:
        _WIDTHS[c] = width


def text_width(text, size, bold=False):
    width = sum(_WIDTHS.get(c, 600) for c in text) * size / 1000
    return width * (1.07 if bold else 1.0)


def n(value):
    """Compact number formatting, stable across runs."""
    return f"{value:.1f}".rstrip("0").rstrip(".")


def rect(x, y, w, h, fill, stroke, sw=1.5, rx=5):
    return (
        f'<rect x="{n(x)}" y="{n(y)}" width="{n(w)}" height="{n(h)}" rx="{n(rx)}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{n(sw)}"/>'
    )


def text(x, y, content, size, fill="#000000", bold=False, anchor="middle"):
    weight = ' font-weight="bold"' if bold else ""
    return (
        f'<text x="{n(x)}" y="{n(y)}" font-size="{n(size)}" fill="{fill}"{weight} '
        f'text-anchor="{anchor}" dominant-baseline="central">{escape(content)}</text>'
    )


def arrow(points, color=DARK, sw=2.5, dashed=False, head=12):
    """Polyline ending with an arrowhead drawn as a polygon (no SVG marker needed)."""
    (x1, y1), (x2, y2) = points[-2], points[-1]
    length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    ux, uy = (x2 - x1) / length, (y2 - y1) / length
    px, py = -uy, ux
    half = head * 0.42
    back = (x2 - ux * head, y2 - uy * head)
    notch = (x2 - ux * head * 0.7, y2 - uy * head * 0.7)
    line_pts = points[:-1] + [notch]
    dash = ' stroke-dasharray="9 6"' if dashed else ""
    d = "M" + " L".join(f"{n(x)} {n(y)}" for x, y in line_pts)
    poly = [
        (x2, y2),
        (back[0] + px * half, back[1] + py * half),
        notch,
        (back[0] - px * half, back[1] - py * half),
    ]
    return (
        f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{n(sw)}" '
        f'stroke-linejoin="round"{dash}/>'
        f'<polygon points="{" ".join(f"{n(x)},{n(y)}" for x, y in poly)}" fill="{color}"/>'
    )


# ---------------------------------------------------------------------------
# Layout and drawing
# ---------------------------------------------------------------------------


def layout(stages):
    x = MARGIN
    for stage in stages:
        stage.col_w = []
        heights = []
        for column in stage.columns:
            widest = MIN_COL_W
            height = 0
            for i, group in enumerate(column):
                widest = max(widest, text_width(group.title, TITLE_FONT, bold=True) * 1.12 + 8)
                for item in group.items:
                    # 12% slack: the viewer's sans-serif font may be wider than Helvetica
                    widest = max(widest, text_width(item.label, ITEM_FONT) * 1.12 + 22)
                height += GROUP_TITLE_H + (GROUP_GAP if i else 0)
                height += (len(group.items) - 1) * ITEM_PITCH + ITEM_H
            stage.col_w.append(round(widest))
            heights.append(height)
        inner = sum(stage.col_w) + COL_GAP * (len(stage.columns) - 1)
        header_min = text_width(stage.name, HEADER_FONT, bold=True) * 1.12 + 40
        if inner < header_min:  # widen the columns evenly to fit the stage name
            extra = (header_min - inner) / len(stage.col_w)
            stage.col_w = [round(w + extra) for w in stage.col_w]
            inner = sum(stage.col_w) + COL_GAP * (len(stage.columns) - 1)
        stage.x = x
        stage.w = inner + 2 * PANEL_PAD_X
        stage.h = HEADER_TOP + HEADER_H + 22 + max(heights) + PANEL_PAD_BOTTOM
        x += stage.w + PANEL_GAP
    return x - PANEL_GAP  # right edge of the last panel


def draw_stage(stage):
    panel_fill, header_fill, _ = PALETTE[stage.key]
    out = [rect(stage.x, PANEL_TOP, stage.w, stage.h, panel_fill, PANEL_STROKE, sw=2, rx=20)]
    header_y = PANEL_TOP + HEADER_TOP
    out.append(
        rect(
            stage.x + HEADER_INSET,
            header_y,
            stage.w - 2 * HEADER_INSET,
            HEADER_H,
            header_fill,
            DARK,
            sw=1.5,
            rx=7,
        )
    )
    out.append(
        text(stage.x + stage.w / 2, header_y + HEADER_H / 2, stage.name, HEADER_FONT, bold=True)
    )

    col_x = stage.x + PANEL_PAD_X
    for column, col_w in zip(stage.columns, stage.col_w):
        y = header_y + HEADER_H + 22
        for i, group in enumerate(column):
            y += GROUP_GAP if i else 0
            out.append(
                text(col_x + col_w / 2, y + 21, group.title, TITLE_FONT, fill="#666666", bold=True)
            )
            y += GROUP_TITLE_H
            for item in group.items:
                out.append(rect(col_x, y, col_w, ITEM_H, item.fill, item.stroke))
                out.append(text(col_x + col_w / 2, y + ITEM_H / 2, item.label, ITEM_FONT))
                y += ITEM_PITCH
            y += ITEM_H - ITEM_PITCH
        col_x += col_w + COL_GAP
    return out


def draw_datasets(cx):
    w, top, bottom, ry = 98, 44, 88, 13
    x0, x1 = cx - w / 2, cx + w / 2
    return [
        f'<path d="M{n(x0)} {top} V{bottom} A{w / 2} {ry} 0 0 0 {n(x1)} {bottom} V{top}" '
        f'fill="#E6E6E6" stroke="{DARK}" stroke-width="1.5"/>',
        f'<ellipse cx="{n(cx)}" cy="{top}" rx="{w / 2}" ry="{ry}" fill="#DFDFDF" stroke="{DARK}" stroke-width="1.5"/>',
        text(cx, 76, "Datasets", 16),
        arrow([(cx, bottom + ry), (cx, PANEL_TOP - 2)]),
    ]


def draw_yaml(cx):
    w, h, top, fold = 108, 84, 36, 20
    x0, x1 = cx - w / 2, cx + w / 2
    return [
        f'<path d="M{n(x0)} {top} H{n(x1 - fold)} L{n(x1)} {top + fold} V{top + h} H{n(x0)} Z" '
        f'fill="#ECECEC" stroke="#666666" stroke-width="1.5" stroke-linejoin="round"/>',
        f'<path d="M{n(x1 - fold)} {top} V{top + fold} H{n(x1)} Z" '
        f'fill="#D2D2D2" stroke="#666666" stroke-width="1.5" stroke-linejoin="round"/>',
        text(cx, top + 30, "YAML", 16),
        text(cx, top + 53, "Config", 16),
    ]


OUTPUTS_SPACING = 122  # distance between the centers of the two output icons


def draw_outputs(cx, from_y):
    """Metrics and Visualization icons, fed by the last stage."""
    w, h, top, spacing = 92, 56, 50, OUTPUTS_SPACING
    blue = "#46698C"
    out = []
    mx, vx = cx - spacing / 2, cx + spacing / 2

    # Metrics: a small table
    x0 = mx - w / 2
    out.append(text(mx, 26, "Metrics", 17))
    out.append(rect(x0, top, w, h, "#E3EDF7", blue, sw=1.5, rx=1))
    for i in (1, 2):
        y = top + i * h / 3
        out.append(f'<path d="M{n(x0)} {n(y)} H{n(x0 + w)}" stroke="#A0B0C0" stroke-width="1.5"/>')
    out.append(f'<path d="M{n(mx)} {top} V{top + h}" stroke="#A0B0C0" stroke-width="1.5"/>')

    # Visualization: a small bar chart
    x0 = vx - w / 2
    out.append(text(vx, 26, "Visualization", 17))
    out.append(rect(x0, top, w, h, "#EEF4FA", blue, sw=1.5, rx=1))
    base = top + h - 9
    for i, (bar_h, fill) in enumerate([(16, "#B1CAE3"), (28, "#92B5D8"), (12, "#73A0CD")]):
        out.append(
            f'<rect x="{n(x0 + 16 + i * 26)}" y="{n(base - bar_h)}" width="18" height="{bar_h}" fill="{fill}"/>'
        )
    out.append(
        f'<path d="M{n(x0 + 10)} {top + 8} V{base} H{n(x0 + w - 8)}" fill="none" stroke="{blue}" stroke-width="2.5"/>'
    )

    # Branching arrow from the last stage
    fork_y = (top + h + from_y) / 2 + 14
    out.append(arrow([(cx, from_y), (cx, fork_y), (mx, fork_y), (mx, top + h + 12)]))
    out.append(arrow([(cx, fork_y), (vx, fork_y), (vx, top + h + 12)]))
    return out


def draw_legend(x, y):
    out = [
        arrow([(x, y), (x + 74, y)]),
        text(x + 90, y, "Data flow", 14.5, anchor="start"),
        arrow([(x, y + 32), (x + 74, y + 32)], color=GRAY, sw=2, dashed=True),
        text(x + 90, y + 32, "Configuration", 14.5, fill="#808080", anchor="start"),
        rect(x + 20, y + 54, 54, 20, PRETRAINED_FILL, PRETRAINED_STROKE),
        text(x + 90, y + 64, "Pretrained encoder", 14.5, anchor="start"),
    ]
    return out


LEGEND_W, LEGEND_H = 230, 84


def render(cfg):
    stages = build_stages(cfg)
    right = layout(stages)
    body = []

    for stage in stages:
        body += draw_stage(stage)

    # Data flow between the stages
    arrow_y = PANEL_TOP + HEADER_TOP + HEADER_H / 2
    for prev, nxt in zip(stages, stages[1:]):
        body.append(arrow([(prev.x + prev.w + 3, arrow_y), (nxt.x - 3, arrow_y)]))

    # Top row: datasets, YAML config, outputs
    first, last = stages[0], stages[-1]
    body += draw_datasets(first.x + first.w / 2)
    yaml_x = (MARGIN + right) / 2
    for stage in stages:
        center = stage.x + stage.w / 2
        # land between the panel's center and the side facing the YAML file
        shift = max(-1, min(1, (yaml_x - center) / 300)) * stage.w * 0.25
        body.append(
            arrow(
                [(yaml_x, 124), (center + shift, PANEL_TOP - 3)],
                color=GRAY,
                sw=2,
                dashed=True,
                head=11,
            )
        )
    body += draw_yaml(yaml_x)
    outputs_x = last.x + last.w / 2
    body += draw_outputs(outputs_x, PANEL_TOP)
    # the "Visualization" label may stick out of the last panel: widen the canvas for it
    canvas_right = max(
        right, outputs_x + OUTPUTS_SPACING / 2 + text_width("Visualization", 17) * 0.56
    )

    # Legend: bottom right, below the panels it would overlap
    bottom = max(PANEL_TOP + s.h for s in stages)
    legend_x = right - LEGEND_W
    below = [PANEL_TOP + s.h for s in stages if s.x + s.w > legend_x - 20]
    legend_y = max(bottom - LEGEND_H, max(below) + 34)
    body += draw_legend(legend_x, legend_y)
    bottom = max(bottom, legend_y + LEGEND_H)

    # Brace and caption
    brace_y = bottom + 26
    x0, x1, xm, r = MARGIN - 6, right + 6, (MARGIN + right) / 2, 12
    body.append(
        f'<path d="M{n(x0)} {n(brace_y)} q0 {r} {r} {r} H{n(xm - r)} q{r} 0 {r} {r} '
        f'q0 -{r} {r} -{r} H{n(x1 - r)} q{r} 0 {r} -{r}" fill="none" stroke="#666666" stroke-width="1.5"/>'
    )
    body.append(text(xm, brace_y + 44, CAPTION, 16, fill="#333333"))

    width = round(canvas_right + MARGIN)
    height = round(brace_y + 70)
    header = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        "<!-- Generated by docs/scripts/gen_pipeline_figure.py from pidsmaker/config/config.py. "
        "Do not edit by hand. -->",
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="{FONT}">',
        "<title>PIDSMaker pipeline</title>",
        # opaque background, so that the figure stays readable on dark pages
        f'<rect width="{width}" height="{height}" rx="14" fill="#FFFFFF"/>',
    ]
    return "\n".join(header + body + ["</svg>"]) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "-o", "--output", type=Path, default=DEFAULT_OUTPUT, help="SVG file to write"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="do not write, exit with status 1 if the file is out of date",
    )
    args = parser.parse_args()

    svg = render(load_config())
    if _fallback_labels:
        print(
            "note: no display name in LABELS for: " + ", ".join(sorted(set(_fallback_labels))),
            file=sys.stderr,
        )

    if args.check:
        current = args.output.read_text() if args.output.exists() else ""
        if current != svg:
            print(f"{args.output} is out of date, run: python docs/scripts/gen_pipeline_figure.py")
            sys.exit(1)
        print(f"{args.output} is up to date")
        return

    args.output.write_text(svg)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
