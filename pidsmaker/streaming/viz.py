"""Records streaming detection results for the interactive web viewer.

`stream_detect.py` normally emits only *alerts* — the nodes whose score crossed
the threshold. That is the right thing for an alert feed, but it throws away the
rest of the distribution, which is exactly what you want when analysing or
tuning: the near-misses, the shape of "normal", how a score moves window to
window. This collector keeps all of it and writes it in the two forms the rest
of PIDSMaker already understands:

- **The web embedding viewer's artifacts** (`embedding_viz_<dataset>_word2vec.html`
  plus its `_points.json` / `_adj.json`), under a run directory the viewer
  discovers on its own. Every scored node becomes one 3D point per time window,
  coloured by its anomaly score, so `pidsmaker.vizgen.web.viz_server` plays the
  run back with the same temporal scrub, per-node inspection and graph overlays
  it gives an offline run.
- **A flat `stream_scores.jsonl`** — one line per node per window
  (`window`, `node`, `node_type`, `label`, `score`, `is_alert`, `source_key`) —
  the raw record the viewer is built from, for pandas/SQL analysis.

The heavy vizgen imports (torch, UMAP) are deferred to `finalize()` so a run
without `--emit_viz` never pays for them.
"""

import contextlib
import io
import json
import os
import time
from typing import Dict, Optional, Set

import numpy as np
import yaml

from pidsmaker.utils.utils import log


class StreamingVizCollector:
    """Accumulates per-window, per-node embeddings + scores, then builds the viewer artifacts.

    Args:
        cfg: The run config (for dataset name, featurization method, model name).
        artifact_dir: The artifacts root the web viewer scans (e.g. `/home/artifacts`).
        method: Dimensionality-reduction method passed to `reduce_to_3d` ("umap"/"tsne"/"pca").
        device: Torch device string for the reduction (falls back to CPU on its own).
        max_points: Safety cap on total (node, window) snapshots held in memory, so a
            long capture cannot exhaust RAM. Once reached, further nodes are skipped
            and a warning is logged at `finalize()`.
    """

    def __init__(
        self,
        cfg,
        artifact_dir: str,
        method: str = "umap",
        device: Optional[str] = None,
        max_points: int = 300_000,
    ):
        self.cfg = cfg
        self.dataset = cfg.dataset.name
        self.method = method
        self.device = device
        self.max_points = max_points

        # Imported here (not at module load) but before any window is scored, so a
        # mistyped vizgen API fails fast at startup rather than after a long run.
        from pidsmaker.vizgen.embed_exporter import ExtractionResult, TemporalEmbedding

        self._TemporalEmbedding = TemporalEmbedding
        self._ExtractionResult = ExtractionResult

        self.embeddings = []  # list[TemporalEmbedding], one per (node, window)
        self.edges: Set[tuple] = set()  # {(src_idx, dst_idx, tw_idx, relation_name)}
        self.node_meta: Dict[int, dict] = {}  # node_id -> {"path","type","cmd"}
        self.tw_idx = 0
        self.capped = False

        # A run directory the viewer discovers by globbing `*/evaluation/*/*/viz/*_points.json`.
        run_id = f"stream_{time.strftime('%Y%m%d_%H%M%S')}"
        self.run_dir = os.path.join(artifact_dir, "detection", "evaluation", run_id, self.dataset)
        self.viz_dir = os.path.join(self.run_dir, "viz")
        os.makedirs(self.viz_dir, exist_ok=True)

        self.scores_path = os.path.join(self.run_dir, "stream_scores.jsonl")
        self._scores_fh = open(self.scores_path, "w", encoding="utf-8")
        log(f"Recording streaming predictions to {self.run_dir}")

    def add_window(
        self,
        window,
        indexid2vec,
        node_scores: Dict[int, float],
        alert_node_ids: Set[int],
        indexid2msg,
        index_to_key,
    ):
        """Records one scored window: every node's embedding, score and metadata."""
        # No embeddings means a type-only featurizer; there is nothing to lay out in
        # 3D, so the viewer cannot show this run. Say so once and stop collecting.
        if indexid2vec is None:
            if not self.capped:
                log(
                    "Viz: this featurizer produces no node embeddings (types only), "
                    "so there is nothing to visualize; only stream_scores.jsonl will be written."
                )
                self.capped = True
            self._record_scores_only(window, node_scores, alert_node_ids, indexid2msg, index_to_key)
            self.tw_idx += 1
            return

        tw_label = window.interval
        for key, vec in indexid2vec.items():
            nid = int(key)
            score = float(node_scores.get(nid, 0.0))
            node_type, label = indexid2msg.get(str(nid), ["unknown", ""])
            is_alert = nid in alert_node_ids

            if len(self.embeddings) < self.max_points:
                self.embeddings.append(
                    self._TemporalEmbedding(
                        node_id=nid,
                        time_window_idx=self.tw_idx,
                        time_window_label=tw_label,
                        embedding=np.asarray(vec, dtype=np.float32),
                        label=0,  # streaming has no ground-truth attack labels
                        detection_status=1 if is_alert else 0,
                        anomaly_score=score,
                    )
                )
                self.node_meta[nid] = {"path": label or "", "type": node_type, "cmd": label or ""}
            else:
                self.capped = True

            self._write_score(nid, tw_label, node_type, label, score, is_alert, index_to_key)

        for u, v, _key, attr in window.graph.edges(keys=True, data=True):
            self.edges.add((int(u), int(v), self.tw_idx, attr.get("label", "")))
        self.tw_idx += 1

    def _record_scores_only(self, window, node_scores, alert_node_ids, indexid2msg, index_to_key):
        for nid, score in node_scores.items():
            node_type, label = indexid2msg.get(str(nid), ["unknown", ""])
            self._write_score(
                int(nid),
                window.interval,
                node_type,
                label,
                float(score),
                nid in alert_node_ids,
                index_to_key,
            )

    def _write_score(self, nid, window, node_type, label, score, is_alert, index_to_key):
        self._scores_fh.write(
            json.dumps(
                {
                    "tw_idx": self.tw_idx,
                    "window": window,
                    "node": nid,
                    "node_type": node_type,
                    "label": label,
                    "score": score,
                    "is_alert": bool(is_alert),
                    "source_key": index_to_key.get(str(nid), ""),
                }
            )
            + "\n"
        )

    def finalize(self, title: Optional[str] = None) -> Optional[str]:
        """Runs dimensionality reduction and writes the viewer artifacts.

        Returns the path of the generated HTML, or None when there was nothing to
        lay out (no windows, or a type-only featurizer).
        """
        self._scores_fh.close()
        log(f"Streaming scores written to {self.scores_path} ({self.tw_idx} windows).")

        if not self.embeddings:
            log("Viz: no node embeddings collected, so no interactive viewer was produced.")
            return None

        # Deferred: these pull torch and UMAP, which a non-viz run must not pay for.
        from pidsmaker.vizgen.dimensionality_reduction import reduce_to_3d
        from pidsmaker.vizgen.html_builder import build_html

        if self.capped:
            log(
                f"Viz: capped at {self.max_points} node-snapshots to bound memory; "
                "the viewer shows the earliest windows. Raise --viz_max_points to keep more."
            )

        log(f"Viz: reducing {len(self.embeddings)} embeddings to 3D via {self.method}...")
        result = self._ExtractionResult(embeddings=self.embeddings, edges=self.edges)
        out_path = os.path.join(self.viz_dir, f"embedding_viz_{self.dataset}_word2vec.html")

        # The reduction and the builder narrate every step, which suits the offline
        # pipeline and is noise at the end of a detector run. Keep their output and
        # surface only what needs attention - or all of it, if they failed.
        captured = io.StringIO()
        try:
            with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
                points = reduce_to_3d(result, method=self.method, device=self.device)
                build_html(
                    points=points,
                    edges=list(self.edges),
                    node_metadata=self.node_meta,
                    title=title or f"Streaming detection — {self.dataset}",
                    out_path=out_path,
                )
        except Exception:
            print(captured.getvalue(), end="")
            raise
        for line in captured.getvalue().splitlines():
            if any(k in line.lower() for k in ("warn", "error", "skipped")):
                log(f"Viz: {line.strip()}")
        self._write_run_config()
        log(f"Viz: interactive viewer written to {out_path}")
        run_name = os.path.basename(os.path.dirname(self.run_dir))
        log(
            "Open it with the web viewer:\n"
            "  python -m pidsmaker.vizgen.web.viz_server\n"
            f"  then pick the '{run_name}' run for {self.dataset}."
        )
        return out_path

    def _write_run_config(self):
        """Minimal run_config.yml so the viewer can label the run."""
        run_config = {
            "_model": str(getattr(self.cfg, "_model", "") or ""),
            "featurization": {"used_method": self.cfg.featurization.used_method},
            "_source": "streaming",
        }
        with open(os.path.join(self.run_dir, "run_config.yml"), "w", encoding="utf-8") as f:
            yaml.safe_dump(run_config, f)
