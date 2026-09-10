"""Real-time detection: scoring a live time window with a trained PIDS.

The detector loads the model a normal PIDSMaker run trained, then scores every
window the stream produces exactly as the offline pipeline scores a test window:

    window graph -> online featurization -> streaming batching -> model -> node scores

Scores are compared against the same threshold the offline evaluation would use,
derived from the validation losses recorded during training. The one unavoidable
difference is the aggregation horizon: offline, a node's score is its maximum over
the whole test set, which a detector cannot know before the fact - here a node is
scored within each window, as soon as that window closes.
"""

import os
import time
from collections import defaultdict
from typing import Optional

import torch

from pidsmaker.detection.evaluation_methods.evaluation_utils import (
    get_threshold,
    reduce_losses_to_score,
)
from pidsmaker.detection.training_methods.inference_loop import build_node_records
from pidsmaker.factory import build_model
from pidsmaker.streaming.batching import StreamingBatcher
from pidsmaker.streaming.featurizer import build_online_featurizer
from pidsmaker.streaming.sinks.alerts import Alert, WindowReport
from pidsmaker.utils.utils import get_device, listdir_sorted, log

# Threshold methods whose per-node score is not the loss, and which additionally
# require the model to have predicted the node's own type.
CONFIDENCE_METHODS = {"threatrace": "threatrace_score", "flash": "flash_score"}


def resolve_val_losses_dir(cfg) -> Optional[str]:
    """Finds the validation losses of the most recent epoch of the trained run.

    Args:
        cfg: Full pipeline config.

    Returns:
        str: Path to the epoch directory holding the validation loss CSVs, or None.
    """
    val_dir = os.path.join(cfg.training._edge_losses_dir, "val")
    if not os.path.isdir(val_dir):
        return None
    epochs = listdir_sorted(val_dir)
    return os.path.join(val_dir, epochs[-1]) if epochs else None


def resolve_threshold(cfg, override: Optional[float] = None) -> float:
    """Determines the score above which a node is reported.

    Args:
        cfg: Full pipeline config.
        override: Explicit threshold, bypassing the trained run's validation losses.

    Returns:
        float: The detection threshold.
    """
    method = cfg.evaluation.node_evaluation.threshold_method
    if override is not None:
        log(f"Using explicit detection threshold {override} (method: {method})")
        return override

    if method == "magic":
        raise NotImplementedError(
            "The `magic` threshold is computed from the test set's own embedding distances, "
            "which a live stream has no equivalent of. Pass `--alert_threshold` to run this "
            "system in real time."
        )

    val_losses_dir = resolve_val_losses_dir(cfg)
    if val_losses_dir is None:
        raise FileNotFoundError(
            f"No validation losses found under {cfg.training._edge_losses_dir}/val. Train the "
            "system first (`python pidsmaker/main.py <system> <dataset>`), or pass "
            "`--alert_threshold` explicitly."
        )

    threshold = get_threshold(val_losses_dir, method)
    log(f"Detection threshold: {threshold:.4f} (method: {method}, from {val_losses_dir})")
    return threshold


def load_trained_weights(model, checkpoint_dir: str):
    """Loads the trained weights, leaving every node-indexed buffer freshly reset.

    A checkpoint's TGN memory is indexed by the node ids of the *training* dataset;
    those ids mean nothing in a live stream, and its memory tables are sized for
    that dataset rather than for the stream's node capacity. Such buffers are
    therefore skipped and start empty, which is also the semantics the offline
    inference loop uses when it resets the memory before evaluating.

    Args:
        model: The model to load into.
        checkpoint_dir: Directory written by `save_model()`.
    """
    state_dict_path = os.path.join(checkpoint_dir, "state_dict.pkl")
    if not os.path.isfile(state_dict_path):
        raise FileNotFoundError(
            f"No trained model at {state_dict_path}. Train the system with `--save_model` "
            "(same arguments otherwise): it writes `training/<hash>/trained_models/model_best`."
        )

    checkpoint = torch.load(state_dict_path, map_location="cpu")
    model_state = model.state_dict()

    compatible, skipped = {}, []
    for key, value in checkpoint.items():
        if key in model_state and model_state[key].shape == value.shape:
            compatible[key] = value
        else:
            skipped.append(key)

    model.load_state_dict(compatible, strict=False)
    model.reset_state()

    log(f"Loaded {len(compatible)} tensors from {state_dict_path}")
    if skipped:
        log(f"Reset {len(skipped)} node-indexed tensors (e.g. TGN memory): {skipped[:5]}")
    return model


class StreamingDetector:
    """Scores live time windows with a trained PIDS.

    Args:
        cfg: Full pipeline config of the trained run. It must be the *same* config
            the model was trained with: the featurization model, the feature
            dimensions and the batching strategy all have to match.
        device: Device to run on. Defaults to the config's device.
        threshold: Explicit detection threshold, overriding the one derived from
            the trained run's validation losses.
        checkpoint_dir: Directory holding the trained model. Defaults to
            `training/<hash>/trained_models/model_best`.
        max_nodes: Node capacity of the batching tables.
        max_events: Capacity of the TGN event buffer.
    """

    def __init__(
        self,
        cfg,
        device=None,
        threshold: Optional[float] = None,
        checkpoint_dir: Optional[str] = None,
        max_nodes: int = 2_000_000,
        max_events: int = 2_000_000,
        viz_collector=None,
    ):
        self.cfg = cfg
        self.device = device or get_device(cfg)
        # Optional `StreamingVizCollector`; when set, every scored window is
        # recorded for the web embedding viewer instead of only alerts.
        self.viz_collector = viz_collector
        self.max_nodes = max_nodes
        self.checkpoint_dir = checkpoint_dir or os.path.join(
            cfg.training._trained_models_dir, "model_best"
        )

        self.featurizer = build_online_featurizer(cfg)
        self.batcher = StreamingBatcher(
            cfg, device=self.device, max_nodes=max_nodes, max_events=max_events
        )
        self.threshold = resolve_threshold(cfg, threshold)
        self.threshold_method = cfg.evaluation.node_evaluation.threshold_method
        self.is_node_level = cfg._is_node_level
        self.use_dst_node_loss = cfg.evaluation.node_evaluation.use_dst_node_loss

        self.model = None
        self.num_windows = 0
        self.num_alerts = 0

    def _ensure_model(self, sample):
        if self.model is not None:
            return
        log("Building the model from the first time window...")
        self.model = build_model(
            data_sample=sample, device=self.device, cfg=self.cfg, max_node_num=self.max_nodes
        )
        load_trained_weights(self.model, self.checkpoint_dir)
        self.model.eval()
        self.model.to_device(self.device)

    @torch.no_grad()
    def process_window(self, window, indexid2msg, index_to_key=None) -> WindowReport:
        """Scores one time window and returns what was found in it.

        Args:
            window: A `TimeWindow` from the streaming graph builder.
            indexid2msg: Mapping of node index to `[node_type, label]`, used to
                describe the nodes an alert points at.
            index_to_key: Mapping of node index to the producer's own identifier,
                so an alert can be traced back to the capture.

        Returns:
            WindowReport: The window's alerts and summary statistics.
        """
        started = time.time()

        indexid2vec = self.featurizer.embed_window(window.graph, indexid2msg)
        batches = self.batcher.build(window.graph, indexid2vec)
        if not batches:
            return WindowReport(
                window=window.interval,
                window_start=window.start_ns,
                window_end=window.end_ns,
                num_nodes=window.graph.number_of_nodes(),
                num_edges=window.graph.number_of_edges(),
                num_events=window.num_events,
                threshold=self.threshold,
                max_score=0.0,
                inference_time=time.time() - started,
            )

        self._ensure_model(batches[0])

        node_to_values = defaultdict(lambda: defaultdict(list))
        for batch in batches:
            batch = batch.to(self.device)
            results = self.model(batch, inference=True, validation=False)
            if self.is_node_level:
                self._collect_node_level(batch, results, node_to_values)
            else:
                self._collect_edge_level(batch, results, node_to_values)

        alerts, max_score, node_scores, alert_node_ids = self._score_nodes(
            node_to_values, window, indexid2msg, index_to_key or {}
        )

        if self.viz_collector is not None:
            self.viz_collector.add_window(
                window=window,
                indexid2vec=indexid2vec,
                node_scores=node_scores,
                alert_node_ids=alert_node_ids,
                indexid2msg=indexid2msg,
                index_to_key=index_to_key or {},
            )

        self.num_windows += 1
        self.num_alerts += len(alerts)
        return WindowReport(
            window=window.interval,
            window_start=window.start_ns,
            window_end=window.end_ns,
            num_nodes=window.graph.number_of_nodes(),
            num_edges=window.graph.number_of_edges(),
            num_events=window.num_events,
            threshold=self.threshold,
            max_score=max_score,
            alerts=alerts,
            inference_time=time.time() - started,
        )

    def _collect_node_level(self, batch, results, node_to_values):
        n_id = getattr(batch, "original_n_id_tgn", getattr(batch, "original_n_id"))
        records = build_node_records(results, batch, n_id, results["loss"], self.threshold_method)
        for record in records:
            values = node_to_values[record["node"]]
            values["loss"].append(record["loss"])
            for key in ("threatrace_score", "flash_score", "correct_pred"):
                if key in record:
                    values[key].append(record[key])

    def _collect_edge_level(self, batch, results, node_to_values):
        # Edge-level systems score events; a node's score is built from the events
        # it takes part in, exactly as `node_evaluation` does offline.
        losses = results["loss"].detach().cpu().numpy()
        edge_index = batch.original_edge_index.detach().cpu().numpy()
        for i, loss in enumerate(losses):
            node_to_values[int(edge_index[0, i])]["loss"].append(float(loss))
            if self.use_dst_node_loss:
                node_to_values[int(edge_index[1, i])]["loss"].append(float(loss))

    def _score_nodes(self, node_to_values, window, indexid2msg, index_to_key):
        """Reduces per-node values to a score and keeps the ones above threshold.

        Returns the alerts (above threshold), the window's max score, and — for
        visualization and analysis — the score of *every* node scored this window
        and the set of node ids that alerted. The full map is what the viz
        collector needs: an alert file only records the tip of the distribution,
        but the viewer colours every node by its score.
        """
        score_key = CONFIDENCE_METHODS.get(self.threshold_method)
        alerts, max_score = [], 0.0
        node_scores: dict[int, float] = {}
        alert_node_ids: set[int] = set()

        for node, values in node_to_values.items():
            if score_key is not None and values.get(score_key):
                # ThreaTrace and Flash only report a node whose type the model got
                # right, on top of the confidence being above threshold.
                scores = values[score_key]
                correct = values.get("correct_pred") or [1] * len(scores)
                score = max(scores)
                is_alert = any(s > self.threshold and c for s, c in zip(scores, correct))
            else:
                score = float(reduce_losses_to_score(values["loss"], self.threshold_method))
                is_alert = score > self.threshold

            max_score = max(max_score, float(score))
            node_scores[int(node)] = float(score)
            if not is_alert:
                continue
            alert_node_ids.add(int(node))

            node_type, label = indexid2msg.get(str(node), ["unknown", ""])
            alerts.append(
                Alert(
                    node=int(node),
                    node_type=node_type,
                    label=label,
                    score=float(score),
                    threshold=self.threshold,
                    window_start=window.start_ns,
                    window_end=window.end_ns,
                    window=window.interval,
                    window_events=window.num_events,
                    source_key=index_to_key.get(str(node), ""),
                )
            )

        alerts.sort(key=lambda alert: alert.score, reverse=True)
        return alerts, max_score, node_scores, alert_node_ids
