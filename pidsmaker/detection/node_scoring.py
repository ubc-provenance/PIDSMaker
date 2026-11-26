"""Node scores of node-level objectives, written at inference and thresholded at evaluation.

The inference loop (`training_methods.inference_loop.test_node_level`) writes one loss file per
batch, with the columns given by a `NodeScorer`, and the node evaluation
(`evaluation_methods.node_evaluation`) flags each node from its values in these files. The default
scorer writes the node losses, reduced to a score per node by `reduce_losses_to_score`. The
threshold methods of ThreaTrace, FLASH and MAGIC rank the nodes by scores of their own, computed
by their subclass in their module (see `node_scorer_factory` in `pidsmaker.factory`).
"""


class NodeScorer:
    # Column of the loss files holding the scores of the nodes (None: the node losses)
    score_column = None

    def __init__(self, cfg):
        self.cfg = cfg

    def prepare(self, model, train_data, split_names, device):
        """Called once before the inference of the splits `split_names` (e.g. to fit a
        detector on the training data)."""

    def columns(self, model, data, results, split, node, loss):
        """The columns of the loss file of the batch `data`.

        Args:
            model: the model, in eval mode.
            data: the batch, on the device.
            results: the output of `model(data, inference=True)`.
            split: the name of the split of the batch.
            node: the original node IDs of the batch (NumPy).
            loss: the node losses (NumPy, float64).
        """
        return {"node": node, "loss": loss}

    def node_prediction(self, values, thr):
        """The score of a node, whether it is flagged with the threshold `thr`, and the index of
        the time window in which it is detected (None: that of its max loss).

        Args:
            values: the values of the node in each column of the loss files of the test split,
                over its time windows, whose indices are in `tw`.
            thr: the threshold of the threshold method.
        """
        # Imported here: `evaluation_methods` imports the factory, which imports the scorers
        from pidsmaker.detection.evaluation_methods.evaluation_utils import reduce_losses_to_score

        pred_score = reduce_losses_to_score(
            values["loss"],
            self.cfg.evaluation.node_evaluation.threshold_method,
        )
        return pred_score, pred_score > thr, None
