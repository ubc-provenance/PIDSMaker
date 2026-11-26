"""Node scores written at inference by node-level objectives.

The inference loop (`training_methods.inference_loop.test_node_level`) writes one loss file per
batch, with the columns given by a `NodeScorer`. The default one writes the node losses, which
the evaluation thresholds. The threshold methods of ThreaTrace, FLASH and MAGIC rank the nodes by
scores of their own, computed by their subclass in their module (see `node_scorer_factory` in
`pidsmaker.factory`).
"""


class NodeScorer:
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
