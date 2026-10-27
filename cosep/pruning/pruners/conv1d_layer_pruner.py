"""Pruner of Conv1d layers."""
from cosep.pruning.pruners.layer_pruner import LayerPruner


class Conv1dLayerPruner(LayerPruner):
    """Pruner of Conv1d layers: a component is an output channel."""

    @property
    def layer_type(self) -> str:
        """Name of the kind of layer: 'Conv1d'."""
        return 'Conv1d'
