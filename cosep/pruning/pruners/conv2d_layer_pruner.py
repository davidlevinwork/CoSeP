"""Pruner of Conv2d layers."""
from cosep.pruning.pruners.layer_pruner import LayerPruner


class Conv2dLayerPruner(LayerPruner):
    """Pruner of Conv2d layers: a component is an output channel."""

    @property
    def layer_type(self) -> str:
        """Name of the kind of layer: 'Conv2d'."""
        return 'Conv2d'
