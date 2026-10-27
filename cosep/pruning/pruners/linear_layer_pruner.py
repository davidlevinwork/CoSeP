"""Pruner of Linear layers."""
from cosep.pruning.pruners.layer_pruner import LayerPruner


class LinearLayerPruner(LayerPruner):
    """Pruner of Linear layers: a component is an output neuron."""

    @property
    def layer_type(self) -> str:
        """Name of the kind of layer: 'Linear'."""
        return 'Linear'
