"""Physical removal of pruned components, and the rule for which layers may be pruned on their own.

CoSeP removes every component it does not keep (paper, Eq. 16). Removing an output channel of one layer also removes
everything tied to it: the matching BatchNorm entries and the matching input weights of every layer that reads the
channel (the next convolution, the first linear layer after a flatten, all later layers of a dense block). The
dependency graph of Torch-Pruning (Fang et al., "DepGraph", CVPR 2023) finds these ties by tracing the model once.

Some layers cannot be pruned on their own, because their output channels are shared with other layers:

* Layers whose outputs are combined element-wise, as in a residual sum y = F(x) + x, must keep identical channels.
  CoSeP selects channels separately for each layer, so these layers are kept whole.
* A depthwise convolution filters each input channel separately, so it keeps exactly the channels of the layer that
  feeds it. The two are pruned as one unit, with the selection made on the feeding layer.
"""
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch_pruning as tp

PRUNABLE_TYPES = (nn.Conv1d, nn.Conv2d, nn.Linear)
_LAYER_TYPES = (tp.ops.TORCH_CONV, tp.ops.TORCH_LINEAR)


def num_components(layer: nn.Module) -> int:
    """Number of structured components of a layer: output channels or output neurons.

    Args:
        layer: A convolution, linear or attention layer.

    Returns:
        The number of output channels (convolution), output features (linear) or the embedding size (attention).

    Raises:
        TypeError: For any other kind of layer.
    """
    if isinstance(layer, nn.Linear):
        return layer.out_features
    if isinstance(layer, nn.MultiheadAttention):
        return layer.embed_dim
    if isinstance(layer, tp.ops.TORCH_CONV):
        return layer.out_channels
    raise TypeError(f'{type(layer).__name__} layers have no structured components.')


def is_depthwise(layer: nn.Module) -> bool:
    """True for a depthwise convolution: one filter per input channel."""
    return (isinstance(layer, tp.ops.TORCH_CONV) and layer.groups > 1
            and layer.groups == layer.in_channels == layer.out_channels)


def _out_channel_fn(layer: nn.Module) -> Callable[..., Any]:
    """Torch-Pruning's function that removes output channels (or neurons) of this kind of layer."""
    return tp.prune_linear_out_channels if isinstance(layer, nn.Linear) else tp.prune_conv_out_channels


@dataclass
class LayerRule:
    """How a layer may be pruned.

    Attributes:
        prunable: True if CoSeP selects this layer's components itself.
        reason: Why the layer is not pruned on its own (empty when prunable).
        pruned_with: Depthwise convolutions whose channels are removed together with this layer's channels.
    """
    prunable: bool
    reason: str = ''
    pruned_with: list[str] = field(default_factory=list)


class ModelStructure:
    """Dependency graph of a model: decides which layers may be pruned and removes channels.

    The graph is traced once from an example input and stays valid while channels are removed.
    """

    def __init__(self, model: nn.Module, example_inputs: torch.Tensor) -> None:
        """Trace the model.

        Args:
            model: Model to prune. Channels are removed from it in place.
            example_inputs: One input batch (a single sample is enough), on the model's device.
        """
        self.model = model
        was_training = model.training
        try:
            self.graph = tp.DependencyGraph().build_dependency(
                model, example_inputs=example_inputs, verbose=False
            )
        finally:
            model.train(was_training)  # tracing switches the model to eval mode
        self.names: dict[nn.Module, str] = {m: n for n, m in model.named_modules()}
        self.output_layers = _layers_producing_output(model, example_inputs)

    def name_of(self, module: nn.Module) -> str:
        """Qualified name of a module in the model."""
        return self.names.get(module, type(module).__name__)

    def rule(self, layer: nn.Module) -> LayerRule:
        """Decide whether a layer's components can be selected and removed on their own.

        Args:
            layer: A layer of the model.

        Returns:
            The rule for this layer.
        """
        if not isinstance(layer, PRUNABLE_TYPES):
            return LayerRule(False, f'{type(layer).__name__} layers are not supported')
        if layer not in self.graph.module2node:
            return LayerRule(False, 'not used in the forward pass')
        if layer in self.output_layers:
            return LayerRule(False, 'model output')
        if is_depthwise(layer):
            return LayerRule(False, 'depthwise convolution; it keeps the channels of the layer that feeds it')

        group = self.graph.get_pruning_group(layer, _out_channel_fn(layer), idxs=[0])
        depthwise: list[str] = []
        shared: list[str] = []
        shared_parameter = False
        for dep, _ in group:
            node, target = dep.target, dep.target.module
            if node.type == tp.ops.OPTYPE.OUTPUT:
                return LayerRule(False, 'model output')
            if node.type == tp.ops.OPTYPE.PARAMETER:
                shared_parameter = True
                continue
            if target is layer or not isinstance(target, _LAYER_TYPES):
                continue
            if not self.graph.is_out_channel_pruning_fn(dep.handler):
                continue  # the target only reads the channels (its inputs shrink); that is not sharing
            (depthwise if is_depthwise(target) else shared).append(self.name_of(target))

        if shared:
            names = sorted(set(shared))
            return LayerRule(False, f'shares its output channels with {len(names)} other layer(s), '
                                    f'e.g. through a residual sum: {", ".join(names[:3])}'
                                    + (', ...' if len(names) > 3 else ''))
        if shared_parameter:
            return LayerRule(False, 'shares its output channels with a model parameter')
        return LayerRule(True, pruned_with=sorted(set(depthwise)))

    def remove(self, layer: nn.Module, keep: Iterable[int]) -> list[str]:
        """Physically remove every output channel (or neuron) of a layer that is not kept.

        Everything tied to the removed channels goes with them: the matching BatchNorm entries, the matching input
        weights of every layer that reads them, and the channels of a depthwise convolution fed by the layer.

        Args:
            layer: Layer to prune.
            keep: Indices of the components to keep, in the layer's current numbering.

        Returns:
            Names of all modules that were resized.

        Raises:
            ValueError: If no component is kept or an index is out of range.
            RuntimeError: If the removal would leave a layer without channels.
        """
        n = num_components(layer)
        kept = sorted({int(i) for i in keep})
        if not kept:
            raise ValueError(f'At least one component of {self.name_of(layer)} must be kept.')
        if kept[0] < 0 or kept[-1] >= n:
            raise ValueError(f'Kept indices of {self.name_of(layer)} must lie in [0, {n}); got {kept[0]}..{kept[-1]}.')
        kept_set = set(kept)
        drop = [i for i in range(n) if i not in kept_set]
        if not drop:
            return []
        group = self.graph.get_pruning_group(layer, _out_channel_fn(layer), idxs=drop)
        if not self.graph.check_pruning_group(group):
            raise RuntimeError(f'Removing {len(drop)} of {n} components of {self.name_of(layer)} '
                               'would leave a layer without channels.')
        resized = sorted({self.name_of(dep.target.module) for dep, _ in group
                          if isinstance(dep.target.module, nn.Module) and dep.target.module in self.names})
        group.prune()
        return resized


def _layers_producing_output(model: nn.Module, example_inputs: torch.Tensor) -> set[nn.Module]:
    """Layers whose output tensor is the model output (possibly reshaped), i.e. the classifier."""
    storage: dict[nn.Module, int] = {}

    def record(module: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
        if isinstance(output, torch.Tensor):
            storage[module] = output.data_ptr()

    layers = [m for m in model.modules() if isinstance(m, _LAYER_TYPES)]
    hooks = [m.register_forward_hook(record) for m in layers]
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            output = model(example_inputs)
    finally:
        for h in hooks:
            h.remove()
        model.train(was_training)
    if not isinstance(output, torch.Tensor):
        return set()
    return {m for m, ptr in storage.items() if ptr == output.data_ptr()}


def count_macs(model: nn.Module, example_inputs: torch.Tensor) -> dict[nn.Module, int]:
    """Multiply-accumulate operations (MACs) of every convolution and linear layer, for one input.

    Bias additions, BatchNorm, activations and pooling are not counted.

    Args:
        model: Model to measure.
        example_inputs: An input batch; the counts are divided by its batch size.

    Returns:
        MACs per layer. Layers that are not executed are absent.
    """
    macs: dict[nn.Module, int] = {}

    def record(module: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
        out = output[0] if isinstance(output, tuple) else output
        if isinstance(module, nn.Linear):
            per_output = module.in_features
        elif isinstance(module, tp.ops.TORCH_CONV):
            per_output = (module.in_channels // module.groups) * int(np.prod(module.kernel_size))
        else:
            return
        macs[module] = macs.get(module, 0) + out.numel() // out.shape[0] * per_output

    layers = [m for m in model.modules()
              if isinstance(m, nn.Linear) or (isinstance(m, tp.ops.TORCH_CONV) and not m.transposed)]
    hooks = [m.register_forward_hook(record) for m in layers]
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            model(example_inputs)
    finally:
        for h in hooks:
            h.remove()
        model.train(was_training)
    return macs


def check_plan(model: nn.Module, plan: dict[str, Any]) -> None:
    """Check that every layer of the model has the number of components recorded in a pruning plan.

    Args:
        model: Pruned model.
        plan: Pruning plan (see Pruner).

    Raises:
        RuntimeError: If a layer's size differs from the plan.
    """
    modules = dict(model.named_modules())
    for entry in plan['layers']:
        layer = modules.get(entry['layer'])
        if layer is None:
            raise RuntimeError(f"Layer {entry['layer']} of the pruning plan is not in the model.")
        if num_components(layer) != entry['components_after']:
            raise RuntimeError(f"Layer {entry['layer']} has {num_components(layer)} components; the pruning plan "
                               f"expects {entry['components_after']}.")


def apply_pruning_plan(model: nn.Module, plan: dict[str, Any], example_inputs: torch.Tensor) -> nn.Module:
    """Rebuild a pruned architecture from the original one.

    Removes, layer by layer in the plan's order, every component not listed under 'kept' for the layers whose status
    is 'pruned' (depthwise convolutions follow the layer that feeds them). Load the pruned weights afterwards.

    Args:
        model: The original (unpruned) model; modified in place.
        plan: Contents of pruning_plan.json.
        example_inputs: One input batch on the model's device.

    Returns:
        The model, with the pruned architecture.

    Raises:
        RuntimeError: If the rebuilt model does not have the layer sizes recorded in the plan.
    """
    structure = ModelStructure(model, example_inputs)
    modules = dict(model.named_modules())
    for entry in plan['layers']:
        if entry['status'] == 'pruned':
            structure.remove(modules[entry['layer']], entry['kept'])
    check_plan(model, plan)
    return model
