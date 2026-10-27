"""Layer utilities: a layer's weight norms and the per-class statistics of its outputs."""
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from cosep.data_types.class_statistics import ClassStatistics
from cosep.utils.model_utils import get_model_layers


def collect_class_statistics(
    model: nn.Module, data_loader: DataLoader, layer_index: int
) -> ClassStatistics:
    """Per-class mean and variance of the activation summary of every component of a layer (paper, Eqs. 1-4).

    Each component gets one value per sample: a convolution channel the mean of its output map over all positions
    (Eq. 1), a linear neuron its output (Eq. 2). A forward hook computes these values and updates the statistics batch
    by batch, in float64, merging batches with the parallel algorithm of Chan et al. Nothing is stored per sample, so
    memory does not grow with the dataset. The hook reads each output as soon as the layer produces it, before later
    in-place operations (such as ReLU(inplace=True) or a residual `+=`) can change it, so the statistics are the same
    on CPU and GPU.

    Args:
        model: Neural network model.
        data_loader: Yields (inputs, labels) batches.
        layer_index: Index of the target layer.

    Returns:
        Statistics of the layer's components for each class in the data, of shape (classes, components).

    Raises:
        ValueError: If the layer's output is not a single tensor, if a linear layer produces outputs with more than one
            non-batch dimension (e.g. tokens), if the layer runs more than once per forward pass, if the labels do not
            match the batch, if its outputs contain NaN or Inf values, or if the data loader is empty.
    """
    layer = list(get_model_layers(model))[layer_index]
    device = next(model.parameters()).device
    # label -> (count, mean, sum of squared deviations from the mean), merged batch by batch
    running: dict[int, tuple[int, torch.Tensor, torch.Tensor]] = {}
    batch: dict[str, Any] = {}

    def hook_fn(module: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
        batch['calls'] += 1
        if batch['calls'] > 1:
            raise ValueError(f'Layer {layer_index} runs more than once per forward pass; this is not supported.')
        if not isinstance(output, torch.Tensor):
            raise ValueError(f'Layer {layer_index} returns {type(output).__name__}, not a tensor.')
        if isinstance(module, nn.Linear) and output.dim() != 2:
            raise ValueError(f'Linear layer {layer_index} produces outputs of shape {tuple(output.shape)}; only '
                             '(batch, features) outputs are supported, not token sequences.')
        x = output.detach().to('cpu', torch.float64)
        labels = batch['labels']
        if labels.shape != (x.shape[0],):
            raise ValueError(f'Got {tuple(labels.shape)} labels for a batch of {x.shape[0]} outputs '
                             f'at layer {layer_index}.')
        finite = torch.isfinite(x)
        if not finite.all():
            raise ValueError(f'Layer {layer_index} produced {int((~finite).sum())} NaN or Inf values on the '
                             'calibration data, so its separability cannot be computed.')
        if x.dim() > 2:
            x = x.flatten(2).mean(2)  # Eq. 1: each channel's output map averaged over its positions
        for label in torch.unique(labels).tolist():
            values = x[labels == label]
            n_b = values.shape[0]
            mean_b = values.mean(dim=0)
            m2_b = ((values - mean_b) ** 2).sum(dim=0)
            if label not in running:
                running[label] = (n_b, mean_b, m2_b)
                continue
            n_a, mean_a, m2_a = running[label]
            n = n_a + n_b
            delta = mean_b - mean_a
            running[label] = (n, mean_a + delta * (n_b / n), m2_a + m2_b + delta ** 2 * (n_a * n_b / n))

    hook = layer.register_forward_hook(hook_fn)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for inputs, labels in data_loader:
                batch.update(calls=0, labels=torch.as_tensor(labels).cpu())
                model(inputs.to(device))
    finally:
        hook.remove()
        model.train(was_training)

    if not running:
        raise ValueError(f'Layer {layer_index} produced no outputs: the data loader is empty.')
    classes = sorted(running)
    return ClassStatistics(
        classes=np.array(classes),
        counts=np.array([running[c][0] for c in classes]),
        means=torch.stack([running[c][1] for c in classes]).numpy(),
        variances=torch.stack([running[c][2] / running[c][0] for c in classes]).numpy(),
    )


def extract_weights(model: nn.Module, layer_index: int) -> np.ndarray:
    """Weight score of every component of a layer: the L2 norm of its parameters (paper, Eq. 14).

    The parameters of a component are its output filter (convolution) or its row of the weight matrix (linear layer).

    Args:
        model: Neural network model.
        layer_index: Index of target layer.

    Returns:
        One weight norm per component, as a NumPy array.

    Raises:
        TypeError: If layer type is unsupported.
    """
    layer = list(get_model_layers(model))[layer_index]

    if isinstance(layer, (nn.Conv1d, nn.Conv2d)):
        return torch.norm(
            layer.weight.view(layer.weight.size(0), -1), p=2, dim=1
        ).data.cpu().numpy()
    elif isinstance(layer, nn.Linear):
        return torch.norm(layer.weight, p=2, dim=1).data.cpu().numpy()
    else:
        raise TypeError(f'Layer {layer_index} is not a supported layer type')
