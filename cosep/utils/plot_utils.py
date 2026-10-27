"""Plots of a run: each pruned layer's components on the t-SNE map, and its weights before and after pruning.

Plots are drawn only in verbose mode (cosep.config.verbose) and saved under plots/ in the run's output folder.
"""
import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from loguru import logger
from matplotlib.figure import Figure
from scipy.stats import gaussian_kde

import cosep.config as config
from cosep.data_types.decorators import verbose_decorator


def _layer_folder(is_linear: bool, layer_idx: int) -> str:
    """Name of a layer's plot folder: 'Linear layer #<index>' or 'Conv layer #<index>'."""
    return f"{'Linear' if is_linear else 'Conv'} layer #{layer_idx}"


@verbose_decorator
def plot_feature_space(
    data: np.ndarray, layer_idx: int, layer_type: str, profiles: Optional[np.ndarray] = None
) -> None:
    """Plot a layer's components on the 2-D t-SNE map of their separability profiles.

    Saved as 'Separability space.png' in the layer's plot folder.

    Args:
        data: 2-D map of the components, one row per component.
        layer_idx: Index of the layer.
        layer_type: Kind of layer: 'Conv1d', 'Conv2d' or 'Linear'.
        profiles: Separability profiles of the components, one row each. If given, points are coloured by their
            separability power: the mean JM distance of the profile over all class pairs.
    """
    sns.set_theme(style='whitegrid')

    fig, ax = plt.subplots(figsize=(10, 8))
    colour = {} if profiles is None else {'c': profiles.mean(axis=1), 'cmap': 'viridis'}
    scatter = ax.scatter(
        data[:, 0], data[:, 1], alpha=0.7,
        edgecolors='w', linewidth=0.5, **colour,
    )
    if profiles is not None:
        fig.colorbar(scatter, ax=ax, label='Separability power (mean JM distance)')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    is_linear = layer_type == 'Linear'
    ax.set_title(f"Layer {layer_idx} ({layer_type}): its {'neurons' if is_linear else 'channels'} on the t-SNE map "
                 "of their separability profiles")
    fig.tight_layout()
    save_plot(fig, _layer_folder(is_linear, layer_idx), 'Separability space')


@verbose_decorator
def plot_weight_distribution(
    layer: nn.Module, layer_idx: int, before_prune: bool
) -> None:
    """Plot the distribution of a layer's weights: a histogram with a kernel density estimate.

    Saved as 'Weight Distribution Before Pruning.png' or 'Weight Distribution After Pruning.png' in the layer's
    plot folder. Nothing is drawn when the weights are all NaN or all equal.

    Args:
        layer: Layer to plot.
        layer_idx: Index of the layer.
        before_prune: Whether the weights are those before pruning.
    """
    weight = getattr(layer, 'weight', None)
    if not isinstance(weight, torch.Tensor):
        return  # nothing to plot
    weights = weight.detach().cpu()
    pruning_state = "Before Pruning" if before_prune else "After Pruning"

    if isinstance(layer, (nn.Conv2d, nn.Conv1d)):
        weights_np = weights.numpy().reshape(weights.shape[0], -1)
    else:
        weights_np = weights.numpy()

    weights_flat = weights_np.flatten()

    # Filter out NaN and check for valid data
    weights_valid = weights_flat[~np.isnan(weights_flat)]
    if len(weights_valid) == 0 or weights_valid.max() == weights_valid.min():
        # Skip plotting if all weights are NaN or constant (e.g., all zeros)
        return

    fig = plt.figure(figsize=(10, 8))
    plt.hist(
        weights_valid, bins=50, alpha=0.6,
        color='blue', edgecolor='black',
        density=True, label='Histogram'
    )

    try:
        kde = gaussian_kde(weights_valid)
        x = np.linspace(weights_valid.min(), weights_valid.max(), 1000)
        plt.plot(x, kde(x), color='red', label='KDE')
    except (np.linalg.LinAlgError, ValueError) as error:
        logger.debug(f"No KDE curve for layer {layer_idx}: {error}")  # degenerate data, e.g. too few weights

    plt.scatter([], [], color='green', label=f"Min: {weights_valid.min():.4f}")
    plt.scatter([], [], color='green', label=f"Max: {weights_valid.max():.4f}")
    plt.title(f'Weight Distribution {pruning_state}')
    plt.xlabel('Weight Values')
    plt.ylabel('Density')
    plt.legend()

    plt.grid(alpha=0.3)
    plt.tight_layout()

    save_plot(fig, _layer_folder(isinstance(layer, nn.Linear), layer_idx), f'Weight Distribution {pruning_state}')


def save_plot(figure: Figure, folder_name: str, file_name: str) -> None:
    """Save a figure to the plots folder of the current run's output directory, then close it.

    A plot that cannot be saved is reported as a warning; plots are not needed for pruning.

    Args:
        figure: Figure to save.
        folder_name: Subfolder of plots/.
        file_name: File name without extension.
    """
    plot_save_path = os.path.join(config.get_output_dir(), 'plots', folder_name, f'{file_name}.png')
    try:
        os.makedirs(os.path.dirname(plot_save_path), exist_ok=True)
        figure.savefig(plot_save_path)
    except Exception as error:  # any failure to save a plot must not stop the pruning
        logger.warning(f"Could not save plot {plot_save_path}: {error}")
    finally:
        plt.close(figure)
