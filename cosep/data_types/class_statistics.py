"""Per-class statistics of a layer's outputs."""
from dataclasses import dataclass

import numpy as np


@dataclass
class ClassStatistics:
    """Mean and variance of every component's activation summary, for each class (paper, Eqs. 1-4).

    A component's activation summary is one value per sample: the mean of a convolution channel's output map over
    its positions (Eq. 1), or a linear neuron's output (Eq. 2). The separability profiles (Eqs. 5-7) depend on the
    outputs only through these statistics.

    Attributes:
        classes: Class labels, in increasing order.
        counts: Number of samples of each class.
        means: Per-class mean of every component, shape (classes, components).
        variances: Per-class population variance (ddof=0) of every component, same shape as `means`.
    """
    classes: np.ndarray
    counts: np.ndarray
    means: np.ndarray
    variances: np.ndarray

    @property
    def num_components(self) -> int:
        """Number of output channels (or neurons) of the layer."""
        return self.means.shape[1]
