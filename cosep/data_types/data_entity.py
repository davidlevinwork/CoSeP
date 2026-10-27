"""Data entity for dataset information."""
from dataclasses import dataclass
from typing import Optional

import numpy as np
from torch.utils.data import DataLoader


@dataclass
class DataEntity:
    """Dataset information container.

    Attributes:
        data_loader: DataLoader that yields (inputs, labels) batches.
        num_classes: Number of distinct classes in the labels.
        data_path: Path the data were loaded from, or None.
        total_samples: Number of samples.
        labels: Label of every sample, in dataset order.
        unique_label_indices: The distinct labels, sorted.
    """
    data_loader: DataLoader
    num_classes: int
    data_path: Optional[str]
    total_samples: int
    labels: np.ndarray
    unique_label_indices: np.ndarray
