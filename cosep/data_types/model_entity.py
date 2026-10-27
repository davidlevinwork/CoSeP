"""Model entity for model information."""
from dataclasses import dataclass
from typing import Optional

import torch.nn as nn


@dataclass
class ModelEntity:
    """Model information container.

    Attributes:
        model: PyTorch model instance.
        model_path: Path of the file the model was loaded from, or None.
    """
    model: nn.Module
    model_path: Optional[str]
