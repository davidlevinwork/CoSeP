"""Seeds for CoSeP's random steps: the training subset and its batch order, t-SNE, k-medoids and the augmentation."""
import numbers
from typing import Any

import numpy as np
import torch

# The selection's generators are made from [seed, layer index]; the augmentation's from [seed, this key], which no
# layer index takes.
_AUGMENTATION_KEY = 2 ** 32 - 1


def check_seed(seed: Any) -> int:
    """Return the seed as an int.

    Args:
        seed: The value to check.

    Returns:
        The seed.

    Raises:
        ValueError: If the seed is not a whole number of 0 or more.
    """
    if isinstance(seed, bool) or not isinstance(seed, numbers.Integral) or seed < 0:
        raise ValueError(f"seed must be a whole number, 0 or more; got {seed!r}.")
    return int(seed)


def layer_random_state(seed: int, layer_index: int) -> np.random.RandomState:
    """The random generator of one layer's selection (t-SNE and k-medoids), from the run's seed and the layer index.

    Each layer has its own generator, so a layer's random steps do not depend on how many random numbers the layers
    before it used, and NumPy's global generator is neither read nor changed.
    """
    state = np.random.SeedSequence([check_seed(seed), check_seed(layer_index)]).generate_state(1)[0]
    return np.random.RandomState(int(state))


def augmentation_generator(seed: int) -> torch.Generator:
    """The generator of the fine-tuning augmentation's shifts and flips, from the run's seed.

    It is a CPU generator of its own: independent of the layers' generators and of the training loader's shuffle
    (which is seeded with the seed itself), and PyTorch's global generator is neither read nor changed.
    """
    state = np.random.SeedSequence([check_seed(seed), _AUGMENTATION_KEY]).generate_state(1, dtype=np.uint64)[0]
    return torch.Generator().manual_seed(int(state))
