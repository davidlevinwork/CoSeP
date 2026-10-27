"""Tests for the seeds of the random steps."""
import numpy as np
import pytest
import torch

from cosep.utils.random_utils import augmentation_generator, check_seed, layer_random_state


@pytest.mark.parametrize('seed', [-1, 1.5, '3', True, None])
def test_invalid_seeds_are_rejected(seed):
    with pytest.raises(ValueError, match='seed'):
        check_seed(seed)


def test_whole_numbers_are_seeds():
    assert check_seed(0) == 0 and check_seed(np.int64(7)) == 7


def test_each_layer_has_its_own_repeatable_generator():
    """The same seed and layer give the same numbers; another layer or seed gives others; the global generator is
    neither read nor changed.
    """
    state = np.random.get_state()
    draw = lambda seed, layer: layer_random_state(seed, layer).randint(0, 2 ** 31, 5)  # noqa: E731
    np.testing.assert_array_equal(draw(0, 3), draw(0, 3))
    assert not np.array_equal(draw(0, 3), draw(0, 4))
    assert not np.array_equal(draw(0, 3), draw(1, 3))
    np.testing.assert_equal(np.random.get_state(), state)


def test_the_augmentation_has_its_own_repeatable_generator():
    """The same seed gives the same shifts and flips, another seed others; the stream differs from the training
    loader's shuffle (seeded with the seed itself), and PyTorch's global generator is neither read nor changed.
    """
    state = torch.get_rng_state()
    draw = lambda generator: torch.randint(0, 2 ** 31, (8,), generator=generator)  # noqa: E731
    assert torch.equal(draw(augmentation_generator(4)), draw(augmentation_generator(4)))
    assert not torch.equal(draw(augmentation_generator(4)), draw(augmentation_generator(5)))
    assert not torch.equal(draw(augmentation_generator(4)), draw(torch.Generator().manual_seed(4)))
    assert torch.equal(torch.get_rng_state(), state)
    with pytest.raises(ValueError, match='seed'):
        augmentation_generator(-1)
