"""Test fixtures and configuration for CoSeP tests."""
import matplotlib

matplotlib.use('Agg')  # plots are saved to files; no display is needed

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402


@pytest.fixture
def sample_distributions():
    """Generate sample distributions for metric testing."""
    np.random.seed(42)
    p = np.random.normal(0, 1, 100)
    q = np.random.normal(2, 1.5, 100)
    return p, q


@pytest.fixture
def identical_distributions():
    """Generate identical distributions."""
    np.random.seed(42)
    p = np.random.normal(0, 1, 100)
    return p, p.copy()


@pytest.fixture
def sample_clustering_data():
    """Generate clustered data for clustering tests."""
    np.random.seed(42)
    cluster1 = np.random.randn(30, 2) + np.array([0, 0])
    cluster2 = np.random.randn(30, 2) + np.array([5, 5])
    cluster3 = np.random.randn(30, 2) + np.array([10, 0])
    return np.vstack([cluster1, cluster2, cluster3])


@pytest.fixture
def simple_model():
    """Create a simple test model."""
    model = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(32, 10)
    )
    return model


@pytest.fixture
def sample_dataloader():
    """Create sample dataloader for testing."""
    torch.manual_seed(42)
    images = torch.randn(100, 3, 32, 32)
    y = torch.randint(0, 10, (100,))
    dataset = TensorDataset(images, y)
    return DataLoader(dataset, batch_size=20, shuffle=False)


@pytest.fixture
def binary_dataloader():
    """Create binary classification dataloader."""
    torch.manual_seed(42)
    images = torch.randn(50, 3, 32, 32)
    y = torch.randint(0, 2, (50,))
    dataset = TensorDataset(images, y)
    return DataLoader(dataset, batch_size=10, shuffle=False)
