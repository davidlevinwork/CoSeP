"""Tests for layer utilities."""
import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from cosep.utils.layer_utils import collect_class_statistics, extract_weights
from cosep.utils.model_utils import get_model_layers


class TestExtractWeights:
    """Test weight extraction from layers."""

    def test_conv2d_weights(self, simple_model):
        """Should extract Conv2d weight norms."""
        weights = extract_weights(simple_model, layer_index=0)

        # First conv layer has 16 output channels
        assert len(weights) == 16
        assert all(w >= 0 for w in weights)  # L2 norms are non-negative

    def test_linear_weights(self, simple_model):
        """Should extract Linear weight norms."""
        weights = extract_weights(simple_model, layer_index=2)

        # Linear layer has 10 outputs
        assert len(weights) == 10

    def test_returns_numpy(self, simple_model):
        """Should return numpy array."""
        weights = extract_weights(simple_model, layer_index=0)
        assert isinstance(weights, np.ndarray)


def _outputs_of(model, layer_index, loader):
    """All outputs of a layer, copied as soon as they are produced (reference)."""
    outputs = []
    layer = list(get_model_layers(model))[layer_index]
    hook = layer.register_forward_hook(lambda module, inputs, output: outputs.append(output.detach().clone()))
    model.eval()
    with torch.no_grad():
        for x, _ in loader:
            model(x)
    hook.remove()
    return torch.cat(outputs)


def _summaries(outputs):
    """Eq. 1 (convolutions) and Eq. 2 (linear layers): one value per component and sample (reference)."""
    x = outputs.double()
    return x.reshape(x.shape[0], x.shape[1], -1).mean(dim=2) if x.dim() > 2 else x


def _class_statistics_of(values, labels):
    """Per-class mean and population variance of all values at once, in float64 (reference)."""
    classes = torch.unique(labels)
    x = values.double()
    means = torch.stack([x[labels == c].mean(0) for c in classes]).numpy()
    variances = torch.stack([x[labels == c].var(0, unbiased=False) for c in classes]).numpy()
    return classes.numpy(), means, variances


class TestCollectClassStatistics:
    """Per-class statistics of a layer's components, collected batch by batch without storing the outputs."""

    def test_conv2d_shapes(self, simple_model, sample_dataloader):
        """Statistics per class and channel: one value per channel and image (Eq. 1)."""
        stats = collect_class_statistics(simple_model, sample_dataloader, layer_index=0)
        labels = torch.cat([y for _, y in sample_dataloader])

        assert stats.classes.tolist() == torch.unique(labels).tolist()
        assert stats.means.shape == (len(stats.classes), 16)
        assert stats.variances.shape == stats.means.shape
        assert stats.counts.sum() == 100
        assert stats.num_components == 16

    def test_linear_shapes(self, simple_model, sample_dataloader):
        """Statistics per class and neuron."""
        stats = collect_class_statistics(simple_model, sample_dataloader, layer_index=2)
        assert stats.means.shape == (len(stats.classes), 10)
        assert stats.num_components == 10

    def test_conv2d_channel_value_is_the_mean_of_its_map(self):
        """Eq. 1: a channel's value for an image is the mean of its output map; the spread within the map is lost."""
        conv = nn.Conv2d(1, 2, 1, bias=False)
        with torch.no_grad():
            conv.weight.copy_(torch.tensor([1.0, -2.0]).view(2, 1, 1, 1))
        model = nn.Sequential(conv, nn.Flatten(), nn.Linear(2 * 2 * 2, 2))
        images = torch.tensor([[[[1.0, 3.0], [5.0, 7.0]]], [[[0.0, 0.0], [0.0, 8.0]]],
                               [[[2.0, 2.0], [2.0, 2.0]]], [[[1.0, 1.0], [1.0, 9.0]]]])
        labels = torch.tensor([0, 0, 1, 1])
        stats = collect_class_statistics(model, DataLoader(TensorDataset(images, labels), batch_size=3), 0)

        # Image means: 4, 2 (class 0) and 2, 3 (class 1); channel 1 is -2 times channel 0.
        np.testing.assert_allclose(stats.means, [[3.0, -6.0], [2.5, -5.0]])
        np.testing.assert_allclose(stats.variances, [[1.0, 4.0], [0.25, 1.0]])

    @pytest.mark.parametrize('layer_index', [0, 1, 2])
    def test_equal_to_the_statistics_of_all_outputs(self, simple_model, layer_index):
        """Merging batches gives the statistics of all outputs at once, for any labels and batch sizes."""
        torch.manual_seed(0)
        x = torch.randn(70, 3, 32, 32) * 3 + 5
        labels = torch.tensor([2, 5, 7, 5, 2, 9, 7] * 10)  # not 0..K-1; the last batch is smaller
        loader = DataLoader(TensorDataset(x, labels), batch_size=16)

        stats = collect_class_statistics(simple_model, loader, layer_index)
        classes, means, variances = _class_statistics_of(
            _summaries(_outputs_of(simple_model, layer_index, loader)), labels)

        np.testing.assert_array_equal(stats.classes, classes)
        np.testing.assert_array_equal(stats.counts, [20, 20, 20, 10])
        np.testing.assert_allclose(stats.means, means, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(stats.variances, variances, rtol=1e-9, atol=1e-12)

    def test_conv1d_channel_value_is_the_mean_over_positions(self):
        torch.manual_seed(0)
        model = nn.Sequential(nn.Conv1d(3, 5, 3, padding=1), nn.Flatten(), nn.Linear(5 * 7, 2))
        x = torch.randn(30, 3, 7)
        labels = torch.arange(30) % 3
        loader = DataLoader(TensorDataset(x, labels), batch_size=8)

        stats = collect_class_statistics(model, loader, layer_index=0)
        _, means, variances = _class_statistics_of(_summaries(_outputs_of(model, 0, loader)), labels)
        assert stats.means.shape == (3, 5)
        np.testing.assert_allclose(stats.means, means, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(stats.variances, variances, rtol=1e-9, atol=1e-12)

    def test_reads_the_output_before_in_place_operations(self):
        """A ReLU(inplace=True) after the layer must not change its statistics (on CPU it did, before)."""
        torch.manual_seed(0)
        model = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(inplace=True), nn.Flatten(),
                              nn.Linear(8 * 6 * 6, 4), nn.ReLU(inplace=True), nn.Linear(4, 2)).eval()
        x = torch.randn(40, 3, 6, 6)
        labels = torch.arange(40) % 2
        loader = DataLoader(TensorDataset(x, labels), batch_size=10)
        with torch.no_grad():
            conv_out = model[0](x)
            fc_out = model[3](torch.flatten(torch.relu(conv_out), 1))
        assert (conv_out < 0).any() and (fc_out < 0).any()  # the ReLUs would change them

        for layer_index, expected in [(0, conv_out), (1, fc_out)]:
            stats = collect_class_statistics(model, loader, layer_index)
            _, means, variances = _class_statistics_of(_summaries(expected), labels)
            np.testing.assert_allclose(stats.means, means, rtol=1e-5, atol=1e-6)
            np.testing.assert_allclose(stats.variances, variances, rtol=1e-5, atol=1e-6)

    def test_restores_training_mode(self, simple_model, sample_dataloader):
        """The model is evaluated in eval mode and handed back in the mode it had."""
        simple_model.train()
        collect_class_statistics(simple_model, sample_dataloader, layer_index=0)
        assert simple_model.training

    def test_non_finite_outputs_raise(self, simple_model, sample_dataloader):
        """NaN or Inf outputs are an error, not 'no separability'."""
        with torch.no_grad():
            simple_model[0].weight[0, 0, 0, 0] = float('nan')
        with pytest.raises(ValueError, match='NaN or Inf'):
            collect_class_statistics(simple_model, sample_dataloader, layer_index=0)

    def test_linear_layer_on_token_sequences_is_rejected(self):
        """A linear layer applied to (batch, tokens, features) inputs has no single value per sample."""
        model = nn.Sequential(nn.Linear(4, 3), nn.Flatten(), nn.Linear(15, 2))
        loader = DataLoader(TensorDataset(torch.randn(8, 5, 4), torch.arange(8) % 2), batch_size=4)
        with pytest.raises(ValueError, match='token sequences'):
            collect_class_statistics(model, loader, layer_index=0)

    def test_layer_used_twice_per_forward_pass_is_rejected(self):
        """Outputs of two calls cannot be matched to the batch's labels."""
        class Twice(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(4, 4)

            def forward(self, x):
                return self.fc(self.fc(x))

        loader = DataLoader(TensorDataset(torch.randn(8, 4), torch.arange(8) % 2), batch_size=4)
        with pytest.raises(ValueError, match='more than once'):
            collect_class_statistics(Twice(), loader, layer_index=0)
