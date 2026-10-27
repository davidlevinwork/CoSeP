"""Tests for model utilities."""
import copy

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from cosep.utils.model_utils import (
    AUGMENT_PADDING,
    augment_batch,
    evaluate_model,
    execution_order,
    get_model_layers,
    train_model,
)


class HeadFirstNet(nn.Module):
    """Defines its classifier before its feature extractor, and has a layer that the forward pass never runs."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 3)
        self.unused = nn.Linear(8, 8)
        self.conv1 = nn.Conv2d(3, 6, 3, padding=1)
        self.conv2 = nn.Conv2d(6, 8, 3, padding=1)

    def forward(self, x):
        x = torch.relu(self.conv2(torch.relu(self.conv1(x))))
        return self.fc(torch.flatten(nn.functional.adaptive_avg_pool2d(x, 1), 1))


class TestExecutionOrder:
    """The order in which one forward pass runs the layers."""

    def test_follows_the_forward_pass_not_the_definitions(self):
        model = HeadFirstNet()
        layers = list(get_model_layers(model))
        assert layers == [model.fc, model.unused, model.conv1, model.conv2]  # definition order
        assert execution_order(model, layers, torch.randn(1, 3, 6, 6)) == [2, 3, 0]  # unused is left out

    def test_restores_training_mode(self):
        model = HeadFirstNet().train()
        execution_order(model, list(get_model_layers(model)), torch.randn(1, 3, 6, 6))
        assert model.training


class TestTrainModel:
    """The fine-tuning optimizer: SGD with the given settings, by default learning rate 0.001, momentum 0.9 and
    weight decay 5e-4.
    """

    @staticmethod
    def _spy_sgd(monkeypatch):
        calls = []
        sgd = torch.optim.SGD

        def spy(params, **kwargs):
            calls.append(kwargs)
            return sgd(params, **kwargs)

        monkeypatch.setattr(torch.optim, 'SGD', spy)
        return calls

    @staticmethod
    def _data():
        torch.manual_seed(0)
        loader = DataLoader(TensorDataset(torch.randn(16, 3, 6, 6), torch.arange(16) % 3), batch_size=8)
        return HeadFirstNet(), loader

    def test_defaults(self, monkeypatch):
        calls = self._spy_sgd(monkeypatch)
        model, loader = self._data()
        train_model(model, loader, loader, num_epochs=1, verbose=False)
        assert calls == [dict(lr=0.001, momentum=0.9, weight_decay=5e-4, nesterov=False)]

    def test_uses_the_given_settings_whatever_the_model_is_called(self, monkeypatch):
        calls = self._spy_sgd(monkeypatch)
        model, loader = self._data()
        train_model(model, loader, loader, num_epochs=1, learning_rate=0.01, momentum=0.8, weight_decay=1e-4,
                    nesterov=True, verbose=False)
        assert calls == [dict(lr=0.01, momentum=0.8, weight_decay=1e-4, nesterov=True)]

    def test_is_plain_sgd_over_the_loader(self):
        """Two epochs of train_model equal a hand-written SGD loop over the same batches: a constant learning rate,
        no schedule, no augmentation, cross-entropy loss.
        """
        model, loader = self._data()
        reference = copy.deepcopy(model)
        train_model(model, loader, loader, num_epochs=2, learning_rate=0.05, momentum=0.9, weight_decay=1e-4,
                    nesterov=True, verbose=False)
        optimizer = torch.optim.SGD(reference.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4, nesterov=True)
        reference.train()
        for _ in range(2):
            for x, y in loader:
                optimizer.zero_grad()
                nn.functional.cross_entropy(reference(x), y).backward()
                optimizer.step()
        for name, value in reference.state_dict().items():
            torch.testing.assert_close(model.state_dict()[name], value, msg=name)

    def test_augmentation_is_off_by_default(self, monkeypatch):
        import cosep.utils.model_utils as model_utils
        monkeypatch.setattr(model_utils, 'augment_batch', lambda *args, **kwargs: pytest.fail('augmented'))
        model, loader = self._data()
        train_model(model, loader, loader, num_epochs=1, verbose=False)

    def test_augmented_fine_tuning_is_sgd_over_the_augmented_batches(self):
        """With augment, train_model equals the hand-written SGD loop fed augment_batch of every training batch, with
        the same generator.
        """
        model, loader = self._data()
        reference = copy.deepcopy(model)
        train_model(model, loader, loader, num_epochs=2, learning_rate=0.05, momentum=0.9, weight_decay=1e-4,
                    nesterov=True, verbose=False, augment=True, generator=torch.Generator().manual_seed(9))
        generator = torch.Generator().manual_seed(9)
        optimizer = torch.optim.SGD(reference.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4, nesterov=True)
        reference.train()
        for _ in range(2):
            for x, y in loader:
                optimizer.zero_grad()
                nn.functional.cross_entropy(reference(augment_batch(x, generator)), y).backward()
                optimizer.step()
        for name, value in reference.state_dict().items():
            torch.testing.assert_close(model.state_dict()[name], value, msg=name)

    def test_augmentation_needs_images(self):
        loader = DataLoader(TensorDataset(torch.randn(8, 5), torch.arange(8) % 2), batch_size=4)
        with pytest.raises(ValueError, match='images'):
            train_model(nn.Linear(5, 2), loader, loader, num_epochs=1, verbose=False, augment=True)


def _augment_one_at_a_time(images, generator, padding=4, fill=0.0):
    """Pad, crop at the drawn position and flip if drawn, one image at a time (the recipe of the review's
    experiments), with the draws in the same order: rows, columns, flips.
    """
    n, c, h, w = images.shape
    padded = torch.as_tensor(fill, dtype=images.dtype).reshape(1, -1, 1, 1).expand(
        n, c, h + 2 * padding, w + 2 * padding).clone()
    padded[:, :, padding:padding + h, padding:padding + w] = images
    rows = torch.randint(0, 2 * padding + 1, (n,), generator=generator).tolist()
    cols = torch.randint(0, 2 * padding + 1, (n,), generator=generator).tolist()
    flips = (torch.rand(n, generator=generator) < 0.5).tolist()
    out = torch.empty_like(images)
    for i in range(n):
        crop = padded[i, :, rows[i]:rows[i] + h, cols[i]:cols[i] + w]
        out[i] = crop.flip(-1) if flips[i] else crop
    return out


def _crop(padded, row, col, flip, size):
    """The crop of a padded image at (row, col), of the image's original size, flipped or not."""
    crop = padded[..., row:row + size[0], col:col + size[1]]
    return crop.flip(-1) if flip else crop


class TestAugmentBatch:
    """The fine-tuning augmentation: a random crop of the image padded by 4 pixels, then a horizontal flip with
    probability 1/2.
    """

    def test_shifts_each_image_by_up_to_four_pixels_and_flips_about_half(self):
        """Pixel values that encode their position show where each crop was taken: every output is its own image,
        zero-padded and cropped at one of the 9 x 9 positions, flipped or not. All positions occur.
        """
        n, h, w = 1000, 6, 6
        images = (1 + torch.arange(h * w, dtype=torch.float32)).reshape(1, 1, h, w).repeat(n, 2, 1, 1)
        images[:, 1] *= -1  # the channels are cropped alike
        out = augment_batch(images, torch.Generator().manual_seed(4))
        positions, flipped = set(), 0
        for image, result in zip(images, out):
            padded = nn.functional.pad(image, (4, 4, 4, 4))
            found = [(row, col, flip) for row in range(9) for col in range(9) for flip in (False, True)
                     if torch.equal(result, _crop(padded, row, col, flip, (h, w)))]
            assert len(found) == 1
            positions.add(found[0][:2])
            flipped += found[0][2]
        assert AUGMENT_PADDING == 4
        assert positions == {(row, col) for row in range(9) for col in range(9)}
        assert 0.45 * n < flipped < 0.55 * n

    @pytest.mark.parametrize('fill', [0.0, -1.5, [-2.0, -1.0, 0.5]])
    def test_equal_to_augmenting_one_image_at_a_time(self, fill):
        images = torch.randn(64, 3, 9, 7)
        out = augment_batch(images, torch.Generator().manual_seed(1), fill=fill)
        assert torch.equal(out, _augment_one_at_a_time(images, torch.Generator().manual_seed(1), fill=fill))

    def test_padded_pixels_take_the_fill_value(self):
        images = torch.ones(200, 3, 5, 5)
        assert set(augment_batch(images, torch.Generator().manual_seed(0)).unique().tolist()) == {0.0, 1.0}
        out = augment_batch(images, torch.Generator().manual_seed(0), fill=[-1.0, -2.0, -3.0])
        for channel, value in enumerate([-1.0, -2.0, -3.0]):
            assert set(out[:, channel].unique().tolist()) == {value, 1.0}

    def test_keeps_shape_and_dtype_and_leaves_the_input_unchanged(self):
        images = torch.randn(8, 2, 5, 4, dtype=torch.float64)
        before = images.clone()
        out = augment_batch(images, torch.Generator().manual_seed(0))
        assert out.shape == images.shape and out.dtype == images.dtype
        assert torch.equal(images, before)

    def test_without_padding_it_only_flips(self):
        images = torch.randn(50, 3, 4, 4)
        out = augment_batch(images, torch.Generator().manual_seed(0), padding=0)
        flipped = [torch.equal(o, i.flip(-1)) for o, i in zip(out, images)]
        assert all(f or torch.equal(o, i) for f, o, i in zip(flipped, out, images))
        assert 0 < sum(flipped) < len(images)

    def test_repeatable_with_a_generator_and_leaves_the_global_generator_alone(self):
        images = torch.randn(32, 3, 8, 8)
        state = torch.get_rng_state()
        first = augment_batch(images, torch.Generator().manual_seed(5))
        assert torch.equal(first, augment_batch(images, torch.Generator().manual_seed(5)))
        assert not torch.equal(first, augment_batch(images, torch.Generator().manual_seed(6)))
        assert torch.equal(torch.get_rng_state(), state)

    @pytest.mark.parametrize('shape', [(4, 3, 10), (4, 10), (3, 8, 8, 2, 2)])
    def test_inputs_that_are_not_images_are_rejected(self, shape):
        with pytest.raises(ValueError, match='images'):
            augment_batch(torch.randn(*shape))

    def test_invalid_padding_and_fill_are_rejected(self):
        with pytest.raises(ValueError, match='padding'):
            augment_batch(torch.randn(2, 3, 4, 4), padding=-1)
        with pytest.raises(ValueError, match='fill'):
            augment_batch(torch.randn(2, 3, 4, 4), fill=[0.0, 1.0])


class TestGetModelLayers:
    """Test model layer extraction."""

    def test_simple_sequential(self, simple_model):
        """Should extract Conv and Linear layers."""
        layers = list(get_model_layers(simple_model))

        layer_types = [type(layer).__name__ for layer in layers]
        assert 'Conv2d' in layer_types
        assert 'Linear' in layer_types

    def test_excludes_non_prunable(self, simple_model):
        """Should exclude ReLU, MaxPool, etc."""
        layers = list(get_model_layers(simple_model))

        for layer in layers:
            assert not isinstance(layer, (nn.ReLU, nn.MaxPool2d))

    def test_counts_layers(self, simple_model):
        """Should find correct number of prunable layers."""
        layers = list(get_model_layers(simple_model))
        # 2 Conv2d + 1 Linear = 3
        assert len(layers) == 3

    def test_attention_is_one_layer(self):
        """A MultiheadAttention layer is listed whole; its output projection is not listed on its own."""
        model = nn.Sequential(nn.Linear(32, 16), nn.MultiheadAttention(16, 4, batch_first=True))
        layers = list(get_model_layers(model))
        assert len(layers) == 2 and isinstance(layers[1], nn.MultiheadAttention)


class TestEvaluateModel:
    """Test model evaluation function."""

    def test_returns_metrics(self, simple_model, sample_dataloader):
        """Should return Metrics object."""
        from cosep.data_types.metrics import Metrics

        metrics = evaluate_model(simple_model, sample_dataloader)

        assert isinstance(metrics, Metrics)
        assert hasattr(metrics, 'accuracy')
        assert hasattr(metrics, 'precision')
        assert hasattr(metrics, 'recall')
        assert hasattr(metrics, 'f1_score')

    def test_accuracy_bounded(self, simple_model, sample_dataloader):
        """Accuracy should be between 0 and 1."""
        metrics = evaluate_model(simple_model, sample_dataloader)
        assert 0 <= metrics.accuracy <= 1

    def test_has_inference_time(self, simple_model, sample_dataloader):
        """Should measure inference time."""
        metrics = evaluate_model(simple_model, sample_dataloader)
        assert metrics.inference_time > 0

    def test_has_model_size(self, simple_model, sample_dataloader):
        """Should measure model size."""
        metrics = evaluate_model(simple_model, sample_dataloader)
        assert metrics.model_size > 0

    def test_per_class_metrics(self, simple_model, sample_dataloader):
        """Should compute per-class metrics."""
        metrics = evaluate_model(simple_model, sample_dataloader)
        assert len(metrics.per_class_metrics) > 0

    def test_binary_classification(self, simple_model, binary_dataloader):
        """Should handle binary classification."""
        # Modify model for binary output
        simple_model[-1] = nn.Linear(32, 2)
        metrics = evaluate_model(simple_model, binary_dataloader)
        assert len(metrics.per_class_metrics) == 2


class _FixedPredictions(nn.Module):
    """Predicts the class stored in the first input feature, out of num_outputs classes."""

    def __init__(self, num_outputs: int):
        super().__init__()
        self.num_outputs = num_outputs
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return nn.functional.one_hot(x[:, 0].long(), self.num_outputs).float() * self.scale


def _loader(predictions, labels):
    return DataLoader(TensorDataset(torch.tensor(predictions).float().unsqueeze(1), torch.tensor(labels)),
                      batch_size=4)


class TestPerClassMetrics:
    """Per-class metrics are computed for the classes in the labels, whatever their values."""

    def test_labels_that_are_not_0_to_k(self):
        metrics = evaluate_model(_FixedPredictions(5), _loader([1, 1, 3, 3, 3, 0], [1, 3, 3, 3, 1, 1]))
        per_class = metrics.per_class_metrics

        assert set(per_class) == {1, 3}
        assert per_class[1]['precision'] == pytest.approx(1 / 2) and per_class[1]['recall'] == pytest.approx(1 / 3)
        assert per_class[3]['precision'] == pytest.approx(2 / 3) and per_class[3]['recall'] == pytest.approx(2 / 3)
        assert per_class[1]['accuracy'] == pytest.approx(1 / 3)
        assert metrics.accuracy == pytest.approx(0.5)
        assert metrics.precision == pytest.approx((1 / 2 + 2 / 3) / 2)

    def test_two_classes_are_scored_separately(self):
        metrics = evaluate_model(_FixedPredictions(2), _loader([0, 0, 1, 1, 1], [0, 1, 1, 1, 0]))
        per_class = metrics.per_class_metrics

        assert per_class[0]['precision'] == pytest.approx(1 / 2) and per_class[0]['recall'] == pytest.approx(1 / 2)
        assert per_class[1]['precision'] == pytest.approx(2 / 3) and per_class[1]['recall'] == pytest.approx(2 / 3)
        assert metrics.recall == pytest.approx((1 / 2 + 2 / 3) / 2)
