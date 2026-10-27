"""Tests for the physical removal of pruned components (cosep.pruning.structure)."""
import copy

import pytest
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, TensorDataset

from cosep.models.cifar_resnet import resnet56
from cosep.models.cifar_vgg import VGG16
from cosep.pruning.structure import ModelStructure, apply_pruning_plan, check_plan, count_macs, num_components
from cosep.utils.model_utils import train_model
from tests.tiny_models import FlattenNet, TinyNet, randomise_batchnorm

TINY_LAYERS = ['stem', 'res_conv1', 'res_conv2', 'expand', 'depthwise', 'project', 'dense1', 'dense2', 'fc1', 'fc2']


def _tiny(seed: int = 0) -> TinyNet:
    torch.manual_seed(seed)
    return randomise_batchnorm(TinyNet()).eval()


def _example() -> torch.Tensor:
    return torch.randn(1, 3, 12, 12)


def _ignore_inputs(model: nn.Module, consumer: str, columns) -> None:
    """Make a layer ignore some of its inputs, by zeroing the matching input weights."""
    with torch.no_grad():
        getattr(model, consumer).weight[:, list(columns)] = 0


class TestLayerRules:
    """Which layers may be pruned on their own."""

    def test_tiny_net(self):
        model = _tiny()
        structure = ModelStructure(model, _example())
        rules = {name: structure.rule(getattr(model, name)) for name in TINY_LAYERS}

        assert rules['res_conv1'].prunable
        assert not rules['res_conv2'].prunable and 'residual sum' in rules['res_conv2'].reason
        assert not rules['stem'].prunable  # writes into the same residual sum as res_conv2
        assert rules['expand'].prunable and rules['expand'].pruned_with == ['depthwise']
        assert not rules['depthwise'].prunable and 'depthwise' in rules['depthwise'].reason
        for name in ['project', 'dense1', 'dense2', 'fc1']:
            assert rules[name].prunable and rules[name].pruned_with == [], name
        assert not rules['fc2'].prunable and rules['fc2'].reason == 'model output'

    def test_repository_vgg16(self):
        model = VGG16(num_classes=10)
        structure = ModelStructure(model, torch.randn(1, 3, 32, 32))
        rules = [structure.rule(m) for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
        assert all(rule.prunable for rule in rules[:-1])  # a plain chain: no layer shares its channels
        assert rules[-1].reason == 'model output'

    def test_repository_resnet56(self):
        model = resnet56(num_classes=10)
        structure = ModelStructure(model, torch.randn(1, 3, 32, 32))
        convs = [(name, m) for name, m in model.named_modules() if isinstance(m, nn.Conv2d)]
        for name, conv in convs:
            rule = structure.rule(conv)
            if name.endswith('conv1'):
                assert rule.prunable, name  # inside a residual block
            else:
                assert not rule.prunable, name  # init_conv and every conv2: residual sum; final_conv: output
        assert sum(structure.rule(c).prunable for _, c in convs) == 27

    def test_mobilenet_v2(self):
        model = torchvision.models.mobilenet_v2(num_classes=10)
        structure = ModelStructure(model, torch.randn(1, 3, 32, 32))
        rules = [structure.rule(m) for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
        assert sum(r.prunable and len(r.pruned_with) == 1 for r in rules) == 17  # stem + 16 expand convs
        assert sum(r.prunable and not r.pruned_with for r in rules) == 3
        assert sum('depthwise' in r.reason for r in rules) == 17
        assert sum('residual sum' in r.reason for r in rules) == 15

    def test_attention_layers_are_not_supported(self):
        class WithAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Linear(8, 16)
                self.attention = nn.MultiheadAttention(16, 4, batch_first=True)
                self.fc = nn.Linear(16, 3)

            def forward(self, x):
                x = self.embed(x).unsqueeze(1)
                return self.fc(self.attention(x, x, x)[0].squeeze(1))

        model = WithAttention()
        structure = ModelStructure(model, torch.randn(1, 8))
        assert not structure.rule(model.attention).prunable
        assert 'not supported' in structure.rule(model.attention).reason

    def test_tracing_keeps_training_mode_and_statistics(self):
        model = TinyNet().train()
        stats = model.stem_bn.running_mean.clone()
        ModelStructure(model, torch.randn(2, 3, 12, 12))
        assert model.training
        assert torch.equal(model.stem_bn.running_mean, stats)


class TestRemoval:
    """Removed channels disappear from every layer that used them, and the network still computes."""

    def test_sequential_removal_equals_ignoring_the_channels(self):
        model = _tiny()
        reference = copy.deepcopy(model)
        structure = ModelStructure(model, _example())
        generator = torch.Generator().manual_seed(1)
        removed = {}
        for name in ['res_conv1', 'expand', 'project', 'dense1', 'dense2', 'fc1']:
            layer = getattr(model, name)
            n = num_components(layer)
            keep = sorted(torch.randperm(n, generator=generator)[: n // 2].tolist())
            removed[name] = [c for c in range(n) if c not in keep]
            structure.remove(layer, keep)
            assert num_components(layer) == len(keep)
        assert model.depthwise.out_channels == model.expand.out_channels == 8

        # The reference keeps every channel, but each layer that reads a removed channel ignores it.
        # Offsets follow the concatenations: [project (12), dense1 (8), dense2 (8)].
        _ignore_inputs(reference, 'res_conv2', removed['res_conv1'])
        _ignore_inputs(reference, 'project', removed['expand'])
        for consumer in ['dense1', 'dense2', 'fc1']:
            _ignore_inputs(reference, consumer, removed['project'])
        for consumer in ['dense2', 'fc1']:
            _ignore_inputs(reference, consumer, [12 + c for c in removed['dense1']])
        _ignore_inputs(reference, 'fc1', [20 + c for c in removed['dense2']])
        _ignore_inputs(reference, 'fc2', removed['fc1'])

        x = torch.randn(4, 3, 12, 12)
        with torch.no_grad():
            torch.testing.assert_close(model(x), reference(x), rtol=1e-4, atol=1e-5)

    def test_removal_before_a_flatten(self):
        torch.manual_seed(0)
        model = randomise_batchnorm(FlattenNet()).eval()
        reference = copy.deepcopy(model)
        keep = [0, 2, 3, 6]
        ModelStructure(model, torch.randn(1, 3, 8, 8)).remove(model.conv2, keep)
        dropped = [c for c in range(8) if c not in keep]
        _ignore_inputs(reference, 'fc1', [4 * c + j for c in dropped for j in range(4)])  # channel c -> 4 features

        assert model.fc1.in_features == 4 * len(keep)
        x = torch.randn(3, 3, 8, 8)
        with torch.no_grad():
            torch.testing.assert_close(model(x), reference(x), rtol=1e-4, atol=1e-5)

    def test_removed_channels_stay_removed_after_fine_tuning(self):
        torch.manual_seed(0)
        model = TinyNet()
        structure = ModelStructure(model, _example())
        structure.remove(model.res_conv1, [0, 1, 2, 3])
        structure.remove(model.expand, list(range(0, 16, 2)))
        images, y = torch.randn(48, 3, 12, 12), torch.arange(48) % 3
        loader = DataLoader(TensorDataset(images, y), batch_size=16, shuffle=True)

        train_model(model, loader, loader, num_epochs=2, verbose=False)

        assert model.res_conv1.weight.shape[0] == 4 and model.res_bn1.num_features == 4
        assert model.res_conv2.weight.shape[1] == 4
        assert model.expand.out_channels == model.depthwise.out_channels == model.project.in_channels == 8
        with torch.no_grad():
            assert model.eval()(images[:2]).shape == (2, 3)

    @pytest.mark.parametrize('keep', [[], [-1, 2], [0, 12]])
    def test_invalid_kept_indices(self, keep):
        model = TinyNet()
        with pytest.raises(ValueError):
            ModelStructure(model, _example()).remove(model.res_conv1, keep)

    def test_keeping_everything_changes_nothing(self):
        model = TinyNet()
        assert ModelStructure(model, _example()).remove(model.res_conv1, range(12)) == []
        assert model.res_conv1.out_channels == 12

    def test_remove_reports_the_resized_modules(self):
        model = TinyNet()
        resized = ModelStructure(model, _example()).remove(model.expand, [1, 2, 3])
        assert {'expand', 'expand_bn', 'depthwise', 'depthwise_bn', 'project'} <= set(resized)


class TestCountMacs:
    """Multiply-accumulates of convolution and linear layers."""

    def test_known_counts(self):
        model = nn.Sequential(
            nn.Conv2d(3, 4, 3, padding=1), nn.Conv2d(4, 4, 3, padding=1, groups=4),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 2),
        )
        macs = count_macs(model, torch.randn(2, 3, 8, 8))  # per input, whatever the batch size
        assert macs[model[0]] == 8 * 8 * 4 * 3 * 9
        assert macs[model[1]] == 8 * 8 * 4 * 1 * 9
        assert macs[model[4]] == 4 * 2

    def test_removal_reduces_macs_of_both_neighbours(self):
        model = TinyNet()
        x = _example()
        before = sum(count_macs(model, x).values())
        ModelStructure(model, x).remove(model.res_conv1, [0, 1, 2])
        after = sum(count_macs(model, x).values())
        # On a 12x12 map, res_conv1 loses 9 of its 12 outputs and res_conv2 loses 9 of its 12 inputs.
        assert before - after == 2 * (12 * 12) * 9 * 8 * 9


class TestPruningPlan:
    """The plan records the kept channels and rebuilds the pruned architecture."""

    def test_check_plan_detects_a_wrong_size(self):
        with pytest.raises(RuntimeError):
            check_plan(TinyNet(), {'layers': [{'layer': 'res_conv1', 'components_after': 5}]})

    def test_apply_plan_reproduces_the_same_channels(self):
        original = _tiny()
        pruned = copy.deepcopy(original)
        structure = ModelStructure(pruned, _example())
        layers = []
        for name, keep in [('res_conv1', [1, 4, 7]), ('expand', [0, 3, 5, 9, 12]), ('dense1', [2, 6])]:
            structure.remove(getattr(pruned, name), keep)
            layers.append({'layer': name, 'status': 'pruned', 'kept': keep, 'components_after': len(keep)})
        layers.append({'layer': 'depthwise', 'status': 'pruned with', 'pruned_with': 'expand',
                       'kept': [0, 3, 5, 9, 12], 'components_after': 5})

        rebuilt = apply_pruning_plan(copy.deepcopy(original), {'layers': layers}, _example())

        for key, value in pruned.state_dict().items():  # same channels kept, not only the same sizes
            assert torch.equal(rebuilt.state_dict()[key], value), key
