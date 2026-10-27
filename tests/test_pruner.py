"""End-to-end test of the Pruner: selection, removal, fine-tuning and the files it writes."""
import csv
import json
import os
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader, TensorDataset

import cosep.config as config
import cosep.pruning.pruner as pruner_module
from cosep.data_types.data_entity import DataEntity
from cosep.data_types.model_entity import ModelEntity
from cosep.pruning.pruner import SUMMARY_COLUMNS, Pruner
from cosep.pruning.structure import ModelStructure, apply_pruning_plan, count_macs, num_components
from cosep.utils.model_utils import execution_order
from tests.tiny_models import TinyNet

PRUNED = ['res_conv1', 'expand', 'project', 'dense1', 'dense2', 'fc1']


def _data(seed: int) -> DataEntity:
    generator = torch.Generator().manual_seed(seed)
    labels = torch.arange(3).repeat_interleave(24)
    images = torch.randn(len(labels), 3, 12, 12, generator=generator) + 0.5 * labels.view(-1, 1, 1, 1)
    loader = DataLoader(TensorDataset(images, labels), batch_size=24, shuffle=False)
    return DataEntity(data_loader=loader, num_classes=3, data_path=None, total_samples=len(labels),
                      labels=labels.numpy(), unique_label_indices=np.arange(3))


@pytest.fixture(scope='module')
def run(tmp_path_factory):
    """Prune TinyNet once; the tests inspect the output folder."""
    workdir = tmp_path_factory.mktemp('pruning_run')
    previous_cwd = os.getcwd()
    os.chdir(workdir)
    config.output_dir = ''
    try:
        torch.manual_seed(0)
        model = TinyNet()
        Pruner(ModelEntity(model=model, model_path=None), _data(0), _data(1), retrain_epochs=1).prune()
        yield os.path.join(workdir, config.get_output_dir())
    finally:
        os.chdir(previous_cwd)
        config.output_dir = ''


def _plan(run_dir):
    with open(os.path.join(run_dir, 'pruning_plan.json')) as f:
        return json.load(f)


def _summary(run_dir):
    with open(os.path.join(run_dir, 'pruning_summary.csv'), newline='') as f:
        return list(csv.DictReader(f))


def _pruned_model(run_dir):
    return torch.load(os.path.join(run_dir, 'pruned_model.pt'), weights_only=False).eval()


def _original_model(run_dir):
    model = TinyNet()
    checkpoint = torch.load(os.path.join(run_dir, 'original_model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    return model.eval()


def test_writes_all_outputs(run):
    for name in ['original_model.pt', 'pruned_model.pt', 'pruning_plan.json', 'pruning_summary.csv']:
        assert os.path.isfile(os.path.join(run, name)), name


def test_plan_follows_the_layer_rules(run):
    layers = {entry['layer']: entry for entry in _plan(run)['layers']}
    assert list(layers) == ['stem', 'res_conv1', 'res_conv2', 'expand', 'depthwise', 'project',
                            'dense1', 'dense2', 'fc1', 'fc2']
    assert layers['stem']['status'] == 'not pruned' and layers['stem']['reason'] == 'first layer'
    assert layers['fc2']['status'] == 'not pruned' and layers['fc2']['reason'] == 'last layer'
    assert layers['res_conv2']['status'] == 'not pruned' and 'residual sum' in layers['res_conv2']['reason']
    for name in PRUNED:
        entry = layers[name]
        kept = entry['kept']
        assert entry['status'] == 'pruned', name
        assert 2 <= entry['components_after'] < entry['components_before'], name
        assert kept == sorted(set(kept)) and len(kept) == entry['components_after'], name
        assert 0 <= kept[0] and kept[-1] < entry['components_before'], name
    assert layers['depthwise']['status'] == 'pruned with' and layers['depthwise']['pruned_with'] == 'expand'
    assert layers['depthwise']['kept'] == layers['expand']['kept']


def test_pruned_model_loads_and_computes(run):
    model = _pruned_model(run)
    modules = dict(model.named_modules())
    for entry in _plan(run)['layers']:
        assert num_components(modules[entry['layer']]) == entry['components_after'], entry['layer']
    with torch.no_grad():
        assert model(torch.randn(5, 3, 12, 12)).shape == (5, 3)


def test_summary_matches_both_models(run):
    rows = _summary(run)
    assert list(rows[0]) == SUMMARY_COLUMNS
    assert rows[-1]['layer'] == 'WHOLE MODEL'
    example = torch.randn(1, 3, 12, 12)

    for column, model in [('before', _original_model(run)), ('after', _pruned_model(run))]:
        modules = dict(model.named_modules())
        macs = count_macs(model, example)
        for row in rows[:-1]:
            layer = modules[row['layer']]
            assert int(row[f'out_{column}']) == num_components(layer), row['layer']
            assert int(row[f'params_{column}']) == sum(p.numel() for p in layer.parameters()), row['layer']
            assert int(row[f'macs_{column}']) == macs[layer], row['layer']
        assert int(rows[-1][f'params_{column}']) == sum(p.numel() for p in model.parameters())
        assert int(rows[-1][f'macs_{column}']) == sum(macs.values())

    total = rows[-1]
    assert int(total['params_after']) < int(total['params_before'])
    assert int(total['macs_after']) < int(total['macs_before'])
    by_layer = {row['layer']: row for row in rows}
    for name in ['stem', 'res_conv2', 'fc2']:
        assert by_layer[name]['out_after'] == by_layer[name]['out_before'], name
    assert int(by_layer['fc2']['in_after']) < int(by_layer['fc2']['in_before'])  # fc1 lost neurons
    assert by_layer['depthwise']['status'] == 'pruned with expand'


def test_plan_rebuilds_the_pruned_architecture(run):
    rebuilt = apply_pruning_plan(_original_model(run), _plan(run), torch.randn(1, 3, 12, 12))
    pruned = _pruned_model(run)
    rebuilt.load_state_dict(pruned.state_dict(), strict=True)
    x = torch.randn(4, 3, 12, 12)
    with torch.no_grad():
        torch.testing.assert_close(rebuilt.eval()(x), pruned(x))


@pytest.mark.parametrize('ignore_layers', [[10], [-1], ['3']])
def test_ignore_layers_must_be_layer_indices(ignore_layers):
    model = ModelEntity(model=TinyNet(), model_path=None)  # 10 layers: indices 0 to 9
    with pytest.raises(ValueError, match='layer indices from 0 to 9'):
        Pruner(model, _data(0), _data(1), retrain_epochs=1, ignore_layers=ignore_layers)


def test_ignored_layers_are_not_pruned():
    model = TinyNet()
    pruner = Pruner(ModelEntity(model=model, model_path=None), _data(0), _data(1), retrain_epochs=1,
                    ignore_layers=[3])
    example = torch.randn(1, 3, 12, 12)
    rules = pruner._layer_rules(ModelStructure(model, example), execution_order(model, pruner.layers, example))
    assert not rules[3].prunable and rules[3].reason == 'ignored (--ignore_layers)'  # expand
    assert [idx for idx, rule in rules.items() if rule.prunable] == [1, 5, 6, 7, 8]


class HeadFirstNet(nn.Module):
    """Defines its classifier first and an unused layer second; the forward pass runs conv1, conv2, conv3, fc."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 3)
        self.unused = nn.Linear(10, 10)
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 12, 3, padding=1)
        self.conv3 = nn.Conv2d(12, 10, 3, padding=1)

    def forward(self, x):
        x = torch.relu(self.conv3(torch.relu(self.conv2(torch.relu(self.conv1(x))))))
        return self.fc(torch.flatten(nn.functional.adaptive_avg_pool2d(x, 1), 1))


def _prune_in(tmp_path, monkeypatch, model, **settings):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config, 'output_dir', '')
    torch.manual_seed(0)
    Pruner(ModelEntity(model=model, model_path=None), _data(0), _data(1), **settings).prune()
    return config.get_output_dir()


def test_layers_are_pruned_in_forward_order(tmp_path, monkeypatch):
    """First and last are the first and last layers the forward pass runs, not the first and last defined."""
    visited = []
    layer_pruner_class = pruner_module._layer_pruner_class

    def spy(layer):
        visited.append(layer)
        return layer_pruner_class(layer)

    monkeypatch.setattr(pruner_module, '_layer_pruner_class', spy)
    model = HeadFirstNet()
    run_dir = _prune_in(tmp_path, monkeypatch, model, retrain_epochs=0)

    entries = _plan(run_dir)['layers']
    assert [e['layer'] for e in entries] == ['conv1', 'conv2', 'conv3', 'fc', 'unused']
    status = {e['layer']: (e['status'], e.get('reason')) for e in entries}
    assert status['conv1'] == ('not pruned', 'first layer')
    assert status['conv2'][0] == 'pruned' and status['conv3'][0] == 'pruned'
    assert status['fc'] == ('not pruned', 'last layer')
    assert status['unused'] == ('not pruned', 'not used in the forward pass')
    assert visited == [model.conv2, model.conv3]


def test_settings_reach_the_selection_and_the_fine_tuning(tmp_path, monkeypatch):
    import cosep.pruning.pruners.layer_pruner as layer_pruner_module
    selections, trainings, generators = [], [], []
    select, train = layer_pruner_module.select_optimal_components, pruner_module.train_model

    def spy_select(graph_space, weights, num_components, poly_degree=2, random_state=None):
        profiles = graph_space['profiles']
        selections.append((poly_degree, profiles.shape[1], type(random_state)))
        return select(graph_space, weights, num_components, poly_degree, random_state)

    def spy_train(*args, **kwargs):
        trainings.append({k: kwargs[k] for k in ('num_epochs', 'learning_rate', 'momentum', 'weight_decay',
                                                 'nesterov', 'augment')})
        generators.append(kwargs['generator'])
        train(*args, **kwargs)

    monkeypatch.setattr(layer_pruner_module, 'select_optimal_components', spy_select)
    monkeypatch.setattr(pruner_module, 'train_model', spy_train)
    _prune_in(tmp_path, monkeypatch, TinyNet(), retrain_epochs=1, poly_degree=3, max_classes=2,
              learning_rate=0.05, momentum=0.5, weight_decay=0.0, nesterov=True, augment=True)

    assert selections == [(3, 1, np.random.RandomState)] * len(PRUNED)  # 2 of the 3 classes kept: one pair
    assert trainings == [dict(num_epochs=1, learning_rate=0.05, momentum=0.5, weight_decay=0.0,
                              nesterov=True, augment=True)] * len(PRUNED)
    # One generator for the whole run's augmentation
    assert isinstance(generators[0], torch.Generator) and all(g is generators[0] for g in generators)


@pytest.mark.parametrize('settings, message', [
    (dict(poly_degree=1), 'poly_degree'), (dict(poly_degree=2.5), 'poly_degree'),
    (dict(max_classes=1), 'max_classes'), (dict(max_classes=-3), 'max_classes'),
    (dict(seed=-1), 'seed'), (dict(seed=1.5), 'seed'), (dict(seed=True), 'seed'),
    (dict(augment=1), 'augment'), (dict(augment='yes'), 'augment'), (dict(augment=None), 'augment'),
])
def test_invalid_settings_are_rejected(settings, message):
    with pytest.raises(ValueError, match=message):
        Pruner(ModelEntity(model=TinyNet(), model_path=None), _data(0), _data(1), **settings)


def test_defaults_are_the_papers():
    pruner = Pruner(ModelEntity(model=TinyNet(), model_path=None), _data(0), _data(1))
    assert (pruner.retrain_epochs, pruner.poly_degree, pruner.max_classes, pruner.seed) == (2, 2, 100, 0)
    assert pruner.fine_tuning == dict(learning_rate=0.001, momentum=0.9, weight_decay=5e-4, nesterov=False,
                                      augment=False)


def _run_with_seed(tmp_path, monkeypatch, seed, retrain_epochs=1, **settings):
    """Prune TinyNet, always from the same initial weights, in a new folder; return the plan and the pruned weights."""
    folder = tmp_path / f'run{len(os.listdir(tmp_path))}'
    folder.mkdir()
    torch.manual_seed(0)
    model = TinyNet()
    run_dir = _prune_in(folder, monkeypatch, model, retrain_epochs=retrain_epochs, seed=seed, **settings)
    return _plan(run_dir), _pruned_model(run_dir).state_dict()


def test_the_same_seed_gives_the_same_pruned_model(tmp_path, monkeypatch):
    """Same seed and settings: the same channels and, after fine-tuning, the same weights. The run neither reads nor
    changes NumPy's global generator, so whatever the caller did with it does not matter.
    """
    plans, weights = [], []
    for global_seed in (1, 2):
        np.random.seed(global_seed)
        state = np.random.get_state()
        plan, state_dict = _run_with_seed(tmp_path, monkeypatch, seed=7)
        np.testing.assert_equal(np.random.get_state(), state)
        plans.append(plan['layers'])
        weights.append(state_dict)
    assert plans[0] == plans[1]
    assert weights[0].keys() == weights[1].keys()
    for name in weights[0]:
        assert torch.equal(weights[0][name], weights[1][name]), name


def test_the_plan_records_the_settings(tmp_path, monkeypatch):
    plan, _ = _run_with_seed(tmp_path, monkeypatch, seed=3, retrain_epochs=0)
    assert plan['settings'] == dict(seed=3, poly_degree=2, max_classes=100, retrain_epochs=0, learning_rate=0.001,
                                    momentum=0.9, weight_decay=5e-4, nesterov=False, augment=False)
    plan, _ = _run_with_seed(tmp_path, monkeypatch, seed=3, retrain_epochs=0, augment=True)
    assert plan['settings']['augment'] is True


def _same_weights(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[name], b[name]) for name in a)


def test_only_the_fine_tuning_batches_are_augmented(tmp_path, monkeypatch):
    """The augmentation runs once per fine-tuning batch and nowhere else. Without fine-tuning, augment=True gives
    exactly the model of augment=False: the class statistics read the training data as they are.
    """
    import cosep.utils.model_utils as model_utils
    calls = []
    augment_batch = model_utils.augment_batch

    def spy(images, generator=None, **kwargs):
        calls.append(len(images))
        return augment_batch(images, generator, **kwargs)

    monkeypatch.setattr(model_utils, 'augment_batch', spy)
    plan_on, weights_on = _run_with_seed(tmp_path, monkeypatch, seed=0, retrain_epochs=0, augment=True)
    plan_off, weights_off = _run_with_seed(tmp_path, monkeypatch, seed=0, retrain_epochs=0)
    assert calls == []
    assert plan_on['layers'] == plan_off['layers'] and _same_weights(weights_on, weights_off)

    _run_with_seed(tmp_path, monkeypatch, seed=0, retrain_epochs=1, augment=True)
    assert calls == [24] * (3 * len(PRUNED))  # after each pruned layer, one epoch: 3 batches of 24 images


def test_augmented_runs_repeat_with_the_same_seed(tmp_path, monkeypatch):
    """Same seed: the same crops and flips, so the same weights; and they differ from a run without augmentation."""
    first = _run_with_seed(tmp_path, monkeypatch, seed=7, augment=True)
    again = _run_with_seed(tmp_path, monkeypatch, seed=7, augment=True)
    plain = _run_with_seed(tmp_path, monkeypatch, seed=7)
    assert first[0]['layers'] == again[0]['layers'] and _same_weights(first[1], again[1])
    assert not _same_weights(first[1], plain[1])


class SignalNet(nn.Module):
    """A Conv1d network on 1-D signals (N, C, L)."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(2, 8, 3, padding=1)
        self.conv2 = nn.Conv1d(8, 8, 3, padding=1)
        self.fc = nn.Linear(8, 3)

    def forward(self, x):
        x = torch.relu(self.conv2(torch.relu(self.conv1(x))))
        return self.fc(x.mean(-1))


def test_augmentation_needs_images(tmp_path, monkeypatch):
    """Shifting and flipping are for images: with other inputs, augment=True stops before any layer is pruned."""
    labels = torch.arange(3).repeat_interleave(8)
    loader = DataLoader(TensorDataset(torch.randn(len(labels), 2, 16), labels), batch_size=8)
    data = DataEntity(data_loader=loader, num_classes=3, data_path=None, total_samples=len(labels),
                      labels=labels.numpy(), unique_label_indices=np.arange(3))
    monkeypatch.setattr(pruner_module, '_layer_pruner_class',
                        lambda layer: pytest.fail('a layer was pruned'))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config, 'output_dir', '')
    with pytest.raises(ValueError, match='image inputs'):
        Pruner(ModelEntity(model=SignalNet(), model_path=None), data, data, augment=True).prune()


class SmallLayersNet(nn.Module):
    """A chain of convolutions with 6 (first layer), 2, 4, 5 and 8 output channels, then the classifier."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 3, padding=1)
        self.conv2 = nn.Conv2d(6, 2, 3, padding=1)
        self.conv3 = nn.Conv2d(2, 4, 3, padding=1)
        self.conv4 = nn.Conv2d(4, 5, 3, padding=1)
        self.conv5 = nn.Conv2d(5, 8, 3, padding=1)
        self.fc = nn.Linear(8, 3)

    def forward(self, x):
        for conv in (self.conv1, self.conv2, self.conv3, self.conv4, self.conv5):
            x = torch.relu(conv(x))
        return self.fc(torch.flatten(nn.functional.adaptive_avg_pool2d(x, 1), 1))


def test_layers_with_fewer_than_five_components_are_kept_whole(tmp_path, monkeypatch):
    """k* is chosen among k = 2, ..., N-1 and the knee needs three candidates: layers with 2 or 4 channels stay
    whole (a 2-channel layer used to stop the run in t-SNE), layers with 5 or more are pruned.
    """
    run_dir = _prune_in(tmp_path, monkeypatch, SmallLayersNet(), retrain_epochs=0)
    status = {e['layer']: (e['status'], e.get('reason')) for e in _plan(run_dir)['layers']}
    too_few = 'fewer than 5 components, too few to choose the number of groups'
    assert status['conv2'] == ('not pruned', too_few)
    assert status['conv3'] == ('not pruned', too_few)
    assert status['conv4'][0] == 'pruned' and status['conv5'][0] == 'pruned'
    assert status['conv1'] == ('not pruned', 'first layer') and status['fc'] == ('not pruned', 'last layer')


def test_plots_go_to_this_run_and_show_the_kept_weights_before_fine_tuning(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config, 'output_dir', '')
    monkeypatch.setattr(config, 'verbose', True)
    events = []
    plot, train = pruner_module.plot_weight_distribution, pruner_module.train_model

    def spy_plot(layer, idx, before_prune):
        events.append(('plot', idx, before_prune, num_components(layer)))
        plot(layer, idx, before_prune)

    def spy_train(*args, **kwargs):
        events.append(('train',))
        train(*args, **kwargs)

    monkeypatch.setattr(pruner_module, 'plot_weight_distribution', spy_plot)
    monkeypatch.setattr(pruner_module, 'train_model', spy_train)
    run_dir = config.get_output_dir()
    later_run = os.path.join('outputs', 'a_later_run')
    os.makedirs(later_run)  # the most recently created output folder now belongs to another run

    torch.manual_seed(0)
    Pruner(ModelEntity(model=TinyNet(), model_path=None), _data(0), _data(1), retrain_epochs=1).prune()

    expected = []
    for idx, entry in enumerate(_plan(run_dir)['layers']):
        if entry['status'] == 'pruned':
            expected += [('plot', idx, True, entry['components_before']),
                         ('plot', idx, False, entry['components_after']), ('train',)]
    assert events == expected

    assert os.listdir(later_run) == []
    plots = os.path.join(run_dir, 'plots')
    for idx, entry in enumerate(_plan(run_dir)['layers']):
        if entry['status'] == 'pruned':
            kind = 'Linear' if entry['type'] == 'Linear' else 'Conv'
            files = os.listdir(os.path.join(plots, f'{kind} layer #{idx}'))
            assert {'Weight Distribution Before Pruning.png', 'Weight Distribution After Pruning.png',
                    'Separability space.png'} <= set(files)


@pytest.mark.parametrize('with_log_file', [False, True])
def test_output_summary_lists_only_the_files_written(tmp_path, monkeypatch, with_log_file):
    """Used from Python without init_logger(), a run writes no log.log, and its closing summary must not list one."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config, 'output_dir', '')
    monkeypatch.setattr(config, 'verbose', False)
    if with_log_file:
        config.init_logger(verbose=False)
    messages = []
    logger.add(lambda message: messages.append(message.record['message']), level='INFO')
    try:
        torch.manual_seed(0)
        Pruner(ModelEntity(model=TinyNet(), model_path=None), _data(0), _data(1), retrain_epochs=0).prune()
    finally:
        logger.remove()
        logger.add(sys.stderr)

    listed = {line.split(':')[0].strip(' -/') for line in messages if line.startswith('  - ')}
    written = set(os.listdir(config.get_output_dir()))
    assert listed == written
    assert ('log.log' in written) == with_log_file
