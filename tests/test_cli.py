"""Tests for the command-line interface."""
import json
import os
import ssl
import sys

import pytest
import torch
from loguru import logger
from torch.utils.data import RandomSampler, SequentialSampler, TensorDataset

import cosep.config as config
import pruning_algorithm
from pruning_algorithm import parse_arguments
from tests.tiny_models import TinyNet

REQUIRED = ['--model_path', 'model.pt', '--train_data_path', 'train.pt', '--test_data_path', 'test.pt']


def test_importing_the_cli_keeps_certificate_checks():
    """The CLI module is imported above; HTTPS downloads must still verify certificates."""
    assert ssl._create_default_https_context is not ssl._create_unverified_context


@pytest.mark.parametrize('values, expected', [
    (['3,12'], [3, 12]),
    (['3', '12'], [3, 12]),
    (['12,3', '7'], [3, 7, 12]),
    (['5'], [5]),
    (['4,4'], [4]),
])
def test_ignore_layers_are_parsed_as_indices(values, expected):
    assert parse_arguments(REQUIRED + ['--ignore_layers', *values]).ignore_layers == expected


def test_ignore_layers_default_to_none():
    assert parse_arguments(REQUIRED).ignore_layers == []


@pytest.mark.parametrize('value', ['a', '3,x', '1.5'])
def test_ignore_layers_reject_non_integers(value):
    with pytest.raises(SystemExit):
        parse_arguments(REQUIRED + ['--ignore_layers', value])


def test_verbose_is_on_by_default_and_can_be_turned_off():
    assert parse_arguments(REQUIRED).verbose is True
    assert parse_arguments(REQUIRED + ['--verbose']).verbose is True
    assert parse_arguments(REQUIRED + ['--no-verbose']).verbose is False


def test_defaults_are_the_papers_settings():
    """Section 4.1 and Appendix C: p = 2, M = 100, 2 epochs on 25% of the data, batch 128; SGD at learning rate 0.001
    (the paper's 0.01 pulls an already-trained network away from its solution); no augmentation.
    """
    args = parse_arguments(REQUIRED)
    assert (args.poly_degree, args.max_classes, args.seed) == (2, 100, 0)
    assert (args.retrain_epochs, args.train_fraction, args.batch_size) == (2, 0.25, 128)
    assert (args.lr, args.momentum, args.weight_decay, args.nesterov) == (0.001, 0.9, 5e-4, False)
    assert args.augment is False


def test_settings_can_be_changed():
    args = parse_arguments(REQUIRED + ['--poly_degree', '4', '--max_classes', '0', '--retrain_epochs', '3',
                                       '--train_fraction', '1', '--batch_size', '256', '--lr', '0.01',
                                       '--momentum', '0.95', '--weight_decay', '1e-4', '--nesterov', '--seed', '7',
                                       '--augment'])
    assert (args.poly_degree, args.max_classes, args.seed) == (4, 0, 7)
    assert (args.retrain_epochs, args.train_fraction, args.batch_size) == (3, 1.0, 256)
    assert (args.lr, args.momentum, args.weight_decay, args.nesterov) == (0.01, 0.95, 1e-4, True)
    assert args.augment is True
    assert parse_arguments(REQUIRED + ['--augment', '--no-augment']).augment is False


@pytest.mark.parametrize('option, value', [
    ('--poly_degree', '1'), ('--poly_degree', '2.5'), ('--max_classes', '1'), ('--max_classes', '-1'),
    ('--retrain_epochs', '-1'), ('--train_fraction', '0'), ('--train_fraction', '1.5'), ('--batch_size', '0'),
    ('--lr', '0'), ('--momentum', '-0.1'), ('--weight_decay', '-1'), ('--seed', '-1'), ('--seed', '1.5'),
    ('--num_classes', '1'), ('--num_classes', 'ten'),
])
def test_invalid_values_are_rejected(option, value):
    with pytest.raises(SystemExit):
        parse_arguments(REQUIRED + [option, value])


def test_imagenet_and_dataset_files_are_exclusive():
    assert parse_arguments(['--model_path', 'model.pt', '--imagenet_path', 'imagenet']).imagenet_path == 'imagenet'
    with pytest.raises(SystemExit):
        parse_arguments(REQUIRED + ['--imagenet_path', 'imagenet'])
    with pytest.raises(SystemExit):
        parse_arguments(['--model_path', 'model.pt', '--train_data_path', 'train.pt'])


def test_nesterov_needs_momentum():
    with pytest.raises(SystemExit):
        parse_arguments(REQUIRED + ['--nesterov', '--momentum', '0'])


def test_metric_option_is_gone():
    """The pruners always use JM, so there is no --metric option."""
    with pytest.raises(SystemExit):
        parse_arguments(REQUIRED + ['--metric', 'wasserstein'])


def _save_inputs(tmp_path):
    torch.manual_seed(0)
    labels = torch.arange(3).repeat_interleave(24)
    images = torch.randn(len(labels), 3, 12, 12) + 0.5 * labels.view(-1, 1, 1, 1)
    torch.save(TensorDataset(images, labels), tmp_path / 'train.pt')
    torch.save(TensorDataset(images[::2], labels[::2]), tmp_path / 'test.pt')
    torch.save(TinyNet(), tmp_path / 'model.pt')


def _run_cli(monkeypatch, argv):
    monkeypatch.setattr(config, 'output_dir', '')
    monkeypatch.setattr(sys, 'argv', ['pruning_algorithm.py'] + argv)
    try:
        pruning_algorithm.main()
    finally:
        logger.remove()
        logger.add(sys.stderr)
    return config.get_output_dir()


def test_main_runs_with_the_given_settings(tmp_path, monkeypatch):
    """End to end: data, model and every setting reach the loaders, the selection and the fine-tuning."""
    _save_inputs(tmp_path)
    seen = {}
    real_pruner = pruning_algorithm.Pruner

    def spy(**kwargs):
        loader = kwargs['train_data'].data_loader
        seen.update(batch_size=loader.batch_size, train_sampler=type(loader.sampler),
                    test_sampler=type(kwargs['test_data'].data_loader.sampler),
                    train_samples=kwargs['train_data'].total_samples,
                    **{k: v for k, v in kwargs.items() if k not in ('model', 'train_data', 'test_data')})
        return real_pruner(**kwargs)

    monkeypatch.setattr(pruning_algorithm, 'Pruner', spy)
    monkeypatch.chdir(tmp_path)
    run_dir = _run_cli(monkeypatch, ['--model_path', 'model.pt', '--train_data_path', 'train.pt',
                                     '--test_data_path', 'test.pt', '--no-verbose', '--retrain_epochs', '1',
                                     '--train_fraction', '0.5', '--batch_size', '16', '--lr', '0.02',
                                     '--momentum', '0.8', '--weight_decay', '0.001', '--nesterov', '--poly_degree',
                                     '3', '--max_classes', '0', '--ignore_layers', '3', '--seed', '5', '--augment'])

    assert seen == dict(batch_size=16, train_sampler=RandomSampler, test_sampler=SequentialSampler,
                        train_samples=36, ignore_layers=[3], retrain_epochs=1, poly_degree=3, max_classes=0,
                        learning_rate=0.02, momentum=0.8, weight_decay=0.001, nesterov=True, seed=5, augment=True)
    for name in ['pruned_model.pt', 'pruning_plan.json', 'pruning_summary.csv', 'log.log']:
        assert os.path.isfile(os.path.join(run_dir, name)), name
    log = open(os.path.join(run_dir, 'log.log'), encoding='utf-8').read()
    assert 'Fine-tuning for 1 epochs (lr=0.02, momentum=0.8, wd=0.001, nesterov=True, augment=True)' in log
    assert 'Using 36/72 (50.0%) training samples (stratified)' in log
    settings = json.load(open(os.path.join(run_dir, 'pruning_plan.json')))['settings']
    assert (settings['seed'], settings['augment']) == (5, True)


def test_the_same_seed_gives_the_same_pruned_model(tmp_path, monkeypatch):
    """Two command-line runs with the same seed and settings: the same training sample and batch order, the same
    channels and the same weights; another seed draws another sample.
    """
    _save_inputs(tmp_path)
    monkeypatch.chdir(tmp_path)
    argv = ['--model_path', 'model.pt', '--train_data_path', 'train.pt', '--test_data_path', 'test.pt',
            '--no-verbose', '--retrain_epochs', '1', '--train_fraction', '0.5', '--batch_size', '8']
    results = []
    for seed in ('4', '4', '9'):
        torch.manual_seed(int(seed) + len(results))  # the caller's own PyTorch state must not matter
        run_dir = _run_cli(monkeypatch, argv + ['--seed', seed])
        results.append((json.load(open(os.path.join(run_dir, 'pruning_plan.json')))['layers'],
                        torch.load(os.path.join(run_dir, 'pruned_model.pt'), weights_only=False).state_dict()))
    (plan_a, weights_a), (plan_b, weights_b) = results[:2]
    assert plan_a == plan_b
    for name in weights_a:
        assert torch.equal(weights_a[name], weights_b[name]), name
    assert results[2][0] != plan_a or any(not torch.equal(weights_a[n], results[2][1][n]) for n in weights_a)
