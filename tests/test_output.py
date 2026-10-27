"""Tests for run output folders and plots."""
import os
from datetime import datetime

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from matplotlib.figure import Figure

import cosep.config as config
from cosep.utils.plot_utils import plot_feature_space, save_plot


def test_runs_started_in_the_same_second_get_different_folders(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    class FrozenClock:
        @staticmethod
        def now():
            return datetime(2026, 1, 2, 3, 4, 5)

    monkeypatch.setattr(config, 'datetime', FrozenClock)
    folders = [config._new_output_dir() for _ in range(3)]
    first = os.path.join('outputs', '20260102_030405')
    assert folders == [first, first + '_2', first + '_3']
    assert all(os.path.isdir(folder) for folder in folders)


def test_a_plot_that_cannot_be_saved_is_reported(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config, 'output_dir', '')

    def fail(*args, **kwargs):
        raise OSError('disk full')

    monkeypatch.setattr(Figure, 'savefig', fail)
    figure = plt.figure()
    warnings = []
    sink = logger.add(lambda message: warnings.append(message.record['message']), level='WARNING')
    try:
        save_plot(figure, 'Conv layer #1', 'test')
    finally:
        logger.remove(sink)
    assert any('disk full' in message for message in warnings)
    assert figure.number not in plt.get_fignums()  # closed even though saving failed


def test_feature_space_points_are_coloured_by_mean_jm(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config, 'output_dir', '')
    monkeypatch.setattr(config, 'verbose', True)
    colours = {}
    scatter = matplotlib.axes.Axes.scatter

    def spy(self, x, y, **kwargs):
        colours.update(kwargs)
        return scatter(self, x, y, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, 'scatter', spy)
    rng = np.random.default_rng(0)
    profiles = rng.random((6, 10))
    plot_feature_space(rng.random((6, 2)), 3, 'Conv2d', profiles=profiles)

    np.testing.assert_allclose(colours['c'], profiles.mean(axis=1))
    assert os.path.isfile(os.path.join(config.get_output_dir(), 'plots', 'Conv layer #3', 'Separability space.png'))
    plot_feature_space(rng.random((6, 2)), 4, 'Linear')
    assert os.path.isfile(os.path.join(config.get_output_dir(), 'plots', 'Linear layer #4', 'Separability space.png'))
