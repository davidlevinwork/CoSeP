"""Tests for the separability profiles (paper, Eqs. 1-7) and class screening (Appendix A, Eqs. 17-18)."""
from itertools import combinations

import numpy as np
import pytest
import torch
import torch.nn as nn
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, TensorDataset

from cosep.data_types.class_statistics import ClassStatistics
from cosep.data_types.data_entity import DataEntity
from cosep.data_types.model_entity import ModelEntity
from cosep.graph.separability import jm_distance, one_vs_rest_scores, screen_classes, separability_profiles
from cosep.pruning.pruners import Conv1dLayerPruner, Conv2dLayerPruner, LinearLayerPruner
from cosep.pruning.pruners.layer_pruner import LayerPruner
from cosep.pruning.structure import ModelStructure
from cosep.utils.layer_utils import collect_class_statistics
from cosep.utils.model_utils import get_model_layers


def _jm_from_samples(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Eqs. 4-6 for two sets of samples (along axis 0), with the code's floor of 1e-5 on the standard deviation."""
    std_p, std_q = np.maximum(p.std(axis=0), 1e-5), np.maximum(q.std(axis=0), 1e-5)
    bd = (0.25 * (p.mean(axis=0) - q.mean(axis=0)) ** 2 / (std_p ** 2 + std_q ** 2)
          + 0.5 * np.log((std_p ** 2 + std_q ** 2) / (2 * std_p * std_q)))
    return 2.0 * (1.0 - np.exp(-bd))


def _paper_profiles(activations: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """The paper's profiles, computed directly from every activation (reference).

    Eq. 1: average each component's output over its positions (one value per sample; a linear neuron's output as is,
    Eq. 2). Eqs. 3-6: JM distance between the two classes' values. Eq. 7: one value per class pair, pairs in order.
    """
    z = activations.reshape(activations.shape[0], activations.shape[1], -1).mean(axis=2)
    return np.stack([_jm_from_samples(z[labels == c1], z[labels == c2])
                     for c1, c2 in combinations(np.unique(labels), 2)], axis=1)


def _statistics(values: np.ndarray, labels: np.ndarray) -> ClassStatistics:
    """ClassStatistics of one value per component and sample (samples, components)."""
    classes = np.unique(labels)
    return ClassStatistics(
        classes=classes,
        counts=np.array([(labels == c).sum() for c in classes]),
        means=np.stack([values[labels == c].mean(axis=0) for c in classes]),
        variances=np.stack([values[labels == c].var(axis=0) for c in classes]),
    )


class TestJMDistance:
    """The elementwise JM distance of Eqs. 5-6."""

    def test_matches_the_closed_form_on_samples(self, sample_distributions):
        p, q = sample_distributions
        expected = _jm_from_samples(p, q)
        assert jm_distance(p.mean(), p.var(), q.mean(), q.var()) == pytest.approx(expected, rel=1e-12)

    def test_identical_classes_give_zero(self):
        jm = jm_distance(np.array([1.0]), np.array([0.5]), np.array([1.0]), np.array([0.5]))
        assert jm[0] == pytest.approx(0.0, abs=1e-15)

    def test_constant_outputs_do_not_divide_by_zero(self):
        jm = jm_distance(np.array([0.0, 1.0]), np.zeros(2), np.array([0.0, 3.0]), np.zeros(2))
        assert jm.tolist() == [0.0, 2.0]


class TestSeparabilityProfiles:
    """The profile of every component: its JM distance for every class pair (Eq. 7)."""

    @pytest.mark.parametrize('shape', [(5,), (4, 7), (3, 5, 6)])  # Linear, Conv1d, Conv2d outputs
    def test_equal_to_the_paper(self, shape):
        rng = np.random.default_rng(0)
        labels = np.repeat([0, 3, 4, 9], 15)
        activations = rng.normal(size=(len(labels), *shape)) + labels.reshape(-1, *[1] * len(shape)) * 0.1
        z = activations.reshape(len(labels), shape[0], -1).mean(axis=2)

        profiles = separability_profiles(_statistics(z, labels))

        assert profiles.shape == (shape[0], 6)  # 6 pairs of 4 classes, one value each
        np.testing.assert_allclose(profiles, _paper_profiles(activations, labels), rtol=1e-10, atol=1e-12)

    def test_from_a_layer_equal_to_the_paper(self):
        """End to end: statistics collected from a layer give the paper's profiles of its stored outputs."""
        torch.manual_seed(0)
        model = nn.Sequential(nn.Conv2d(3, 6, 3, padding=1), nn.ReLU(), nn.Flatten(), nn.Linear(6 * 5 * 5, 3))
        x = torch.randn(48, 3, 5, 5)
        labels = torch.arange(48) % 3
        loader = DataLoader(TensorDataset(x, labels), batch_size=10)

        stored = []
        conv = list(get_model_layers(model))[0]
        hook = conv.register_forward_hook(lambda module, inputs, output: stored.append(output.detach().clone()))
        with torch.no_grad():
            for inputs, _ in loader:
                model.eval()(inputs)
        hook.remove()
        expected = _paper_profiles(torch.cat(stored).double().numpy(), labels.numpy())

        profiles = separability_profiles(collect_class_statistics(model, loader, layer_index=0))
        assert profiles.shape == (6, 3)
        np.testing.assert_allclose(profiles, expected, rtol=1e-9, atol=1e-12)

    def test_one_class_is_not_enough(self):
        stats = ClassStatistics(np.array([1]), np.array([4]), np.zeros((1, 3)), np.ones((1, 3)))
        with pytest.raises(ValueError, match='two classes'):
            separability_profiles(stats)

    def test_statistics_per_position_are_rejected(self):
        stats = ClassStatistics(np.array([0, 1]), np.array([4, 4]), np.zeros((2, 3, 5)), np.ones((2, 3, 5)))
        with pytest.raises(ValueError, match=r'shape \(classes, components\)'):
            separability_profiles(stats)


def _brute_force_class_scores(values: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Eq. 17 from the samples: pool every other class, JM per component, mean over components (reference)."""
    return np.array([_jm_from_samples(values[labels == t], values[labels != t]).mean() for t in np.unique(labels)])


def _many_classes(num_classes: int, components: int = 6, seed: int = 0):
    """Samples of many classes; a class's offset sets how well it separates from the others."""
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(num_classes) * 2 + 1, 12)  # labels need not be 0..C-1
    offsets = rng.normal(scale=2.0, size=(num_classes, components)) * rng.uniform(0, 1, size=(num_classes, 1))
    values = rng.normal(size=(len(labels), components)) * rng.uniform(0.5, 2.0, components) + 50.0
    values += np.repeat(offsets, 12, axis=0)
    return values, labels


class TestClassScreening:
    """Appendix A: with more than M classes, each layer keeps the M classes it separates best from the rest."""

    def test_scores_equal_the_one_vs_rest_jm_of_the_samples(self):
        values, labels = _many_classes(12)
        scores = one_vs_rest_scores(_statistics(values, labels))
        np.testing.assert_allclose(scores, _brute_force_class_scores(values, labels), rtol=1e-9, atol=1e-12)

    def test_keeps_the_top_m_classes_in_their_order(self):
        values, labels = _many_classes(12)
        stats = _statistics(values, labels)
        screened = screen_classes(stats, 5)

        best = np.argsort(-_brute_force_class_scores(values, labels))[:5]
        np.testing.assert_array_equal(screened.classes, np.sort(stats.classes[best]))
        keep = np.isin(stats.classes, screened.classes)
        np.testing.assert_array_equal(screened.counts, stats.counts[keep])
        np.testing.assert_array_equal(screened.means, stats.means[keep])
        np.testing.assert_array_equal(screened.variances, stats.variances[keep])

    def test_profiles_use_the_pairs_of_the_kept_classes(self):
        values, labels = _many_classes(12)
        screened = screen_classes(_statistics(values, labels), 5)
        profiles = separability_profiles(screened)

        assert profiles.shape == (6, 10)  # 5 classes: 10 pairs
        subset = np.isin(labels, screened.classes)
        np.testing.assert_allclose(profiles, separability_profiles(_statistics(values[subset], labels[subset])),
                                   rtol=1e-12)

    @pytest.mark.parametrize('max_classes', [None, 0, 12, 100])
    def test_no_screening_up_to_m_classes(self, max_classes):
        values, labels = _many_classes(12)
        stats = _statistics(values, labels)
        assert screen_classes(stats, max_classes) is stats

    def test_ties_keep_the_first_classes(self):
        stats = ClassStatistics(np.arange(4), np.full(4, 10), np.array([[0.0], [1.0], [0.0], [1.0]]),
                                np.ones((4, 1)))
        assert screen_classes(stats, 2).classes.tolist() == [0, 1]

    def test_m_below_two_is_rejected(self):
        values, labels = _many_classes(5)
        with pytest.raises(ValueError, match='at least 2'):
            screen_classes(_statistics(values, labels), 1)

    def test_the_layer_pruner_builds_profiles_from_the_screened_classes(self, monkeypatch):
        import cosep.pruning.pruners.layer_pruner as layer_pruner_module
        torch.manual_seed(0)
        model = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.Flatten(), nn.Linear(8 * 4 * 4, 6)).eval()
        labels = torch.arange(60) % 6
        loader = DataLoader(TensorDataset(torch.randn(60, 3, 4, 4) + labels.view(-1, 1, 1, 1) * 0.3, labels),
                            batch_size=16)
        data = DataEntity(data_loader=loader, num_classes=6, data_path=None, total_samples=60, labels=labels.numpy(),
                          unique_label_indices=np.arange(6))
        seen = {}

        def select(graph_space, weights, num_components, poly_degree=2, random_state=None):
            seen['profiles'], seen['poly_degree'] = graph_space['profiles'].shape, poly_degree
            seen['random_state'] = type(random_state)
            return list(range(num_components // 2))

        monkeypatch.setattr(layer_pruner_module, 'select_optimal_components', select)
        structure = ModelStructure(model, torch.randn(1, 3, 4, 4))
        pruner = Conv2dLayerPruner(ModelEntity(model=model, model_path=None), data, 0, structure,
                                   poly_degree=3, max_classes=4)
        pruner.prune()
        assert seen == {'profiles': (8, 6), 'poly_degree': 3,  # 4 of 6 classes: 6 pairs
                        'random_state': np.random.RandomState}  # the layer's own generator
        assert pruner.kept_components == [0, 1, 2, 3]


def test_tsne_rejects_non_finite_profiles():
    """NaN or Inf in the profiles is an error, not 'no separability'."""
    profiles = np.random.default_rng(0).random((10, 6))
    profiles[2, 3] = np.nan
    with pytest.raises(ValueError, match='NaN or Inf'):
        LayerPruner._reduce_dimension(profiles)


def test_tsne_maps_single_value_profiles():
    """Two classes give one value per component; t-SNE still maps the components to 2-D."""
    profiles = np.random.default_rng(0).random((12, 1))
    assert LayerPruner._reduce_dimension(profiles).shape == (12, 2)


def test_tsne_map_is_repeatable_with_a_generator():
    """The same generator state gives the same map, including the noise added to all-zero profiles, and NumPy's
    global generator is neither read nor changed.
    """
    profiles = np.random.default_rng(0).random((20, 6))
    profiles[[3, 7]] = 0.0
    state = np.random.get_state()
    first = LayerPruner._reduce_dimension(profiles.copy(), np.random.RandomState(3))
    second = LayerPruner._reduce_dimension(profiles.copy(), np.random.RandomState(3))
    np.testing.assert_array_equal(first, second)
    np.testing.assert_equal(np.random.get_state(), state)


@pytest.mark.parametrize('kind', ['Conv1d', 'Conv2d', 'Linear'])
def test_each_layer_runs_tsne_once(kind, monkeypatch):
    """One t-SNE map per layer: no unused second map, and no maps per class pair."""
    torch.manual_seed(0)
    models = {
        'Conv1d': (nn.Sequential(nn.Conv1d(3, 8, 3, padding=1), nn.Flatten(), nn.Linear(48, 4)), (3, 6)),
        'Conv2d': (nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.Flatten(), nn.Linear(288, 4)), (3, 6, 6)),
        'Linear': (nn.Sequential(nn.Flatten(), nn.Linear(108, 8), nn.Linear(8, 4)), (3, 6, 6)),
    }
    pruners = {'Conv1d': Conv1dLayerPruner, 'Conv2d': Conv2dLayerPruner, 'Linear': LinearLayerPruner}
    model, shape = models[kind]
    labels = torch.arange(40) % 4  # 6 class pairs
    loader = DataLoader(TensorDataset(torch.randn(40, *shape), labels), batch_size=10)
    data = DataEntity(data_loader=loader, num_classes=4, data_path=None, total_samples=40, labels=labels.numpy(),
                      unique_label_indices=np.arange(4))
    pruner = pruners[kind](ModelEntity(model=model.eval(), model_path=None), data, 0, None)
    pruner.statistics = collect_class_statistics(model, loader, layer_index=0)

    runs = []
    fit_transform = TSNE.fit_transform

    def counting(self, *args, **kwargs):
        runs.append(1)
        return fit_transform(self, *args, **kwargs)

    monkeypatch.setattr(TSNE, 'fit_transform', counting)
    space = pruner.create_layer_space()
    assert len(runs) == 1
    assert space['reduced_matrix'].shape == (8, 2)
    assert space['profiles'].shape == (8, 6)


def test_the_layer_space_needs_the_class_statistics():
    """create_layer_space() is called by prune(), after the class statistics are collected."""
    model = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.Flatten(), nn.Linear(288, 4))
    loader = DataLoader(TensorDataset(torch.randn(8, 3, 6, 6), torch.arange(8) % 4), batch_size=4)
    data = DataEntity(data_loader=loader, num_classes=4, data_path=None, total_samples=8,
                      labels=(np.arange(8) % 4), unique_label_indices=np.arange(4))
    pruner = Conv2dLayerPruner(ModelEntity(model=model, model_path=None), data, 0, None)
    with pytest.raises(RuntimeError, match='class statistics'):
        pruner.create_layer_space()
