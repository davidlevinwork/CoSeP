"""Tests for clustering utilities."""
import numpy as np
import pytest
from kmedoids import fasterpam
from kneed import KneeLocator
from sklearn.metrics.pairwise import pairwise_distances

from cosep.graph.clustering import (
    MIN_COMPONENTS,
    find_optimal_weighted_medoids,
    get_knee,
    kmedoids_fasterpam,
    select_optimal_components,
)


class TestKMedoidsFasterPAM:
    """Test k-medoids clustering function."""

    def test_basic_clustering(self, sample_clustering_data):
        """Should cluster data into k groups."""
        result = kmedoids_fasterpam(sample_clustering_data, k=3)

        assert 'labels' in result
        assert 'medoids' in result
        assert 'medoids_loc' in result
        assert len(result['labels']) == len(sample_clustering_data)
        assert len(result['medoids']) == 3

    def test_labels_range(self, sample_clustering_data):
        """Labels should be in range [0, k-1]."""
        result = kmedoids_fasterpam(sample_clustering_data, k=3)
        assert np.all(result['labels'] >= 0)
        assert np.all(result['labels'] < 3)

    def test_medoids_are_data_points(self, sample_clustering_data):
        """Medoids should be actual data points."""
        result = kmedoids_fasterpam(sample_clustering_data, k=3)
        for medoid_idx in result['medoids']:
            assert medoid_idx < len(sample_clustering_data)

    def test_precomputed_distances(self, sample_clustering_data):
        """Should accept precomputed distance matrix."""
        precomputed = pairwise_distances(sample_clustering_data)

        result = kmedoids_fasterpam(sample_clustering_data, k=3, precomputed_distances=precomputed)
        assert len(result['medoids']) == 3

    def test_same_algorithm_as_the_kmedoids_package_below_1000_points(self):
        """FasterPAM in its fixed processing order, from k medoids drawn from the generator: what kmedoids.fasterpam
        does by default below 1,000 points, so the seed changes only where the start comes from.
        """
        data = np.random.default_rng(0).normal(size=(300, 2))
        distances = pairwise_distances(data)
        for k in (3, 20, 100):
            np.random.seed(k)
            reference = fasterpam(distances, k)
            np.random.seed(k)
            ours = kmedoids_fasterpam(data, k, distances)
            np.testing.assert_array_equal(ours['medoids'], reference.medoids)
            np.testing.assert_array_equal(ours['labels'], reference.labels)

    def test_repeatable_with_a_generator(self, sample_clustering_data):
        """The same generator state gives the same groups, and NumPy's global generator is untouched."""
        state = np.random.get_state()
        first = kmedoids_fasterpam(sample_clustering_data, k=7, random_state=np.random.RandomState(4))
        second = kmedoids_fasterpam(sample_clustering_data, k=7, random_state=np.random.RandomState(4))
        np.testing.assert_array_equal(first['medoids'], second['medoids'])
        np.testing.assert_array_equal(first['labels'], second['labels'])
        np.testing.assert_equal(np.random.get_state(), state)

    def test_repeatable_above_1000_points(self):
        """From 1,000 points the kmedoids package switches to a parallel version that can return different medoids
        for the same seed; the selection always runs FasterPAM on one thread.
        """
        data = np.random.default_rng(1).normal(size=(1200, 2))
        distances = pairwise_distances(data)
        runs = [kmedoids_fasterpam(data, 400, distances, random_state=np.random.RandomState(9))['medoids']
                for _ in range(3)]
        for medoids in runs[1:]:
            np.testing.assert_array_equal(medoids, runs[0])


class TestGetKnee:
    """k* from the MSS curve: Kneedle on a degree-p fit, and the middle candidate when the fit has no concave knee."""

    @staticmethod
    def _warnings():
        from loguru import logger
        messages = []
        sink = logger.add(lambda message: messages.append(message.record['message']), level='WARNING')
        return messages, lambda: logger.remove(sink)

    def test_finds_knee(self):
        """Should find knee in concave curve."""
        x = list(range(1, 11))
        y = [1, 2, 3, 3.5, 3.8, 3.9, 3.95, 3.98, 3.99, 4.0]
        knee = get_knee(x, y)
        assert knee is not None
        assert 3 <= knee <= 6

    @pytest.mark.parametrize('poly_deg', [2, 3, 4, 5])
    @pytest.mark.parametrize('scale', [5, 20, 60])
    def test_knee_of_a_rising_curve_is_kneedles(self, poly_deg, scale):
        """For curves that rise and level off, the guard changes nothing: k* is Kneedle's knee of the fit."""
        x = np.arange(2, 128)
        y = 1 - np.exp(-(x - 2) / scale)
        expected = KneeLocator(list(x), list(y), curve='concave', direction='increasing',
                               interp_method='polynomial', polynomial_degree=poly_deg).knee
        assert expected is not None and expected > 2
        assert get_knee(list(x), list(y), poly_deg) == expected

    def test_higher_degree_keeps_fewer(self):
        """The degree p sets the overall level: on the same curve, higher degrees give a smaller k*."""
        x = list(range(2, 128))
        y = list(1 - np.exp(-(np.array(x) - 2) / 20))
        knees = [get_knee(x, y, p) for p in (2, 3, 4, 5)]
        assert knees == sorted(knees, reverse=True) and knees[0] > knees[-1]

    def test_linear_no_knee(self):
        """A straight line has no knee: the middle candidate, with a warning."""
        x = list(range(1, 11))
        y = list(range(1, 11))
        messages, stop = self._warnings()
        try:
            knee = get_knee(x, y)
        finally:
            stop()
        assert knee == 6  # x[10 // 2]
        assert any('no knee' in m and 'k = 6' in m for m in messages)

    @pytest.mark.parametrize('curve', ['bends upward', 'dips then climbs'])
    def test_fit_that_bends_upward_gives_the_middle_not_two(self, curve):
        """Kneedle returns the first candidate (k = 2) for these fits; the guard uses the middle candidate instead."""
        x = np.arange(2, 512)
        t = (x - 2) / 509
        y = 0.25 + 0.7 * t ** 2 if curve == 'bends upward' else 0.3 - 0.2 * np.sin(0.4 * np.pi * t) + 0.8 * t ** 2
        plain = KneeLocator(list(x), list(y), curve='concave', direction='increasing',
                            interp_method='polynomial', polynomial_degree=2).knee
        assert plain == 2  # what the code returned before the guard

        messages, stop = self._warnings()
        try:
            knee = get_knee(list(x), list(y))
        finally:
            stop()
        assert knee == x[len(x) // 2] == 257
        assert any('bends upward' in m for m in messages)

    @pytest.mark.parametrize('x, expected', [([2], 2), ([2, 3], 3)])
    def test_too_few_candidates(self, x, expected):
        assert get_knee(x, [0.5] * len(x)) == expected

    def test_no_candidates_is_an_error(self):
        with pytest.raises(ValueError, match='no candidates'):
            get_knee([], [])


class TestFindOptimalWeightedMedoids:
    """Test weighted medoid selection."""

    def test_selects_highest_weight(self):
        """Should select highest weight point in each cluster."""
        weights = np.array([1.0, 5.0, 2.0, 3.0, 8.0, 1.0])
        labels = np.array([0, 0, 0, 1, 1, 1])
        medoids = np.array([0, 3])

        result = find_optimal_weighted_medoids(weights, labels, medoids)

        # Cluster 0: indices 0,1,2 with weights 1,5,2 -> select 1
        # Cluster 1: indices 3,4,5 with weights 3,8,1 -> select 4
        assert result[0] == 1
        assert result[1] == 4


class TestSelectOptimalComponents:
    """Test optimal component selection."""

    def test_returns_list(self, sample_clustering_data):
        """Should return list of indices."""
        graph_space = {
            'reduced_matrix': sample_clustering_data
        }
        weights = np.random.rand(len(sample_clustering_data))

        result = select_optimal_components(
            graph_space, weights, num_components=len(sample_clustering_data)
        )

        assert isinstance(result, list)
        assert all(isinstance(i, (int, np.integer)) for i in result)

    def test_indices_in_range(self, sample_clustering_data):
        """Selected indices should be valid."""
        graph_space = {
            'reduced_matrix': sample_clustering_data
        }
        weights = np.random.rand(len(sample_clustering_data))

        result = select_optimal_components(
            graph_space, weights, num_components=len(sample_clustering_data)
        )

        assert all(0 <= i < len(sample_clustering_data) for i in result)

    def test_keeps_the_heaviest_component_of_each_group(self, sample_clustering_data, monkeypatch):
        """One representative per group of the final k-medoids run: its component with the largest weight norm."""
        import cosep.graph.clustering as clustering
        runs = []
        fasterpam = clustering.kmedoids_fasterpam

        def recording(data, k, precomputed_distances=None, random_state=None):
            runs.append(fasterpam(data, k, precomputed_distances, random_state=random_state))
            return runs[-1]

        monkeypatch.setattr(clustering, 'kmedoids_fasterpam', recording)
        weights = np.random.default_rng(1).random(len(sample_clustering_data))
        result = select_optimal_components({'reduced_matrix': sample_clustering_data}, weights,
                                           num_components=len(sample_clustering_data))
        final = runs[-1]
        expected = [int(np.flatnonzero(final['labels'] == g)[np.argmax(weights[final['labels'] == g])])
                    for g in final['labels'][final['medoids']]]
        assert sorted(result) == sorted(expected)
        assert len(result) == len(final['medoids'])

    def test_passes_the_polynomial_degree_to_the_knee(self, sample_clustering_data, monkeypatch):
        import cosep.graph.clustering as clustering
        degrees = []

        def knee(x, y, poly_deg=2):
            degrees.append(poly_deg)
            return 4

        monkeypatch.setattr(clustering, 'get_knee', knee)
        result = select_optimal_components({'reduced_matrix': sample_clustering_data},
                                           np.ones(len(sample_clustering_data)),
                                           num_components=len(sample_clustering_data), poly_degree=4)
        assert degrees == [4] and len(result) == 4

    def test_needs_the_2d_map(self, sample_clustering_data):
        with pytest.raises(KeyError, match='reduced_matrix'):
            select_optimal_components({'profiles': sample_clustering_data}, np.ones(len(sample_clustering_data)),
                                      num_components=len(sample_clustering_data))

    def test_repeatable_with_a_seed(self, sample_clustering_data):
        """The same generator state gives the same selection, and NumPy's global generator is neither read nor
        changed.
        """
        weights = np.random.default_rng(2).random(len(sample_clustering_data))
        state = np.random.get_state()
        results = [select_optimal_components({'reduced_matrix': sample_clustering_data}, weights,
                                             num_components=len(sample_clustering_data),
                                             random_state=np.random.RandomState(5)) for _ in range(2)]
        assert results[0] == results[1]
        np.testing.assert_equal(np.random.get_state(), state)

    def test_one_generator_drives_the_whole_sweep(self, sample_clustering_data, monkeypatch):
        """An int seed becomes one generator for every k-medoids run, not the same start for every k."""
        import cosep.graph.clustering as clustering
        generators = []
        original = clustering.kmedoids_fasterpam

        def recording(data, k, precomputed_distances=None, random_state=None):
            generators.append(random_state)
            return original(data, k, precomputed_distances, random_state=random_state)

        monkeypatch.setattr(clustering, 'kmedoids_fasterpam', recording)
        select_optimal_components({'reduced_matrix': sample_clustering_data}, np.ones(len(sample_clustering_data)),
                                  num_components=len(sample_clustering_data), random_state=11)
        assert len(generators) > 2 and isinstance(generators[0], np.random.RandomState)
        assert all(g is generators[0] for g in generators)


def test_a_layer_needs_five_components():
    """k* is chosen among k = 2, ..., N-1, and a knee needs at least three candidates."""
    assert MIN_COMPONENTS == 5 and len(range(2, MIN_COMPONENTS - 1)) == 2 and len(range(2, MIN_COMPONENTS)) == 3


class TestNonFiniteValues:
    """NaN or Inf is an error, not a distance of 0 ('identical')."""

    def test_embedding_with_nan_raises(self, sample_clustering_data):
        data = sample_clustering_data.copy()
        data[3, 1] = np.nan
        with pytest.raises(ValueError, match='NaN or Inf'):
            select_optimal_components({'reduced_matrix': data}, np.ones(len(data)), num_components=len(data))

    def test_distances_with_inf_raise(self, sample_clustering_data):
        from sklearn.metrics.pairwise import pairwise_distances
        distances = pairwise_distances(sample_clustering_data)
        distances[0, 1] = np.inf
        with pytest.raises(ValueError, match='NaN or Inf'):
            kmedoids_fasterpam(sample_clustering_data, k=3, precomputed_distances=distances)
