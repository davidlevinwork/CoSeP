"""Tests for MSS (Mean Simplified Silhouette) calculation."""
import numpy as np
import pytest

from cosep.graph.mss import calc_mss_value


class TestCalcMSSValue:
    """Test MSS calculation function."""

    def test_perfect_clustering(self):
        """Perfect clustering should return 1.0."""
        # All points at their cluster centers
        space = np.array([[0, 0], [0, 0], [5, 5], [5, 5]])
        clustering = {
            'labels': np.array([0, 0, 1, 1]),
            'medoids_loc': np.array([[0, 0], [5, 5]])
        }

        result = calc_mss_value(space, clustering)
        assert result == 1.0

    def test_good_clustering(self, sample_clustering_data):
        """Well-separated clusters should have high MSS."""
        from cosep.graph.clustering import kmedoids_fasterpam

        clustering = kmedoids_fasterpam(sample_clustering_data, k=3)
        result = calc_mss_value(sample_clustering_data, clustering)

        # Well-separated clusters should have positive silhouette
        assert result > 0

    def test_returns_float(self, sample_clustering_data):
        """Should return a float value."""
        from cosep.graph.clustering import kmedoids_fasterpam

        clustering = kmedoids_fasterpam(sample_clustering_data, k=3)
        result = calc_mss_value(sample_clustering_data, clustering)

        assert isinstance(result, float)

    def test_bounded(self, sample_clustering_data):
        """MSS should be bounded between -1 and 1."""
        from cosep.graph.clustering import kmedoids_fasterpam

        clustering = kmedoids_fasterpam(sample_clustering_data, k=3)
        result = calc_mss_value(sample_clustering_data, clustering)

        assert -1 <= result <= 1

    def test_single_cluster(self):
        """Single point clusters should handle gracefully."""
        space = np.array([[0, 0], [10, 10]])
        clustering = {
            'labels': np.array([0, 1]),
            'medoids_loc': np.array([[0, 0], [10, 10]])
        }

        result = calc_mss_value(space, clustering)
        # Single point clusters have zero intra-distance
        assert result == 1.0


def _mss_reference(space, labels, medoids):
    """The MSS as documented, computed component by component (reference).

    a: distance to the own medoid (Eq. 9); b: MEAN distance to the other medoids; s = (b - a) / max(a, b) (Eq. 11);
    averaged over the components that are not on a medoid (a > 0).
    """
    scores = []
    for point, label in zip(space, labels):
        a = np.linalg.norm(point - medoids[label])
        if a == 0:
            continue
        b = np.mean([np.linalg.norm(point - m) for j, m in enumerate(medoids) if j != label])
        scores.append((b - a) / max(a, b))
    return float(np.mean(scores))


class TestMSSFormula:
    """The MSS that CoSeP uses: b is the mean distance to the other medoids, and medoids are left out."""

    @pytest.mark.parametrize('k', [2, 3, 5, 8])
    def test_equal_to_the_reference(self, sample_clustering_data, k):
        from cosep.graph.clustering import kmedoids_fasterpam
        clustering = kmedoids_fasterpam(sample_clustering_data, k=k)
        expected = _mss_reference(sample_clustering_data, clustering['labels'], clustering['medoids_loc'])
        assert calc_mss_value(sample_clustering_data, clustering) == pytest.approx(expected, rel=1e-12)

    def test_b_is_the_mean_distance_not_the_nearest(self):
        """One component at distance 1 from its medoid and 3 and 9 from the others: b = 6, so s = 5/6 (not 2/3)."""
        space = np.array([[0.0, 0.0], [1.0, 0.0], [-2.0, 0.0], [10.0, 0.0]])
        clustering = {'labels': np.array([0, 0, 1, 2]), 'medoids_loc': space[[0, 2, 3]]}
        assert calc_mss_value(space, clustering) == pytest.approx((6 - 1) / 6)

    def test_components_on_a_medoid_are_left_out(self):
        """A medoid (a = 0) would score 1; it is not averaged in."""
        space = np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0], [4.0, 0.0]])
        clustering = {'labels': np.array([0, 0, 1, 1]), 'medoids_loc': space[[0, 3]]}
        # Non-medoids: [1, 0] (a = 1, b = 3) and [3, 0] (a = 1, b = 3): both score 2/3; with medoids it would be 5/6.
        assert calc_mss_value(space, clustering) == pytest.approx(2 / 3)
