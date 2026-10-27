"""Mean Simplified Silhouette (MSS) calculation."""
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


def calc_mss_value(space: np.ndarray, clustering: dict[str, np.ndarray]) -> float:
    """Mean Simplified Silhouette (MSS) of a grouping (paper, Eqs. 9-12, with two choices made explicit).

    For every component, a is its distance to the medoid of its own group (Eq. 9) and b is its MEAN distance to the
    medoids of all other groups. The standard simplified silhouette takes the nearest other medoid instead (Eq. 10 as
    printed); with that choice b shrinks as groups are added and the MSS curve has no knee to find, so the mean is
    used. The silhouette of a component is (b - a) / max(a, b) (Eq. 11). The MSS is its average over the components
    that are not on a medoid (a > 0): a medoid would score 1 whatever the grouping, which only adds k/N to the
    average (Eq. 12 averages over all N components).

    Args:
        space: Positions of the components, one row per component (the 2-D t-SNE map of their profiles).
        clustering: The grouping, as returned by kmedoids_fasterpam: 'labels' (group of every component) and
            'medoids_loc' (position of every medoid).

    Returns:
        The MSS, in [-1, 1]; 1.0 if every component is on a medoid.
    """
    cluster_labels = clustering['labels']
    cluster_centers = clustering['medoids_loc']

    # Distance of every component to its own medoid (a) and mean distance to the other medoids (b)
    intra_cluster_distances = np.linalg.norm(
        space - cluster_centers[cluster_labels], axis=1
    )

    n_points = space.shape[0]
    n_clusters = cluster_centers.shape[0]

    # Compute all point-to-center distances at once
    all_distances = euclidean_distances(space, cluster_centers)

    # For each point, the mean distance to the medoids of the other groups
    other_cluster_distances = np.zeros(n_points)

    for cluster_idx in range(n_clusters):
        mask = cluster_labels == cluster_idx
        if not np.any(mask):
            continue

        # Get distances to all other centers
        other_centers_mask = np.ones(n_clusters, dtype=bool)
        other_centers_mask[cluster_idx] = False

        distances_to_others = all_distances[mask][:, other_centers_mask]
        other_cluster_distances[mask] = distances_to_others.mean(axis=1)

    # Components on a medoid (a = 0) are left out
    valid_mask = intra_cluster_distances != 0
    if not np.any(valid_mask):
        return 1.0

    a = intra_cluster_distances[valid_mask]
    b = other_cluster_distances[valid_mask]
    silhouette_scores = (b - a) / np.maximum(a, b)

    return float(np.mean(silhouette_scores))
