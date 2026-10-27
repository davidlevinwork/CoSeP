"""k-medoids grouping, the MSS sweep, the knee, and the representatives of the groups (paper, Sections 3.3-3.5)."""
from collections.abc import Sequence
from typing import Optional, Union

import numpy as np
from kmedoids import fasterpam
from kneed import KneeLocator
from loguru import logger
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.utils import check_random_state

from cosep.graph.mss import calc_mss_value

# k* is chosen among the candidates k = 2, ..., N-1, and a knee needs at least three of them: a layer needs at least
# five components. The pruner keeps smaller layers whole.
MIN_COMPONENTS = 5


def _check_finite(values: np.ndarray, what: str) -> None:
    """Raise if an array contains NaN or Inf values (they would otherwise read as 'identical' or 'no separability')."""
    finite = np.isfinite(values)
    if not finite.all():
        raise ValueError(f"{what} contains {int((~finite).sum())} NaN or Inf values.")


def kmedoids_fasterpam(data: np.ndarray, k: int, precomputed_distances: Optional[np.ndarray] = None,
                       random_state: Optional[Union[int, np.random.RandomState]] = None) -> dict[str, np.ndarray]:
    """Group points into k groups with k-medoids (the FasterPAM algorithm).

    FasterPAM starts from k medoids drawn at random from `random_state` and runs on one thread, in its fixed
    processing order, so the same generator state always gives the same result. (The kmedoids package's parallel
    version, which it uses by default for 1,000 points or more, can return different medoids for the same seed.)

    Args:
        data: Points to group, one row per point.
        k: Number of groups.
        precomputed_distances: Pairwise distances between the points; computed from `data` if None.
        random_state: Generator or seed for the starting medoids. None: NumPy's global generator.

    Returns:
        A dict with 'labels' (group of every point), 'medoids' (row index of every medoid) and 'medoids_loc' (the
        medoids' rows of `data`).

    Raises:
        ValueError: If the data or the distances contain NaN or Inf values.
    """
    _check_finite(data, "The data to cluster")
    if precomputed_distances is not None:
        distances = precomputed_distances
        _check_finite(distances, "The distance matrix")
    else:
        distances = pairwise_distances(data)

    start = check_random_state(random_state).choice(distances.shape[0], k, replace=False).astype(np.uintp)
    k_medoids = fasterpam(distances, start, n_cpu=1)
    medoids_loc = data[k_medoids.medoids]

    return {
        'labels': k_medoids.labels,
        'medoids': k_medoids.medoids,
        'medoids_loc': medoids_loc,
    }


def get_knee(x: Sequence[int], y: Sequence[float], poly_deg: int = 2) -> int:
    """Number of groups k* from the MSS curve: the Kneedle knee of a degree-p polynomial fit (paper, Eq. 13).

    Kneedle rescales the fitted curve to the unit square and takes the point of its concave bend that lies farthest
    above the straight line joining its ends. The degree p sets the overall pruning level: with p = 2 the fit is a
    parabola, whose knee lies near the middle of the candidates whenever it rises across the whole range, so about
    half of each layer is kept; higher degrees put the knee earlier and keep fewer components.

    When the fit has no concave bend, the middle candidate is used instead and a warning is logged. This covers a
    curve without a knee (for example a straight line) and a fit that bends upward: Kneedle then returns the first
    candidate, which would keep only two components.

    Args:
        x: Candidate numbers of groups, in increasing order.
        y: MSS value of each candidate.
        poly_deg: Degree p of the polynomial fit.

    Returns:
        k*, one of the candidates.

    Raises:
        ValueError: If there are no candidates.
    """
    x = [int(v) for v in x]
    if not x:
        raise ValueError('The MSS curve has no candidates.')
    middle = x[len(x) // 2]
    if len(x) < 3:
        logger.warning(f"Only {len(x)} candidate(s) for the number of groups; using k = {middle}.")
        return middle

    kn = KneeLocator(
        x, list(y),
        curve='concave',
        direction='increasing',
        interp_method='polynomial',
        polynomial_degree=poly_deg,
    )
    if kn.knee is None:
        reason = 'Kneedle found no knee in the fitted MSS curve'
    else:
        index = x.index(int(kn.knee))
        if index > 0 and kn.y_difference[index] > 0:
            return int(kn.knee)
        reason = 'The fitted MSS curve has no concave knee (it bends upward)'
    logger.warning(f"{reason}; using the middle of the {len(x)} candidates, k = {middle}.")
    return middle


def select_optimal_components(
    graph_space: dict[str, np.ndarray],
    weights: np.ndarray,
    num_components: int,
    poly_degree: int = 2,
    random_state: Optional[Union[int, np.random.RandomState]] = None
) -> list[int]:
    """Select the components to keep (paper, Sections 3.3-3.5).

    For every candidate number of groups k = 2, ..., N-1, k-medoids groups the components on the 2-D t-SNE map of
    their separability profiles, and the grouping is scored with the MSS (see calc_mss_value). If the MSS reaches 1,
    every component sits on its group's medoid; the sweep stops there and the remaining candidates score 1. The knee
    of the MSS curve gives k* (Eq. 13, see get_knee). k-medoids is then run with k*, and each group keeps its
    component with the largest weight norm (Eqs. 14-15).

    Args:
        graph_space: The space the components are grouped in (see LayerPruner.create_layer_space); must contain
            'reduced_matrix', the 2-D t-SNE map of the profiles (one row per component).
        weights: Weight norm of every component (Eq. 14).
        num_components: Number of components N of the layer.
        poly_degree: Degree p of the polynomial fit of the MSS curve.
        random_state: Generator or seed for the starting medoids of every k-medoids run. None: NumPy's global
            generator.

    Returns:
        Indices of the kept components, one per group.

    Raises:
        KeyError: If graph_space has no 'reduced_matrix'.
        ValueError: If the embedding contains NaN or Inf values.
    """
    embedding = graph_space.get('reduced_matrix')
    if embedding is None:
        raise KeyError("graph_space must contain 'reduced_matrix'")
    _check_finite(embedding, "The 2-D embedding of the separability profiles")
    random_state = check_random_state(random_state)  # one generator for the whole sweep, even from an int seed

    # Precompute pairwise distances once for reuse
    precomputed = pairwise_distances(embedding)

    mss_values = []
    k_values = range(2, num_components)

    for i, k in enumerate(k_values):
        k_medoids = kmedoids_fasterpam(embedding, k, precomputed, random_state=random_state)
        mss_value = calc_mss_value(clustering=k_medoids, space=embedding)
        mss_values.append(mss_value)

        if mss_value >= 1.0:
            mss_values.extend([1.0] * (len(k_values) - i - 1))
            break

    knee = get_knee(list(k_values), mss_values, poly_degree)
    optimal_kmedoids = kmedoids_fasterpam(embedding, int(knee), precomputed, random_state=random_state)
    return find_optimal_weighted_medoids(weights, optimal_kmedoids['labels'], optimal_kmedoids['medoids'])


def find_optimal_weighted_medoids(
    weights: np.ndarray, labels: np.ndarray, medoids: np.ndarray
) -> list[int]:
    """The representative of every group: its component with the largest weight norm (paper, Eq. 15).

    Args:
        weights: Weight norm of every component (Eq. 14).
        labels: Group of every component.
        medoids: Index of each group's medoid; it identifies the group.

    Returns:
        Index of the representative of each group, in the order of `medoids`.
    """
    highest_weight_indices = np.empty_like(medoids)

    for i, medoid in enumerate(medoids):
        cluster_indices = np.where(labels == labels[medoid])[0]
        cluster_weights = weights[cluster_indices]
        max_weight_index = cluster_indices[np.argmax(cluster_weights)]
        highest_weight_indices[i] = max_weight_index

    return highest_weight_indices.tolist()
