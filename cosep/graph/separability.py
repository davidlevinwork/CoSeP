"""Class-separability profiles of a layer's components (paper, Eqs. 5-7), and class screening (Appendix A)."""
from itertools import combinations
from typing import Optional

import numpy as np

from cosep.data_types.class_statistics import ClassStatistics

MIN_VARIANCE = 1e-10


def jm_distance(mean_p: np.ndarray, var_p: np.ndarray, mean_q: np.ndarray, var_q: np.ndarray) -> np.ndarray:
    """Jeffries-Matusita distance between two univariate Gaussians, elementwise (paper, Eqs. 5-6).

    Standard deviations are floored at sqrt(1e-10), so that constant outputs do not divide by zero.

    Args:
        mean_p: Means of the first class.
        var_p: Variances of the first class.
        mean_q: Means of the second class.
        var_q: Variances of the second class.

    Returns:
        JM distances in [0, 2], with the shape of the inputs.
    """
    min_std = np.sqrt(MIN_VARIANCE)
    std_p = np.maximum(np.sqrt(var_p), min_std)
    std_q = np.maximum(np.sqrt(var_q), min_std)
    var_sum = std_p ** 2 + std_q ** 2

    term1 = 0.125 * ((mean_p - mean_q) ** 2) * (2.0 / var_sum)
    log_arg = np.maximum(var_sum / (2.0 * std_p * std_q), 1e-10)
    term2 = 0.5 * np.log(log_arg)

    bhattacharyya = term1 + term2
    jm = 2.0 * (1.0 - np.exp(-bhattacharyya))
    return np.clip(jm, 0.0, 2.0)


def _check_two_dimensional(statistics: ClassStatistics) -> None:
    """Raise if the statistics are not one value per class and component (shape (classes, components))."""
    if statistics.means.ndim != 2 or statistics.variances.shape != statistics.means.shape:
        raise ValueError('Class statistics must have shape (classes, components): one activation value per component '
                         f'and sample (paper, Eqs. 1-2); got means of shape {statistics.means.shape}.')


def separability_profiles(statistics: ClassStatistics) -> np.ndarray:
    """Separability profile of every component of a layer (paper, Eq. 7).

    The profile of a component is its JM distance for every pair of classes: one value per class pair, computed from
    the per-class mean and variance of the component's activation summary (Eqs. 1-6). The pairs are in increasing
    order of their classes.

    Args:
        statistics: Per-class statistics of the layer's components, of shape (classes, components).

    Returns:
        Array of shape (components, pairs); row i is the profile of component i.

    Raises:
        ValueError: If the data has fewer than two classes, or the statistics are not of shape (classes, components).
    """
    _check_two_dimensional(statistics)
    num_classes = len(statistics.classes)
    if num_classes < 2:
        raise ValueError('Separability profiles need data from at least two classes.')
    pairs = list(combinations(range(num_classes), 2))
    profiles = np.empty((statistics.num_components, len(pairs)))
    means, variances = statistics.means, statistics.variances
    for j, (p, q) in enumerate(pairs):
        profiles[:, j] = jm_distance(means[p], variances[p], means[q], variances[q])
    return profiles


def one_vs_rest_scores(statistics: ClassStatistics) -> np.ndarray:
    """Screening score of every class (paper, Appendix A, Eq. 17).

    For each class t, the samples of all other classes are pooled into one complement group. Each component gets the
    JM distance between class t and its complement (the same Gaussian/JM construction as Eqs. 4-6), and the class
    score is the mean of these distances over the layer's components. The complement's mean and variance follow
    exactly from the per-class counts, means and variances.

    Args:
        statistics: Per-class statistics of the layer's components, of shape (classes, components).

    Returns:
        One score per class, in the order of `statistics.classes`.

    Raises:
        ValueError: If the data has fewer than two classes, or the statistics are not of shape (classes, components).
    """
    _check_two_dimensional(statistics)
    if len(statistics.classes) < 2:
        raise ValueError('Screening classes needs data from at least two classes.')
    counts = statistics.counts.astype(np.float64)[:, None]
    means, variances = statistics.means, statistics.variances
    # Work relative to the overall mean of each component: JM depends on means only through their differences, and
    # centring keeps the pooled variances accurate.
    centred = means - (counts * means).sum(axis=0) / counts.sum()
    sum_means = (counts * centred).sum(axis=0)
    sum_squares = (counts * (variances + centred ** 2)).sum(axis=0)
    rest_counts = counts.sum() - counts
    rest_means = (sum_means - counts * centred) / rest_counts
    rest_squares = (sum_squares - counts * (variances + centred ** 2)) / rest_counts
    rest_variances = np.maximum(rest_squares - rest_means ** 2, 0.0)
    return jm_distance(centred, variances, rest_means, rest_variances).mean(axis=1)


def screen_classes(statistics: ClassStatistics, max_classes: Optional[int]) -> ClassStatistics:
    """Keep the classes that a layer separates best, when there are more than `max_classes` (paper, Appendix A).

    With C classes a profile has C(C-1)/2 values, which is impractical for ImageNet (499,500 pairs). When C exceeds
    `max_classes` (M), each class is scored by Eq. 17 and only the M classes with the largest scores are kept, so the
    profiles are built from their M(M-1)/2 pairs (Eq. 18). The selection is made separately for every layer. Ties are
    broken in favour of the class that comes first.

    Args:
        statistics: Per-class statistics of the layer's components, of shape (classes, components).
        max_classes: M, the largest number of classes whose pairs are used. None or 0: always use every class.

    Returns:
        The statistics of the kept classes, in their original order (the input itself if no class is dropped).

    Raises:
        ValueError: If `max_classes` is 1 or negative.
    """
    if not max_classes or len(statistics.classes) <= max_classes:
        return statistics
    if max_classes < 2:
        raise ValueError(f'max_classes must be at least 2 (or 0 for no screening); got {max_classes}.')
    scores = one_vs_rest_scores(statistics)
    keep = np.sort(np.argsort(-scores, kind='stable')[:max_classes])
    return ClassStatistics(classes=statistics.classes[keep], counts=statistics.counts[keep],
                           means=statistics.means[keep], variances=statistics.variances[keep])
