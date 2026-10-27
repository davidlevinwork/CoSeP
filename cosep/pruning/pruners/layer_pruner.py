"""Base class of the layer pruners: select the components of one layer to keep, and remove the others."""
from abc import ABC, abstractmethod
from typing import Optional, Union

import numpy as np
from loguru import logger
from sklearn.manifold import TSNE
from sklearn.utils import check_random_state

from cosep.data_types.class_statistics import ClassStatistics
from cosep.data_types.data_entity import DataEntity
from cosep.data_types.decorators import timing_decorator
from cosep.data_types.model_entity import ModelEntity
from cosep.graph.clustering import select_optimal_components
from cosep.graph.separability import screen_classes, separability_profiles
from cosep.pruning.structure import ModelStructure
from cosep.utils.layer_utils import collect_class_statistics, extract_weights
from cosep.utils.model_utils import get_model_layers
from cosep.utils.plot_utils import plot_feature_space
from cosep.utils.random_utils import check_seed, layer_random_state


class LayerPruner(ABC):
    """Prune one layer: select the components to keep (paper, Sections 3.1-3.5), then remove all the others.

    The selection is the same for every kind of layer; a subclass only names the kind (see layer_type).

    Attributes:
        kept_components: Indices of the components that prune() kept, in increasing order.
        statistics: Per-class statistics of the layer's components while prune() runs; None otherwise.
        random_state: Generator of the layer's t-SNE and k-medoids, made by prune() from the seed and the layer index.
    """

    def __init__(
        self,
        model: ModelEntity,
        train_data: DataEntity,
        layer_index: int,
        structure: ModelStructure,
        poly_degree: int = 2,
        max_classes: Optional[int] = 100,
        seed: int = 0
    ) -> None:
        """Initialize the pruner.

        Args:
            model: Model entity to prune.
            train_data: Calibration data: their per-class statistics give the separability profiles.
            layer_index: Index of the layer to prune (in the order of get_model_layers).
            structure: Dependency graph of the model, used to remove the components that are not kept.
            poly_degree: Degree p of the polynomial fitted to the MSS curve (paper, Eq. 13).
            max_classes: M: with more classes than this, the profiles use only the M classes that the layer separates
                best (paper, Appendix A). None or 0: always use every class.
            seed: Seed of the run. The layer's t-SNE and k-medoids draw from a generator made from this seed and the
                layer index (see cosep.utils.random_utils.layer_random_state).
        """
        self.layer_index = layer_index
        self.model: ModelEntity = model
        self.train_data: DataEntity = train_data
        self.structure = structure
        self.poly_degree = poly_degree
        self.max_classes = max_classes
        self.seed = check_seed(seed)
        self.random_state: Optional[np.random.RandomState] = None  # set by prune()

        self.statistics: Optional[ClassStatistics] = None
        self.kept_components: list[int] = []

    @property
    @abstractmethod
    def layer_type(self) -> str:
        """Name of the kind of layer: 'Conv1d', 'Conv2d' or 'Linear'."""

    @timing_decorator
    def prune(self) -> ModelEntity:
        """Select the layer's components to keep and remove all the others.

        Collects the per-class statistics of the layer's components (paper, Eqs. 1-4), keeps the classes the layer
        separates best when there are more than `max_classes` (Appendix A), computes the separability profiles
        (Eqs. 5-7), maps them to 2-D with t-SNE and selects the components to keep (Sections 3.3-3.5). All other
        components are then removed from the network (Eq. 16), together with their BatchNorm entries and the
        matching inputs of every layer that reads them. Because the removed weights no longer exist, the fine-tuning
        that follows cannot bring them back.

        Returns:
            The pruned model entity (the same object, changed in place). The kept indices are in `kept_components`.
        """
        self.random_state = layer_random_state(self.seed, self.layer_index)
        layer = list(get_model_layers(self.model.model))[self.layer_index]
        weights = extract_weights(self.model.model, self.layer_index)
        statistics = collect_class_statistics(self.model.model, self.train_data.data_loader, self.layer_index)
        num_classes = len(statistics.classes)
        statistics = screen_classes(statistics, self.max_classes)
        if len(statistics.classes) < num_classes:
            logger.info(f"Layer {self.layer_index}: profiles over the {len(statistics.classes)} of {num_classes} "
                        "classes that it separates best (one-vs-rest JM screening).")
        num_components = statistics.num_components
        self.statistics = statistics

        graph_space = self.create_layer_space()
        optimal_components = select_optimal_components(
            graph_space, weights, num_components, poly_degree=self.poly_degree, random_state=self.random_state
        )
        self.kept_components = sorted({int(i) for i in optimal_components})
        resized = self.structure.remove(layer, self.kept_components)
        self.statistics = None  # release the statistics of this layer

        logger.info(
            f"Finished pruning layer {self.layer_index}. "
            f"Keeping [{len(self.kept_components)}/{num_components}] "
            f"=> {(len(self.kept_components) / num_components) * 100:.2f}%. "
            f"Removed the other components; resized modules: {', '.join(resized) or 'none'}."
        )

        return self.model

    @timing_decorator
    def create_layer_space(self) -> dict[str, np.ndarray]:
        """Build the space the components are grouped in: their separability profiles, mapped to 2-D with t-SNE.

        Returns:
            A dict with 'profiles' (one row per component, one JM distance per class pair; paper, Eq. 7) and
            'reduced_matrix' (their 2-D t-SNE map, one row per component).

        Raises:
            RuntimeError: If the layer's class statistics have not been collected (prune() collects them).
        """
        if self.statistics is None:
            raise RuntimeError("The class statistics of the layer have not been collected; call prune().")
        logger.debug(f"Creating the separability space of {self.layer_type} layer {self.layer_index}")
        profiles = separability_profiles(self.statistics)
        reduced = self._reduce_dimension(profiles, self.random_state)
        plot_feature_space(reduced, self.layer_index, self.layer_type, profiles=profiles)
        return {'profiles': profiles, 'reduced_matrix': reduced}

    @staticmethod
    def _reduce_dimension(data: np.ndarray,
                          random_state: Optional[Union[int, np.random.RandomState]] = None) -> np.ndarray:
        """Map the separability profiles to 2-D with t-SNE; the components are grouped on this map.

        t-SNE keeps the components that separate the same class pairs close together, and the k-medoids groups, the
        MSS and the knee are all computed on its 2-D map. Perplexity min(10, max(2, N // 4)) for N components,
        1,000 iterations, PCA initialisation (random when the profiles have a single value, i.e. two classes).

        Args:
            data: Separability profiles, one row per component. Not modified.
            random_state: Generator or seed for t-SNE and for the tiny noise added to all-zero profiles. The pruner
                passes the layer's generator; None uses NumPy's global generator.

        Returns:
            2-D map of the components, one row per component.

        Raises:
            ValueError: If the data contains NaN or Inf values.
        """
        finite = np.isfinite(data)
        if not finite.all():
            raise ValueError(f"The separability profiles contain {int((~finite).sum())} NaN or Inf values.")
        rng: np.random.RandomState = check_random_state(random_state)

        # All-zero profiles (components that separate no class pair) can trouble t-SNE: add tiny noise to them
        zero_rows = np.linalg.norm(data, axis=1) == 0
        if zero_rows.any():
            data = data.copy()
            data[zero_rows] += rng.normal(0, 1e-10, data[zero_rows].shape)

        # Perplexity must stay below the number of components
        n_samples = data.shape[0]
        perplexity = min(10, max(2, n_samples // 4))
        init = 'pca' if data.shape[1] >= 2 else 'random'  # PCA needs at least two columns

        return TSNE(n_components=2, max_iter=1000, perplexity=perplexity, init=init,
                    random_state=rng).fit_transform(data)
