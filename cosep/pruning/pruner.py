"""The Pruner: CoSeP applied to a whole network, layer by layer (paper, Algorithm 1)."""
import copy
import csv
import json
import numbers
import os
import re
from typing import Any, Optional, Union

import torch
import torch.nn as nn
from loguru import logger

from cosep import config
from cosep.data_types.data_entity import DataEntity
from cosep.data_types.metrics import Metrics
from cosep.data_types.model_entity import ModelEntity
from cosep.graph.clustering import MIN_COMPONENTS
from cosep.pruning.pruners import Conv1dLayerPruner, Conv2dLayerPruner, LinearLayerPruner
from cosep.pruning.pruners.layer_pruner import LayerPruner
from cosep.pruning.structure import LayerRule, ModelStructure, check_plan, count_macs, num_components
from cosep.utils.model_utils import (
    evaluate_model,
    execution_order,
    get_model_layers,
    save_model,
    save_pruned_model,
    train_model,
)
from cosep.utils.plot_utils import plot_weight_distribution
from cosep.utils.random_utils import augmentation_generator, check_seed

PLAN_FORMAT = 'cosep-pruning-plan/1'

SUMMARY_COLUMNS = [
    'layer', 'type', 'status',
    'out_before', 'out_after', 'out_kept_pct',
    'in_before', 'in_after',
    'params_before', 'params_after', 'params_kept_pct',
    'macs_before', 'macs_after', 'macs_kept_pct',
]


def _count_params(module: nn.Module) -> int:
    """Count all parameters of a module, including its sub-modules."""
    return sum(p.numel() for p in module.parameters())


def _in_out(layer: nn.Module) -> tuple[int, int]:
    """Input and output width of a convolution, linear or attention layer."""
    if isinstance(layer, nn.Linear):
        return layer.in_features, layer.out_features
    if isinstance(layer, nn.MultiheadAttention):
        return layer.embed_dim, layer.embed_dim
    if isinstance(layer, nn.modules.conv._ConvNd):
        return layer.in_channels, layer.out_channels
    raise TypeError(f'{type(layer).__name__} layers are not reported in the summary.')


def _pct(after: int, before: int) -> Union[float, str]:
    """Share kept, in percent (empty when there was nothing before)."""
    return round(100.0 * after / before, 2) if before else ''


def _device_of(model: nn.Module) -> torch.device:
    """Device of a model's parameters."""
    return next(model.parameters()).device


def _layer_pruner_class(layer: nn.Module) -> type[LayerPruner]:
    """The layer pruner for a kind of layer.

    Args:
        layer: Layer to prune.

    Returns:
        The LayerPruner subclass for this kind of layer.

    Raises:
        TypeError: If the layer is not a Conv1d, Conv2d or Linear layer.
    """
    if isinstance(layer, nn.Conv1d):
        return Conv1dLayerPruner
    if isinstance(layer, nn.Conv2d):
        return Conv2dLayerPruner
    if isinstance(layer, nn.Linear):
        return LinearLayerPruner
    raise TypeError(f'No pruner for {type(layer).__name__} layers.')


def _status_text(entry: dict[str, Any]) -> str:
    """Status of a plan entry, as written to the summary CSV."""
    if entry['status'] == 'pruned with':
        return f"pruned with {entry['pruned_with']}"
    if entry['status'] == 'not pruned':
        return f"not pruned: {entry['reason']}"
    return entry['status']


@torch.no_grad()
def _output_shape(model: nn.Module, example_inputs: torch.Tensor) -> tuple[int, ...]:
    """Shape of the model's output for the example inputs (computed in evaluation mode)."""
    was_training = model.training
    model.eval()
    try:
        return tuple(model(example_inputs.to(_device_of(model))).shape)
    finally:
        model.train(was_training)


class Pruner:
    """Prune a network with CoSeP, layer by layer (paper, Algorithm 1).

    For each prunable layer, in the order the forward pass runs them: select the components to keep, remove all
    other components from the network, then briefly fine-tune. Removal is physical, so fine-tuning cannot bring
    removed components back.

    Layers whose output channels are shared with other layers (for example, the layers that write into a residual
    sum) are kept whole. A depthwise convolution is pruned together with the layer that feeds it. The first and the
    last layer that the forward pass runs are not pruned, nor are layers with fewer than five components.

    Attributes:
        model: The model entity; prune() prunes its model in place.
        layers: The model's convolution, linear and attention layers, in the order of get_model_layers.
        plan: After prune(), one entry per layer, in the order the pruner visited them (see pruning_plan.json).
    """

    def __init__(
        self,
        model: ModelEntity,
        train_data: DataEntity,
        test_data: DataEntity,
        retrain_epochs: int = 2,
        ignore_layers: Optional[list[int]] = None,
        example_inputs: Optional[torch.Tensor] = None,
        poly_degree: int = 2,
        max_classes: Optional[int] = 100,
        learning_rate: float = 0.001,
        momentum: float = 0.9,
        weight_decay: float = 5e-4,
        nesterov: bool = False,
        seed: int = 0,
        augment: bool = False
    ) -> None:
        """Initialize the Pruner.

        Args:
            model: Model entity to prune.
            train_data: Training data entity: the calibration data for the class statistics, and the data for the
                fine-tuning after each layer (its loader should shuffle).
            test_data: Test data entity, used to report accuracy.
            retrain_epochs: Fine-tuning epochs after each pruned layer.
            ignore_layers: Layer indices to skip (numbered from 0, in the order of get_model_layers).
            example_inputs: One input batch, used to trace the model and count MACs.
                Defaults to the first sample of the training data.
            poly_degree: Degree p of the polynomial fitted to the MSS curve (paper, Eq. 13). It sets the overall
                pruning level: p = 2 keeps about half of each layer, higher degrees keep fewer components.
            max_classes: M: with more classes than this, each layer's profiles use only the M classes that it
                separates best (paper, Appendix A). None or 0: always use every class.
            learning_rate: Fine-tuning SGD learning rate. Low on purpose: the network is already trained (see
                train_model).
            momentum: Fine-tuning SGD momentum.
            weight_decay: Fine-tuning SGD weight decay.
            nesterov: Use Nesterov momentum when fine-tuning.
            seed: Seed of the selection's random steps: each layer's t-SNE and k-medoids draw from a generator made
                from this seed and the layer index, never from NumPy's global generator. The augmentation's shifts
                and flips draw from a generator of their own, also made from this seed. The training loader's
                sample and batch order have their own seed (see load_dataset); dropout during fine-tuning follows
                PyTorch's global generator.
            augment: Augment the fine-tuning batches: shift each image by up to 4 pixels and flip half of them
                horizontally (see augment_batch). For natural images; off by default. The calibration data are
                never augmented.

        Raises:
            ValueError: If an index in ignore_layers is not the index of a layer, if poly_degree is below 2, if
                max_classes is 1 or negative, if seed is not a whole number of 0 or more, or if augment is not True
                or False. prune() raises ValueError if augment is on and the inputs are not images (N, C, H, W).
        """
        self.model = model
        self.train_data = train_data
        self.test_data = test_data
        self.retrain_epochs = retrain_epochs
        self.example_inputs = example_inputs
        if not (isinstance(poly_degree, numbers.Integral) and poly_degree >= 2):
            raise ValueError(f"poly_degree must be a whole number of at least 2; got {poly_degree!r}.")
        if max_classes is not None and not (isinstance(max_classes, numbers.Integral)
                                            and (max_classes == 0 or max_classes >= 2)):
            raise ValueError(f"max_classes must be at least 2, or 0 for no screening; got {max_classes!r}.")
        if not isinstance(augment, bool):
            raise ValueError(f"augment must be True or False; got {augment!r}.")
        self.poly_degree: int = int(poly_degree)
        self.max_classes: Optional[int] = max_classes
        self.seed: int = check_seed(seed)
        self.fine_tuning: dict[str, Any] = {'learning_rate': learning_rate, 'momentum': momentum,
                                            'weight_decay': weight_decay, 'nesterov': nesterov, 'augment': augment}

        self.layers: list[nn.Module] = list(get_model_layers(self.model.model))
        extra = list(ignore_layers or [])
        invalid = [i for i in extra if not (isinstance(i, numbers.Integral) and 0 <= i < len(self.layers))]
        if invalid:
            raise ValueError(f"ignore_layers must be layer indices from 0 to {len(self.layers) - 1}; got {invalid}.")
        self.ignore_layers: list[int] = [int(i) for i in extra]

        # One entry per layer, in the order the pruner visits them (written to pruning_plan.json)
        self.plan: list[dict[str, Any]] = []
        # Indices of the layers the forward pass runs, in that order (set by prune())
        self.run_order: list[int] = []

        logger.debug(
            f"Pruner initialized with {len(self.layers)} layers, "
            f"retrain_epochs={self.retrain_epochs}, poly_degree={self.poly_degree}, "
            f"max_classes={self.max_classes}, seed={self.seed}, augment={augment}, "
            f"{len(self.ignore_layers)} ignored layers."
        )

    def prune(self) -> None:
        """Prune the model, then save it with its pruning plan and a summary.

        Writes to the run's output folder (cosep.config.get_output_dir()): original_model.pt, pruned_model.pt,
        pruning_plan.json and pruning_summary.csv, and, in verbose mode, plots/.

        Raises:
            ValueError: If augmentation is on and the inputs are not images (N, C, H, W).
            RuntimeError: If a layer does not have its planned size after pruning, or the pruned model's output shape
                differs from the original's.
        """
        output_dir = config.get_output_dir()
        model_name = type(self.model.model).__name__

        logger.info(f"Starting pruning for: {model_name}")

        # Save original model before pruning
        original_model_path = os.path.join(output_dir, 'original_model.pt')
        save_model(self.model.model, original_model_path, is_original=True)
        logger.info(f"Original model saved to: {original_model_path}")
        original_model = copy.deepcopy(self.model.model)

        initial_metrics = evaluate_model(self.model.model, self.test_data.data_loader)
        logger.debug(f"Initial metrics: {initial_metrics}")

        example = self._get_example_inputs()
        if self.fine_tuning['augment'] and example.dim() != 4:
            raise ValueError(f"augment=True needs image inputs (N, C, H, W); the model's inputs have shape "
                             f"{tuple(example.shape)}. Turn augmentation off for other kinds of input.")
        structure = ModelStructure(self.model.model, example.to(_device_of(self.model.model)))
        self.run_order = execution_order(self.model.model, self.layers, example.to(_device_of(self.model.model)))
        self._apply_pruning(structure)
        self._check_pruned_model(original_model, example)

        pruned_model_path = os.path.join(output_dir, 'pruned_model.pt')
        save_pruned_model(self.model.model, pruned_model_path)
        logger.info(f"Pruned model saved to: {pruned_model_path}")

        plan_path = os.path.join(output_dir, 'pruning_plan.json')
        self._save_pruning_plan(plan_path, model_name, example)
        logger.info(f"Pruning plan saved to: {plan_path}")

        final_metrics = evaluate_model(self.model.model, self.test_data.data_loader)

        logger.info(f"Initial metrics: {initial_metrics}")
        logger.info(f"Final metrics: {final_metrics}")

        # Generate CSV summary
        totals = self._save_pruning_summary(output_dir, original_model, example)

        # Log summary
        self._log_output_summary(output_dir, initial_metrics, final_metrics, totals)
        logger.success("Pruning complete and model saved.")

    def _get_example_inputs(self) -> torch.Tensor:
        """One input sample, used to trace the model and count MACs."""
        if self.example_inputs is not None:
            return self.example_inputs
        inputs, _ = next(iter(self.train_data.data_loader))
        return inputs[:1].clone()

    def _layer_rules(self, structure: ModelStructure, run_order: list[int]) -> dict[int, LayerRule]:
        """Decide, for every layer, whether CoSeP selects and removes its components.

        A layer that the structure allows to prune is still kept whole when it has fewer than MIN_COMPONENTS (5)
        components: k* is chosen among k = 2, ..., N-1, and the knee needs at least three candidates.

        Args:
            structure: Dependency graph of the model.
            run_order: Indices of the layers that the forward pass runs, in that order (see execution_order).
        """
        rules = {}
        first, last = (run_order[0], run_order[-1]) if run_order else (None, None)
        for idx, layer in enumerate(self.layers):
            if idx == first:
                rules[idx] = LayerRule(False, 'first layer')
            elif idx == last:
                rules[idx] = LayerRule(False, 'last layer')
            elif idx in self.ignore_layers:
                rules[idx] = LayerRule(False, 'ignored (--ignore_layers)')
            else:
                rule = structure.rule(layer)
                if rule.prunable and num_components(layer) < MIN_COMPONENTS:
                    rule = LayerRule(False, f'fewer than {MIN_COMPONENTS} components, too few to choose the number '
                                            'of groups')
                rules[idx] = rule
        return rules

    def _apply_pruning(self, structure: ModelStructure) -> None:
        """Select, remove and fine-tune, layer by layer in forward order, and record the pruning plan."""
        model = self.model.model
        names = {m: n for n, m in model.named_modules()}
        modules = dict(model.named_modules())
        index_of = {names[layer]: idx for idx, layer in enumerate(self.layers)}
        sizes_before = [num_components(layer) for layer in self.layers]
        rules = self._layer_rules(structure, self.run_order)
        # Layers the forward pass does not run come last; they are not pruned
        order = self.run_order + [idx for idx in range(len(self.layers)) if idx not in self.run_order]

        # Depthwise convolutions pruned together with the layer that feeds them: name -> index of that layer
        followers = {name: idx for idx, rule in rules.items() if rule.prunable for name in rule.pruned_with}
        entries: dict[int, dict] = {}
        # One generator for the whole run's augmentation; nothing else draws from it
        generator = augmentation_generator(self.seed) if self.fine_tuning['augment'] else None

        for idx in order:
            layer = self.layers[idx]
            name = names[layer]
            if name in followers:
                continue  # recorded when the layer that feeds it is pruned
            rule = rules[idx]
            entry = {'layer': name, 'type': type(layer).__name__, 'components_before': sizes_before[idx]}

            if not rule.prunable:
                logger.debug(f"Skipping layer {idx} ({name}): {rule.reason}.")
                entries[idx] = {**entry, 'status': 'not pruned', 'reason': rule.reason,
                                'components_after': sizes_before[idx]}
                continue

            plot_weight_distribution(layer, idx, before_prune=True)

            logger.info(f"Pruning layer {idx} ({name}).")
            pruner = _layer_pruner_class(layer)(self.model, self.train_data, idx, structure,
                                                poly_degree=self.poly_degree, max_classes=self.max_classes,
                                                seed=self.seed)
            self.model = pruner.prune()
            plot_weight_distribution(layer, idx, before_prune=False)  # the kept weights, before fine-tuning
            kept = pruner.kept_components
            entries[idx] = {**entry, 'status': 'pruned', 'components_after': len(kept), 'kept': kept}
            if rule.pruned_with:
                entries[idx]['pruned_together_with'] = rule.pruned_with
            for follower in rule.pruned_with:
                if follower not in index_of:
                    continue  # not one of the layers the pruner lists (e.g. Conv3d); resized all the same
                f_idx = index_of[follower]
                entries[f_idx] = {
                    'layer': follower, 'type': type(modules[follower]).__name__,
                    'components_before': sizes_before[f_idx], 'status': 'pruned with', 'pruned_with': name,
                    'components_after': len(kept), 'kept': kept,
                }

            logger.debug(f"Retraining for {self.retrain_epochs} epochs after layer {idx}.")
            train_model(
                self.model.model,
                self.train_data.data_loader,
                self.test_data.data_loader,
                num_epochs=self.retrain_epochs,
                verbose=config.verbose,
                generator=generator,
                **self.fine_tuning
            )

            # Removed components no longer exist, so fine-tuning cannot bring them back; check it anyway.
            check_plan(self.model.model, {'layers': list(entries.values())})

        self.plan = [entries[idx] for idx in order if idx in entries]

    def _check_pruned_model(self, original_model: nn.Module, example: torch.Tensor) -> None:
        """Check that every layer has its planned size and that the pruned model still computes its output."""
        check_plan(self.model.model, {'layers': self.plan})
        before = _output_shape(original_model, example)
        after = _output_shape(self.model.model, example)
        if before != after:
            raise RuntimeError(f"Pruning changed the model's output shape from {before} to {after}.")

    def _save_pruning_plan(self, plan_path: str, model_name: str, example: torch.Tensor) -> None:
        """Save which components of every layer were kept (pruning_plan.json)."""
        plan = {
            'format': PLAN_FORMAT,
            'model': model_name,
            'input_shape': list(example.shape),
            'settings': {'seed': self.seed, 'poly_degree': self.poly_degree, 'max_classes': self.max_classes,
                         'retrain_epochs': self.retrain_epochs, **self.fine_tuning},
            'pruned_model': 'pruned_model.pt',
            'description': (
                "One entry per convolution, linear or attention layer, in the order the pruner visits them (the "
                "order of the forward pass). "
                "For a pruned layer, 'kept' lists the output channels (or neurons) that were kept, numbered as in "
                "the original model; all others were removed. To rebuild the "
                "pruned architecture from the original model, call cosep.pruning.structure.apply_pruning_plan"
                "(model, plan, example_input), then load the weights of pruned_model.pt."
            ),
            'layers': self.plan,
        }
        text = json.dumps(plan, indent=2)
        # Keep each list of indices on one line
        text = re.sub(r'\[\s+([-\d,\s]+?)\s+\]',
                      lambda m: '[' + ', '.join(v.strip() for v in m.group(1).split(',')) + ']', text)
        with open(plan_path, 'w') as f:
            f.write(text + '\n')

    def _save_pruning_summary(self, output_dir: str, original_model: nn.Module,
                              example: torch.Tensor) -> dict[str, tuple[int, int]]:
        """Save the per-layer CSV summary, comparing the original and the pruned model.

        Each row gives a layer's output and input width, parameters and MACs, before and after pruning. A layer that
        is not pruned can still shrink, because the layer before it lost channels. The last row, WHOLE MODEL, counts
        every parameter of the model and the MACs of every convolution and linear layer, for one input.

        Args:
            output_dir: Folder to write pruning_summary.csv to.
            original_model: The model before pruning.
            example: One input batch, for counting MACs.

        Returns:
            The whole model's totals: 'params' and 'macs', each as (before, after).
        """
        model = self.model.model
        macs_before = count_macs(original_model, example.to(_device_of(original_model)))
        macs_after = count_macs(model, example.to(_device_of(model)))
        before_modules = dict(original_model.named_modules())
        after_modules = dict(model.named_modules())

        rows = []
        for entry in self.plan:
            before, after = before_modules[entry['layer']], after_modules[entry['layer']]
            (in_before, out_before), (in_after, out_after) = _in_out(before), _in_out(after)
            params_before, params_after = _count_params(before), _count_params(after)
            layer_macs_before, layer_macs_after = macs_before.get(before, 0), macs_after.get(after, 0)
            rows.append({
                'layer': entry['layer'], 'type': entry['type'], 'status': _status_text(entry),
                'out_before': out_before, 'out_after': out_after, 'out_kept_pct': _pct(out_after, out_before),
                'in_before': in_before, 'in_after': in_after,
                'params_before': params_before, 'params_after': params_after,
                'params_kept_pct': _pct(params_after, params_before),
                'macs_before': layer_macs_before, 'macs_after': layer_macs_after,
                'macs_kept_pct': _pct(layer_macs_after, layer_macs_before),
            })

        params = (_count_params(original_model), _count_params(model))
        macs = (sum(macs_before.values()), sum(macs_after.values()))
        rows.append({
            'layer': 'WHOLE MODEL',
            'params_before': params[0], 'params_after': params[1], 'params_kept_pct': _pct(params[1], params[0]),
            'macs_before': macs[0], 'macs_after': macs[1], 'macs_kept_pct': _pct(macs[1], macs[0]),
        })

        csv_path = os.path.join(output_dir, 'pruning_summary.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS, restval='')
            writer.writeheader()
            writer.writerows(rows)

        logger.info(f"Pruning summary saved to: {csv_path}")
        return {'params': params, 'macs': macs}

    def _log_output_summary(self, output_dir: str, initial_metrics: Metrics, final_metrics: Metrics,
                            totals: dict[str, tuple[int, int]]) -> None:
        """Log what the run wrote, and the parameters, MACs and accuracy before and after pruning."""
        params, macs = totals['params'], totals['macs']
        logger.info("=" * 60)
        logger.info("OUTPUT FOLDER SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Location: {output_dir}")
        logger.info("")
        logger.info("Files:")
        logger.info("  - original_model.pt    : Model before pruning")
        logger.info("  - pruned_model.pt      : Pruned model (whole module)")
        logger.info("  - pruning_plan.json    : Kept components of every pruned layer")
        logger.info("  - pruning_summary.csv  : Per-layer sizes, parameters and MACs")
        # The log file exists only if init_logger() set it up (the command line does), and plots only in verbose mode
        if os.path.isfile(os.path.join(output_dir, 'log.log')):
            logger.info("  - log.log              : Run log")
        if os.path.isdir(os.path.join(output_dir, 'plots')):
            logger.info("  - plots/               : Weight distributions and t-SNE maps of the pruned layers")
        logger.info("")
        logger.info(f"Parameters: {params[0]:,} -> {params[1]:,} ({_pct(params[1], params[0])}%)")
        logger.info(f"MACs per input: {macs[0]:,} -> {macs[1]:,} ({_pct(macs[1], macs[0])}%)")
        logger.info(f"Accuracy: {initial_metrics.accuracy*100:.2f}% -> {final_metrics.accuracy*100:.2f}%")
        logger.info("=" * 60)
