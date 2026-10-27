"""Model utilities: listing a model's layers, saving models, fine-tuning (optionally augmented) and evaluation."""
import copy
import io
import time
from collections.abc import Iterator, Sequence
from typing import Any, Callable, Optional, Union

import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader
from torchmetrics.functional.classification import f1_score, precision, recall
from tqdm import tqdm

from cosep.data_types.metrics import Metrics

AUGMENT_PADDING = 4  # largest shift of the fine-tuning augmentation, in pixels


def get_model_layers(model: nn.Module) -> Iterator[nn.Module]:
    """Get all prunable layers from model recursively, in the order the modules are defined.

    This order numbers the layers (the indices used by --ignore_layers and in the log). The pruner visits them in
    the order the forward pass runs them (see execution_order).

    Args:
        model: Neural network model.

    Yields:
        Linear, Conv1d and Conv2d layers. An nn.MultiheadAttention layer is yielded as one layer (its projections are
        not listed separately); the pruner keeps it whole.
    """
    layer_types = (
        nn.Linear,
        nn.Conv1d,
        nn.Conv2d,
    )
    # MultiheadAttention should be yielded as-is, not recursed into
    attention_types = (nn.MultiheadAttention,)

    for layer in model.children():
        if isinstance(layer, attention_types):
            # Yield attention layer directly (don't recurse into its children)
            yield layer
        elif isinstance(layer, layer_types):
            yield layer
        elif len(list(layer.children())) > 0:
            # Recurse into container modules
            yield from get_model_layers(layer)


@torch.no_grad()
def execution_order(model: nn.Module, layers: Sequence[nn.Module], example_inputs: torch.Tensor) -> list[int]:
    """Positions of `layers` in the order one forward pass first runs them.

    Args:
        model: The model the layers belong to.
        layers: Layers of the model (for example, those of get_model_layers).
        example_inputs: One input batch on the model's device.

    Returns:
        Indices into `layers` of the layers that the forward pass runs, in the order it first runs them. Layers that
        it does not run are left out.
    """
    order: list[int] = []

    def record(index: int) -> Callable[[nn.Module, tuple[Any, ...], Any], None]:
        def hook(module: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
            if index not in order:
                order.append(index)
        return hook

    hooks = [layer.register_forward_hook(record(i)) for i, layer in enumerate(layers)]
    was_training = model.training
    model.eval()
    try:
        model(example_inputs)
    finally:
        for hook in hooks:
            hook.remove()
        model.train(was_training)
    return order


def save_model(
    model: nn.Module,
    file_path: str,
    is_original: bool = False
) -> None:
    """Save a model's state_dict in a checkpoint dict, the format load_model() reads.

    Args:
        model: Model to save.
        file_path: Output file path (normally in the run's output directory).
        is_original: True if saving the original model (before pruning).
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_type': type(model).__name__,
        'is_pruned': not is_original,
    }
    torch.save(checkpoint, file_path)
    logger.debug(f"Model saved to: {file_path}")


def save_pruned_model(model: nn.Module, file_path: str) -> None:
    """Save the whole pruned model: its architecture and its weights.

    Pruning removes channels, so the pruned layers no longer match the original model class and a state_dict alone
    cannot be loaded into it. The whole module is saved instead, on the CPU. Load it with
    `torch.load(file_path, weights_only=False)` (the model's class must be importable), or with
    cosep.utils.data_utils.load_model().

    Args:
        model: Pruned model.
        file_path: Output file path.
    """
    torch.save(copy.deepcopy(model).cpu(), file_path)
    logger.debug(f"Pruned model saved to: {file_path}")


def augment_batch(
    images: torch.Tensor,
    generator: Optional[torch.Generator] = None,
    padding: int = AUGMENT_PADDING,
    fill: Union[float, Sequence[float]] = 0.0
) -> torch.Tensor:
    """Randomly shift and flip a batch of images (the usual CIFAR augmentation).

    Each image is padded by `padding` pixels on every side, cropped back to its size at a random position, and
    flipped horizontally with probability 1/2. Padded pixels take the value `fill`: by default 0, which is black for
    images in [0, 1] and the mean colour for images normalised by the dataset's mean and standard deviation. The
    crop positions (rows, then columns) and the flips are drawn in that order from `generator`, on the CPU.

    Args:
        images: A batch of images, (N, C, H, W), on any device.
        generator: CPU generator of the positions and flips; PyTorch's global generator if None.
        padding: Largest shift, in pixels (0 or more).
        fill: Value of the padded pixels: one number, or one per channel.

    Returns:
        The augmented batch: a new tensor with the shape, dtype and device of `images`.

    Raises:
        ValueError: If `images` is not a batch of images, if padding is negative, or if fill has neither one value
            nor one per channel.
    """
    if images.dim() != 4:
        raise ValueError(f"Augmentation needs a batch of images (N, C, H, W); got shape {tuple(images.shape)}.")
    if padding < 0:
        raise ValueError(f"padding must be 0 or more; got {padding}.")
    n, c, h, w = images.shape
    value = torch.as_tensor(fill, dtype=images.dtype, device=images.device).reshape(1, -1, 1, 1)
    if value.shape[1] not in (1, c):
        raise ValueError(f"fill must be one value or one per channel ({c}); got {value.shape[1]} values.")

    rows_start = torch.randint(0, 2 * padding + 1, (n,), generator=generator)
    cols_start = torch.randint(0, 2 * padding + 1, (n,), generator=generator)
    flip = torch.rand(n, generator=generator) < 0.5

    padded = value.expand(n, c, h + 2 * padding, w + 2 * padding).clone()
    padded[:, :, padding:padding + h, padding:padding + w] = images
    rows = rows_start[:, None] + torch.arange(h)  # (N, H): the padded rows each image keeps
    cols = cols_start[:, None] + torch.arange(w)  # (N, W)
    cols = torch.where(flip[:, None], cols.flip(1), cols)  # a flipped image reads its columns backwards
    rows, cols = rows.to(images.device), cols.to(images.device)
    batch = torch.arange(n, device=images.device)[:, None, None, None]
    channel = torch.arange(c, device=images.device)[None, :, None, None]
    return padded[batch, channel, rows[:, None, :, None], cols[:, None, None, :]]


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    num_epochs: int,
    learning_rate: float = 0.001,
    momentum: float = 0.9,
    weight_decay: float = 5e-4,
    nesterov: bool = False,
    verbose: bool = True,
    augment: bool = False,
    generator: Optional[torch.Generator] = None
) -> None:
    """Briefly fine-tune a model with SGD (paper, Section 3.6 and Appendix C).

    The defaults: learning rate 0.001, momentum 0.9, weight decay 5e-4; use the momentum, weight decay and Nesterov
    setting of the model's own training run. The learning rate is low on purpose: the network is already trained,
    and a learning rate meant for training from scratch pulls it away from its solution. Shuffling is up to the train
    loader. With augment, each training batch is randomly shifted by up to 4 pixels and flipped (augment_batch); the
    test batches are not, and the loader itself is left unchanged, so the calibration data are never augmented.

    Args:
        model: Model to train.
        train_loader: Training data loader.
        test_loader: Test data loader, used to report the accuracy before and after.
        num_epochs: Number of training epochs.
        learning_rate: SGD learning rate.
        momentum: SGD momentum.
        weight_decay: SGD weight decay.
        nesterov: Use Nesterov momentum.
        verbose: Show progress bars and detailed output.
        augment: Augment the training batches (for natural images; off by default).
        generator: CPU generator of the augmentation's shifts and flips; PyTorch's global generator if None.

    Raises:
        ValueError: If augment is on and the training inputs are not images (N, C, H, W).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    initial_metrics = evaluate_model(model, test_loader)
    logger.debug(f"Pre fine-tune accuracy: {initial_metrics.accuracy*100:.2f}%")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=nesterov
    )

    logger.info(
        f"Fine-tuning for {num_epochs} epochs "
        f"(lr={learning_rate}, momentum={momentum}, wd={weight_decay}, nesterov={nesterov}, augment={augment})"
    )

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Progress bar for training
        pbar = tqdm(
            train_loader,
            desc=f'Epoch {epoch}/{num_epochs}',
            leave=False,
            disable=not verbose
        )

        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            if augment:
                inputs = augment_batch(inputs, generator)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })

        # Epoch summary
        epoch_loss = running_loss / total
        train_acc = 100. * correct / total
        metrics = evaluate_model(model, test_loader)

        logger.debug(
            f"Epoch {epoch}/{num_epochs} - "
            f"Loss: {epoch_loss:.4f}, Train Acc: {train_acc:.2f}%, "
            f"Val Acc: {metrics.accuracy*100:.2f}%"
        )

    final_metrics = evaluate_model(model, test_loader)
    logger.info(
        f"Fine-tuning complete. "
        f"Accuracy: {initial_metrics.accuracy*100:.2f}% -> {final_metrics.accuracy*100:.2f}%"
    )


def evaluate_model(model: nn.Module, data_loader: DataLoader) -> Metrics:
    """Evaluate a classifier: accuracy, macro-averaged precision, recall and F1, per-class metrics and timing.

    The model is moved to the GPU when one is available, and left in evaluation mode.

    Args:
        model: Model to evaluate.
        data_loader: Yields (inputs, labels) batches.

    Returns:
        The metrics.
    """
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    all_labels: list[torch.Tensor] = []
    all_predictions: list[torch.Tensor] = []
    correct = 0
    total = 0
    num_outputs = 0

    start_time = time.perf_counter()

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predicted = outputs.argmax(dim=1)
            num_outputs = outputs.shape[1]

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.append(labels.cpu())
            all_predictions.append(predicted.cpu())

    end_time = time.perf_counter()

    accuracy = correct / total
    inference_time = (end_time - start_time) / len(data_loader)
    labels_all = torch.cat(all_labels)
    predictions_all = torch.cat(all_predictions)

    # Labels need not be 0..K-1 and the data may lack some classes: score every class the model can predict (or the
    # labels use), and report the classes present in the labels.
    num_classes = max(num_outputs, int(labels_all.max()) + 1)
    classes = torch.unique(labels_all)

    # Compute all metrics in single calls
    precision_all = precision(
        predictions_all, labels_all, average="none", num_classes=num_classes, task="multiclass"
    )
    recall_all = recall(
        predictions_all, labels_all, average="none", num_classes=num_classes, task="multiclass"
    )
    f1_all = f1_score(
        predictions_all, labels_all, average="none", num_classes=num_classes, task="multiclass"
    )

    # Macro averages over the classes present in the labels
    precision_macro = precision_all[classes].mean().item()
    recall_macro = recall_all[classes].mean().item()
    f1_macro = f1_all[classes].mean().item()

    per_class_metrics: dict[int, dict[str, float]] = {}
    for class_idx in classes.tolist():
        class_mask = labels_all == class_idx
        per_class_metrics[class_idx] = {
            "precision": precision_all[class_idx].item(),
            "recall": recall_all[class_idx].item(),
            "f1_score": f1_all[class_idx].item(),
            "accuracy": (predictions_all[class_mask] == class_idx).sum().item() / class_mask.sum().item()
        }

    model_size = get_model_size(model)

    return Metrics(
        accuracy=accuracy,
        precision=precision_macro,
        recall=recall_macro,
        f1_score=f1_macro,
        inference_time=inference_time,
        model_size=model_size,
        per_class_metrics=per_class_metrics,
    )


def get_model_size(model: nn.Module) -> float:
    """Size of a model's serialized state_dict, in megabytes (10^6 bytes)."""
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getbuffer().nbytes / 1e6
