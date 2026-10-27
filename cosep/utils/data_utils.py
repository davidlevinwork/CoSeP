"""Loading models and data: model files, datasets saved with torch.save, and ImageNet folders.

Model and dataset files are loaded with torch.load(weights_only=False), which unpickles them: load only files you
trust.
"""
import os
from collections import OrderedDict
from typing import IO, Callable, Optional, TypeVar, Union

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader, Dataset, Subset

from cosep.data_types.data_entity import DataEntity
from cosep.data_types.model_entity import ModelEntity
from cosep.models import VGG16, resnet56
from cosep.utils.random_utils import check_seed

ModelFactory = Callable[..., nn.Module]
FactoryT = TypeVar("FactoryT", bound=ModelFactory)

# Registry of known model architectures: name -> class or function that builds the model from `num_classes`
_MODEL_REGISTRY: dict[str, ModelFactory] = {}


def register_model(name: str) -> Callable[[FactoryT], FactoryT]:
    """Register a model architecture under a name, for load_model(model_arch=name).

    Use it as a decorator on a model class, or on a function that builds the model; either must accept the keyword
    argument `num_classes`.

    Args:
        name: Name to register the architecture under.

    Returns:
        A decorator that registers its argument and returns it unchanged.
    """
    def decorator(factory: FactoryT) -> FactoryT:
        _MODEL_REGISTRY[name] = factory
        return factory
    return decorator


def get_model_class(name: str) -> ModelFactory:
    """Get a registered model architecture by name.

    Args:
        name: Registered name.

    Returns:
        The model class or the function that builds the model.

    Raises:
        ValueError: If no architecture is registered under this name.
    """
    if name not in _MODEL_REGISTRY:
        available = list(_MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model: {name}. Available: {available}")
    return _MODEL_REGISTRY[name]


def get_available_models() -> list[str]:
    """Names of the registered model architectures."""
    return list(_MODEL_REGISTRY.keys())


def _register_builtin_models() -> None:
    """Register the example architectures: the paper's CIFAR-10 VGG-16 and ResNet-56."""
    for name, factory in (('vgg16', VGG16), ('VGG16', VGG16), ('resnet56', resnet56), ('ResNet56', resnet56)):
        register_model(name)(factory)


_register_builtin_models()


def load_model(
    model_source: Union[str, IO[bytes]],
    model_arch: Optional[str] = None,
    num_classes: int = 10
) -> ModelEntity:
    """Load a model from a file: a whole model saved with torch.save, or the state_dict of a registered architecture.

    A checkpoint dict with the state_dict under 'model_state_dict' (as save_model writes) is also accepted.

    Args:
        model_source: Path or file-like object.
        model_arch: Registered architecture to build when the file holds a state_dict (see get_available_models).
        num_classes: Number of output classes of that architecture (10 for CIFAR-10, 100 for CIFAR-100).

    Returns:
        ModelEntity with the loaded model.

    Raises:
        ValueError: If the file cannot be loaded, if it holds a state_dict and no `model_arch` is given, or if
            `model_arch` is not registered.
        RuntimeError: If the state_dict does not fit the architecture.
    """
    try:
        loaded = torch.load(model_source, weights_only=False)
    except Exception as error:
        raise ValueError(f"Failed to load model: {error}") from error
    if isinstance(model_source, str):
        logger.debug(f"Loaded model from {model_source}")
        model_path: Optional[str] = model_source
    else:
        logger.debug("Loaded model from file-like object")
        model_path = None

    # Check if loaded object is a state_dict or full model
    if isinstance(loaded, (dict, OrderedDict)) and not isinstance(loaded, nn.Module):
        # Check if it's a checkpoint dict (contains 'model_state_dict' key)
        if 'model_state_dict' in loaded:
            loaded = loaded['model_state_dict']  # Extract the actual state_dict

        # It's a state_dict, need to instantiate model first
        if model_arch is None:
            available = get_available_models()
            raise ValueError(
                f"Model file contains state_dict, not full model. "
                f"Please specify --model_arch. Available: {available}"
            )
        model_class = get_model_class(model_arch)
        model = model_class(num_classes=num_classes)
        model.load_state_dict(loaded)
        logger.debug(f"Instantiated {model_arch} with num_classes={num_classes} and loaded state_dict")
    else:
        # It's a full model
        model = loaded

    return ModelEntity(model=model, model_path=model_path)


def _stratified_sample_indices(labels: np.ndarray, fraction: float, seed: int = 0) -> np.ndarray:
    """Draw a sample of the data with the same class proportions (at least one sample per class).

    Args:
        labels: Label of every sample.
        fraction: Fraction of each class to draw, in (0, 1].
        seed: Seed of the sample's own generator; NumPy's global generator is neither read nor changed.

    Returns:
        Indices of the drawn samples, in increasing order.
    """
    rng = np.random.default_rng(check_seed(seed))
    selected: list[np.ndarray] = []

    for cls in np.unique(labels):
        cls_indices = np.where(labels == cls)[0]
        n_samples = max(1, int(len(cls_indices) * fraction))
        # Random sample without replacement
        selected.append(rng.choice(cls_indices, size=n_samples, replace=False))

    return np.sort(np.concatenate(selected))


def _shuffling_generator(seed: int) -> torch.Generator:
    """The generator of the training loader's shuffle, so the batch order depends on the seed only."""
    return torch.Generator().manual_seed(check_seed(seed))


def _labels_of(dataset: Dataset) -> np.ndarray:
    """Label of every sample of a dataset of (input, label) pairs, read in one pass over the data, in batches."""
    return torch.cat([torch.as_tensor(labels) for _, labels in DataLoader(dataset, batch_size=1024)]).numpy()


def load_dataset(
    train_data_source: Union[str, IO[bytes]],
    test_data_source: Union[str, IO[bytes]],
    batch_size: int = 128,
    train_fraction: float = 0.25,
    seed: int = 0
) -> tuple[DataEntity, DataEntity]:
    """Load a training and a test dataset saved with torch.save (for example TensorDatasets of normalised images).

    Each dataset must yield (input, label) pairs with integer class labels.

    Args:
        train_data_source: Path or file-like object of the training data.
        test_data_source: Path or file-like object of the test data.
        batch_size: Batch size of both loaders.
        train_fraction: Fraction of the training data to use, sampled per class (the paper uses 0.25). It is the
            calibration data and the fine-tuning data.
        seed: Seed of the training sample and of the training loader's shuffle.

    Returns:
        Tuple of (train_entity, test_entity). The training loader shuffles; the test loader does not.
    """
    train_dataset = torch.load(train_data_source, weights_only=False)
    test_dataset = torch.load(test_data_source, weights_only=False)

    # Labels of the whole training set, for the stratified sample
    all_train_labels = _labels_of(train_dataset)

    # Apply stratified sampling if fraction < 1
    if train_fraction < 1.0:
        selected_indices = _stratified_sample_indices(all_train_labels, train_fraction, seed)
        train_dataset = Subset(train_dataset, selected_indices.tolist())
        train_labels = all_train_labels[selected_indices]
        logger.info(
            f"Using {len(selected_indices)}/{len(all_train_labels)} "
            f"({train_fraction*100:.1f}%) training samples (stratified)"
        )
    else:
        train_labels = all_train_labels

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              generator=_shuffling_generator(seed))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    test_labels = torch.cat([labels for _, labels in test_loader]).numpy()

    train_entity = DataEntity(
        data_loader=train_loader,
        num_classes=len(np.unique(train_labels)),
        data_path=train_data_source if isinstance(train_data_source, str) else None,
        total_samples=len(train_dataset),
        labels=train_labels,
        unique_label_indices=np.unique(train_labels)
    )

    test_entity = DataEntity(
        data_loader=test_loader,
        num_classes=len(np.unique(test_labels)),
        data_path=test_data_source if isinstance(test_data_source, str) else None,
        total_samples=len(test_dataset),
        labels=test_labels,
        unique_label_indices=np.unique(test_labels)
    )

    return train_entity, test_entity


def load_imagenet(
    imagenet_path: str,
    batch_size: int = 128,
    train_fraction: float = 0.25,
    num_workers: int = 4,
    seed: int = 0
) -> tuple[DataEntity, DataEntity]:
    """Load ImageNet from its image folders, resized to 256 and centre-cropped to 224, with ImageNet normalisation.

    Expected folder structure::

        imagenet_path/
        ├── train/
        │   ├── class_name/
        │   │   └── *.JPEG
        │   └── ...
        └── val/  (or validation/)
            └── ...

    Args:
        imagenet_path: Path to the ImageNet root directory.
        batch_size: Batch size of both loaders (the paper uses 256 for ImageNet).
        train_fraction: Fraction of the training data to use, sampled per class (the paper uses 0.25). It is the
            calibration data and the fine-tuning data.
        num_workers: Number of data loading workers.
        seed: Seed of the training sample and of the training loader's shuffle.

    Returns:
        Tuple of (train_entity, test_entity). The training loader shuffles; the validation loader does not.

    Raises:
        ValueError: If the train/ folder, or both val/ and validation/, are missing.
    """
    import torchvision  # imported here: only ImageNet runs need it
    import torchvision.transforms as transforms

    train_dir = os.path.join(imagenet_path, 'train')
    val_dir = next((os.path.join(imagenet_path, name) for name in ('validation', 'val')
                    if os.path.isdir(os.path.join(imagenet_path, name))), None)

    if not os.path.exists(train_dir):
        raise ValueError(f"ImageNet train directory not found: {train_dir}")
    if val_dir is None:
        raise ValueError(
            f"ImageNet validation directory not found: expected {os.path.join(imagenet_path, 'val')} "
            f"or {os.path.join(imagenet_path, 'validation')}"
        )

    # ImageNet normalization
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    # The same deterministic transform for both sets: the class statistics need the images as they are
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = torchvision.datasets.ImageFolder(train_dir, transform=transform)
    val_dataset = torchvision.datasets.ImageFolder(val_dir, transform=transform)

    # Get all train labels
    all_train_labels = np.array([sample[1] for sample in train_dataset.samples])

    # Apply stratified sampling if fraction < 1
    if train_fraction < 1.0:
        selected_indices = _stratified_sample_indices(all_train_labels, train_fraction, seed)
        train_dataset = Subset(train_dataset, selected_indices.tolist())
        train_labels = all_train_labels[selected_indices]
        logger.info(
            f"Using {len(selected_indices)}/{len(all_train_labels)} "
            f"({train_fraction*100:.1f}%) ImageNet training samples (stratified)"
        )
    else:
        train_labels = all_train_labels

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, generator=_shuffling_generator(seed)
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    # Get val labels
    val_labels = np.array([sample[1] for sample in val_dataset.samples])

    train_entity = DataEntity(
        data_loader=train_loader,
        num_classes=len(np.unique(train_labels)),
        data_path=train_dir,
        total_samples=len(train_dataset),
        labels=train_labels,
        unique_label_indices=np.unique(train_labels)
    )

    test_entity = DataEntity(
        data_loader=val_loader,
        num_classes=len(np.unique(val_labels)),
        data_path=val_dir,
        total_samples=len(val_dataset),
        labels=val_labels,
        unique_label_indices=np.unique(val_labels)
    )

    logger.info(f"Loaded ImageNet: {len(train_dataset)} train, {len(val_dataset)} val samples")

    return train_entity, test_entity
