"""Tests for data loading."""
import os

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import RandomSampler, SequentialSampler, TensorDataset

from cosep.utils.data_utils import get_available_models, load_dataset, load_imagenet


def _image_folder(root, classes=('a', 'b'), per_class=2):
    for name in classes:
        os.makedirs(os.path.join(root, name))
        for i in range(per_class):
            Image.new('RGB', (8, 8), color=(60 * i, 0, 0)).save(os.path.join(root, name, f'{i}.png'))


@pytest.mark.parametrize('validation_folder', ['val', 'validation'])
def test_imagenet_validation_folder_may_be_val_or_validation(tmp_path, validation_folder):
    _image_folder(tmp_path / 'train')
    _image_folder(tmp_path / validation_folder)

    train, test = load_imagenet(str(tmp_path), num_workers=0, train_fraction=1.0)

    assert train.total_samples == 4 and test.total_samples == 4
    assert test.data_path == str(tmp_path / validation_folder)


def test_missing_imagenet_validation_folder_is_reported(tmp_path):
    _image_folder(tmp_path / 'train')
    with pytest.raises(ValueError, match='validation directory not found'):
        load_imagenet(str(tmp_path), num_workers=0)


def test_imagenet_defaults_shuffle_a_quarter_of_the_training_data(tmp_path):
    _image_folder(tmp_path / 'train', per_class=8)
    _image_folder(tmp_path / 'val')
    train, test = load_imagenet(str(tmp_path), num_workers=0)
    assert train.total_samples == 4  # 2 of 8 per class
    assert train.data_loader.batch_size == 128 and isinstance(train.data_loader.sampler, RandomSampler)
    assert isinstance(test.data_loader.sampler, SequentialSampler)


def test_dataset_defaults_are_the_papers(tmp_path):
    """25% of the training data, sampled per class; batch 128; the training data shuffled, the test data not."""
    labels = torch.arange(4).repeat_interleave(40)
    torch.save(TensorDataset(torch.randn(160, 3, 4, 4), labels), tmp_path / 'train.pt')
    torch.save(TensorDataset(torch.randn(20, 3, 4, 4), torch.arange(20) % 4), tmp_path / 'test.pt')

    train, test = load_dataset(str(tmp_path / 'train.pt'), str(tmp_path / 'test.pt'))

    assert train.total_samples == 40
    assert np.bincount(train.labels).tolist() == [10, 10, 10, 10]
    assert train.data_loader.batch_size == 128 and test.data_loader.batch_size == 128
    assert isinstance(train.data_loader.sampler, RandomSampler)
    assert isinstance(test.data_loader.sampler, SequentialSampler)


def _sample_and_order(train):
    """Which training samples were drawn, and their order in the first epoch."""
    return list(train.data_loader.dataset.indices), torch.cat([y for _, y in train.data_loader]).tolist()


def test_the_seed_fixes_the_training_sample_and_its_order(tmp_path):
    """The sample and the batch order depend on the seed only, not on NumPy's or PyTorch's global generators; NumPy's
    is not changed either.
    """
    labels = torch.arange(4).repeat_interleave(40)
    torch.save(TensorDataset(torch.randn(160, 3, 4, 4), labels), tmp_path / 'train.pt')
    torch.save(TensorDataset(torch.randn(20, 3, 4, 4), torch.arange(20) % 4), tmp_path / 'test.pt')
    paths = str(tmp_path / 'train.pt'), str(tmp_path / 'test.pt')

    runs = []
    for global_seed in (1, 2):
        np.random.seed(global_seed)
        torch.manual_seed(global_seed)
        numpy_state = np.random.get_state()
        train, _ = load_dataset(*paths, batch_size=8, seed=5)
        runs.append(_sample_and_order(train))
        np.testing.assert_equal(np.random.get_state(), numpy_state)
    assert runs[0] == runs[1]
    assert np.bincount(labels[runs[0][0]].numpy()).tolist() == [10, 10, 10, 10]

    other, _ = load_dataset(*paths, batch_size=8, seed=6)
    assert _sample_and_order(other)[0] != runs[0][0]


def test_imagenet_sample_follows_the_seed(tmp_path):
    _image_folder(tmp_path / 'train', per_class=8)
    _image_folder(tmp_path / 'val')
    numpy_state = np.random.get_state()
    samples = [list(load_imagenet(str(tmp_path), num_workers=0, seed=3)[0].data_loader.dataset.indices)
               for _ in range(2)]
    assert samples[0] == samples[1]
    np.testing.assert_equal(np.random.get_state(), numpy_state)


@pytest.mark.parametrize('seed', [-1, 1.5])
def test_invalid_seed_is_rejected(tmp_path, seed):
    torch.save(TensorDataset(torch.randn(8, 3, 4, 4), torch.arange(8) % 2), tmp_path / 'train.pt')
    with pytest.raises(ValueError, match='seed'):
        load_dataset(str(tmp_path / 'train.pt'), str(tmp_path / 'train.pt'), seed=seed)


def test_bundled_models_are_the_papers_cifar_models():
    assert sorted(get_available_models()) == ['ResNet56', 'VGG16', 'resnet56', 'vgg16']
