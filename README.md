# CoSeP: Complementary Separability Pruning via Class-Separability Clustering

<p>
  <a href="https://arxiv.org/abs/2505.13225"><img src="https://img.shields.io/badge/arXiv-2505.13225-b31b1b.svg" alt="arXiv:2505.13225"></a>
  <a href="https://arxiv.org/abs/2505.13225"><img src="https://img.shields.io/badge/BMVC-2026-1f6feb.svg" alt="BMVC 2026"></a>
  <a href="#license-and-patent-notice"><img src="https://img.shields.io/badge/license-PolyForm%20Noncommercial%201.0.0-lightgrey.svg" alt="License: PolyForm Noncommercial 1.0.0"></a>
  <img src="https://img.shields.io/badge/python-3.9%2B-3776ab.svg" alt="Python 3.9+">
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg" alt="PyTorch 2.0+">
</p>

Official PyTorch implementation of **CoSeP**, accepted at the **British Machine Vision Conference (BMVC) 2026**.

**David Levin** and **Gonen Singer**, Faculty of Engineering, Bar-Ilan University, Israel

[Paper (arXiv:2505.13225)](https://arxiv.org/abs/2505.13225) · [Citation](#citation) ·
[License and patent notice](#license-and-patent-notice)

> [!IMPORTANT]
> **Patent pending.** The CoSeP method is the subject of a pending patent application. This code is released for
> research and other noncommercial use under the [PolyForm Noncommercial License 1.0.0](LICENSE); commercial use
> requires a separate license. See [License and patent notice](#license-and-patent-notice).

## Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Command-line usage](#command-line-usage)
- [Arguments](#arguments)
- [Outputs](#outputs)
- [Project structure](#project-structure)
- [Citation](#citation)
- [License and patent notice](#license-and-patent-notice)

## Overview

Structured pruning methods often score each filter or neuron on its own, so the components they keep can capture the
same discriminative pattern while others are lost; and the share of each layer to keep is usually tuned by hand, layer
by layer. CoSeP addresses both.

It represents every component of a layer (a convolution channel or a linear neuron) by its **class-separability
profile**: its Jeffries–Matusita (JM) distance for every pair of classes. In this separability space, nearby
components are potentially redundant and distant ones complementary. CoSeP maps the profiles to 2-D with t-SNE,
groups the components on this map with **k-medoids**, scores every candidate number of groups with the
**Mean Simplified Silhouette (MSS)**, and takes the **knee** of that curve as the number of components to keep, so no
per-layer pruning ratio has to be chosen. Each group keeps one representative, and all other components are removed.

CoSeP prunes `Conv1d`, `Conv2d` and `Linear` layers one at a time, in forward order, with a brief fine-tuning after
each. It keeps whole the first and the last layer, layers with fewer than five components, and layers that share
their output channels with other layers, such as those that write into a residual sum.

The paper evaluates CoSeP on CIFAR-10, CIFAR-100 and ImageNet-1K with ResNet, VGG, MobileNet and DenseNet
architectures; see the [paper](https://arxiv.org/abs/2505.13225) for the method in full and for the results.

## Installation

CoSeP needs Python 3.9 or later and PyTorch 2.0 or later. It runs on a GPU when one is available, and otherwise on the
CPU. If you need a specific CUDA build of PyTorch, install it first by following [pytorch.org](https://pytorch.org).
Then, from the root of this repository:

```bash
pip install -e .                    # the cosep package, its dependencies and the cosep-prune command
# or, to run the scripts in place without installing the package:
pip install -r requirements.txt
```

## Quick start

This example prunes a ResNet-56 pretrained on CIFAR-10. First, save CIFAR-10 as two `TensorDataset`s of normalised
images:

```python
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import TensorDataset

normalize = T.Compose([T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
for split, train in (("train", True), ("test", False)):
    data = torchvision.datasets.CIFAR10("data", train=train, download=True, transform=normalize)
    images = torch.stack([image for image, _ in data])
    torch.save(TensorDataset(images, torch.tensor(data.targets)), f"cifar10_{split}.pt")
```

Then prune the model, here one from [chenyaofo/pytorch-cifar-models](https://github.com/chenyaofo/pytorch-cifar-models)
loaded with `torch.hub`:

```python
import torch
from cosep.data_types.model_entity import ModelEntity
from cosep.pruning.pruner import Pruner
from cosep.utils.data_utils import load_dataset

model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet56", pretrained=True)
train_data, test_data = load_dataset("cifar10_train.pt", "cifar10_test.pt", seed=0)  # 25% of the training data

Pruner(ModelEntity(model=model, model_path=None), train_data, test_data, seed=0).prune()
```

The pruned model, its pruning plan and a per-layer summary are written to `outputs/<timestamp>/` (see
[Outputs](#outputs)). With the default settings this takes a few hours on a CPU, and much less on a GPU. To see the
pipeline run in about ten minutes on a CPU, pass `train_fraction=0.02` to `load_dataset` and `retrain_epochs=0` to
`Pruner`: without fine-tuning, the pruned model's accuracy collapses (to about 11% in our run), so use this only to
try the pipeline.

To prune your own model from the command line, save it whole with `torch.save(model, "model.pt")` and run:

```bash
python pruning_algorithm.py --model_path model.pt \
    --train_data_path cifar10_train.pt --test_data_path cifar10_test.pt
```

## Command-line usage

```text
python pruning_algorithm.py --model_path PATH
    (--train_data_path PATH --test_data_path PATH | --imagenet_path DIR) [options]
```

After `pip install -e .`, the same command is available as `cosep-prune`. `python pruning_algorithm.py --help` lists
every option.

```bash
# A model saved whole with torch.save, and CIFAR-10 saved as TensorDatasets (.pt)
python pruning_algorithm.py --model_path model.pt \
    --train_data_path cifar10_train.pt --test_data_path cifar10_test.pt

# The bundled ResNet-56, from a state_dict, with its own training run's weight decay
python pruning_algorithm.py --model_path resnet56.pt --model_arch resnet56 --weight_decay 1e-4 \
    --train_data_path cifar10_train.pt --test_data_path cifar10_test.pt

# ImageNet, with the paper's ImageNet fine-tuning settings
python pruning_algorithm.py --model_path resnet50.pt --imagenet_path /data/imagenet \
    --retrain_epochs 3 --batch_size 256 --weight_decay 1e-4

# Prune more (a higher polynomial degree), augment the fine-tuning batches, and use another seed
python pruning_algorithm.py --model_path model.pt \
    --train_data_path cifar10_train.pt --test_data_path cifar10_test.pt --poly_degree 3 --augment --seed 1
```

**Model files.** `--model_path` takes a whole model saved with `torch.save(model, path)`, whose class must be
importable when the command runs (run it from the folder of your model's code, or add that folder to `PYTHONPATH`), or
the `state_dict` of a bundled architecture, given with `--model_arch`. Model and dataset files are loaded with
`torch.load(..., weights_only=False)`, which runs the code they contain: load only files you trust.

**Datasets.** A training and a test set saved with `torch.save` as PyTorch datasets of `(image, label)` pairs, such as
`TensorDataset`s of normalised images with integer class labels, or an ImageNet folder. The training data are both the
calibration data, from which the class statistics are computed, and the fine-tuning data.

## Arguments

The options are grouped as in `--help`. A flag written `--name` / `--no-name` turns a setting on or off.

**Model**

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_path PATH` | *required* | Model file (`.pt` or `.pth`): a whole model saved with `torch.save`, or a `state_dict` of a bundled architecture (then give `--model_arch`). |
| `--model_arch ARCH` | — | Bundled architecture of a `state_dict` file: `vgg16` or `resnet56`, the CIFAR-10 VGG-16 and ResNet-56 of the paper's experiments (`VGG16` and `ResNet56` are accepted too). |
| `--num_classes N` | `10` | Number of classes of the bundled architecture (at least 2). |

**Data** (either both `.pt` files, or `--imagenet_path`)

| Argument | Default | Description |
|----------|---------|-------------|
| `--train_data_path PATH` | — | Training dataset: a PyTorch dataset of `(image, label)` pairs saved with `torch.save`. |
| `--test_data_path PATH` | — | Test dataset, saved the same way; used to report the accuracy before, during and after pruning. |
| `--imagenet_path DIR` | — | ImageNet folder with `train/` and `val/` (or `validation/`) class subfolders. Images are resized to 256, centre-cropped to 224 and normalised with the ImageNet mean and standard deviation. |

**Method**

| Argument | Default | Description |
|----------|---------|-------------|
| `--poly_degree P` | `2` | Degree p of the polynomial fitted to the MSS curve (paper, Eq. 13). It sets the overall pruning level: p = 2 keeps about half of each layer, higher degrees keep fewer components. At least 2. |
| `--max_classes M` | `100` | With more classes than M, each layer builds its profiles from the M classes it separates best (paper, Appendix A); `0` always uses every class. |
| `--ignore_layers INDICES` | none | Indices of layers not to prune, e.g. `3,12` (or `3 12`). Layers are numbered from 0 in the order the model defines them, counting `Conv1d`, `Conv2d`, `Linear` and `MultiheadAttention` layers; the log shows each layer's index and name. |
| `--seed N` | `0` | Seed of the random steps: which training images are used and their order, t-SNE and k-medoids, the shifts and flips of `--augment`, and dropout. |

**Fine-tuning after each pruned layer** (paper, Appendix C). The defaults are a starting point: set them for your
model and data, for example to the momentum, weight decay and Nesterov setting of the model's own training run.

| Argument | Default | Description |
|----------|---------|-------------|
| `--retrain_epochs N` | `2` | Epochs of fine-tuning after each pruned layer; `0` skips fine-tuning. |
| `--train_fraction F` | `0.25` | Fraction of the training data to use, sampled per class, in (0, 1]: the calibration data and the fine-tuning data. |
| `--batch_size N` | `128` | Batch size; the training data are shuffled. |
| `--lr LR` | `0.001` | SGD learning rate; low on purpose, as the network is already trained. |
| `--momentum M` | `0.9` | SGD momentum. |
| `--weight_decay WD` | `5e-4` | SGD weight decay. |
| `--nesterov` / `--no-nesterov` | off | Nesterov momentum (needs a momentum above 0). |
| `--augment` / `--no-augment` | off | Augment the fine-tuning batches: shift each image by up to 4 pixels and flip half of them horizontally. For natural images; the calibration data are never augmented. |

**Output**

| Argument | Default | Description |
|----------|---------|-------------|
| `--verbose` / `--no-verbose` | on | Plots, debug messages on the console and progress bars. The log file always has the debug messages. |

## Outputs

Each run writes to its own folder, `outputs/<timestamp>/` (with `_2`, `_3`, ... added when runs start in the same
second):

| File | Contents |
|------|----------|
| `original_model.pt` | The model before pruning (state_dict checkpoint) |
| `pruned_model.pt` | The pruned model, saved whole (architecture and weights); load it with `torch.load(path, weights_only=False)` |
| `pruning_plan.json` | Which components of each layer were kept, and the run's settings (seed, p, M, fine-tuning) |
| `pruning_summary.csv` | Sizes, parameters and MACs of each layer and of the whole model, before and after pruning |
| `log.log` | Run log; from Python, only after `cosep.config.init_logger()` (the command line calls it) |
| `plots/` | With `--verbose`: for each pruned layer, its weights before and after pruning (the kept weights, before fine-tuning) and its components on the t-SNE map, coloured by their mean JM distance |

## Project structure

```text
.
├── cosep/
│   ├── config.py               # Run output folder, logging, verbose mode
│   ├── data_types/             # Containers: model and data entities, metrics, per-class activation statistics
│   ├── graph/
│   │   ├── separability.py     # JM distance, separability profiles and class screening
│   │   ├── mss.py              # Mean Simplified Silhouette
│   │   └── clustering.py       # k-medoids, the MSS sweep, the knee and the representatives
│   ├── models/                 # The paper's CIFAR-10 VGG-16 and ResNet-56 (for --model_arch)
│   ├── pruning/
│   │   ├── pruner.py           # Pruner: layer-by-layer selection, removal and fine-tuning; the output files
│   │   ├── structure.py        # Channel removal, layer rules, MAC counting, replaying a pruning plan
│   │   └── pruners/            # Layer pruners (Conv1d, Conv2d, Linear)
│   └── utils/                  # Data and model loading, fine-tuning and evaluation, statistics, plots, seeds
├── docs/                       # Project page (static site, served with GitHub Pages)
├── tests/                      # Test suite (pytest)
├── pruning_algorithm.py        # Command line (also installed as cosep-prune)
├── pyproject.toml              # Package metadata, dependencies, lint settings
├── requirements.txt            # Dependencies, for running without installing
├── CITATION.cff                # Citation metadata
└── LICENSE                     # PolyForm Noncommercial License 1.0.0, with the copyright and patent notices
```

## Citation

If you use this code or build on CoSeP, please cite the paper:

```bibtex
@inproceedings{levin2026cosep,
  title         = {CoSeP: Complementary Separability Pruning via Class-Separability Clustering},
  author        = {Levin, David and Singer, Gonen},
  booktitle     = {British Machine Vision Conference (BMVC)},
  year          = {2026},
  eprint        = {2505.13225},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV}
}
```

GitHub's "Cite this repository" button gives the same reference, from [CITATION.cff](CITATION.cff).

## License and patent notice

This code is released under the [PolyForm Noncommercial License 1.0.0](LICENSE). You may use, modify and share it
for research, teaching, personal study and other noncommercial purposes; noncommercial organisations, such as
universities and public research institutions, may use it whatever their source of funding. Commercial use is not
permitted under this license. When you share the code, pass on the license and the `Required Notice:` lines at the top
of [LICENSE](LICENSE).

**Patent notice.** The CoSeP method is the subject of a pending patent application. Using this code under the license
gives no rights beyond those the license states. For commercial use of the code or of the method, please contact the
authors.
