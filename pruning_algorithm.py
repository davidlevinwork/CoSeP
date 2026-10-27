"""CoSeP: Complementary Separability Pruning via Class-Separability Clustering (command-line interface).

Run `python pruning_algorithm.py --help` (or `cosep-prune --help` after `pip install .`) for the options.
"""
import argparse
from typing import Callable, Optional, TypeVar

import torch

import cosep.config as config
from cosep.config import init_logger
from cosep.pruning.pruner import Pruner
from cosep.utils.data_utils import get_available_models, load_dataset, load_imagenet, load_model

NumberT = TypeVar("NumberT", int, float)


def main() -> None:
    """Parse the command line, load the model and the data, and prune the model."""
    args = parse_arguments()
    init_logger(verbose=args.verbose)

    config.verbose = args.verbose
    torch.manual_seed(args.seed)  # dropout and any other PyTorch randomness during fine-tuning

    # Load data - either from .pt files or ImageNet folder
    if args.imagenet_path:
        train_data, test_data = load_imagenet(
            args.imagenet_path,
            batch_size=args.batch_size,
            train_fraction=args.train_fraction,
            seed=args.seed
        )
    else:
        train_data, test_data = load_dataset(
            args.train_data_path,
            args.test_data_path,
            batch_size=args.batch_size,
            train_fraction=args.train_fraction,
            seed=args.seed
        )

    model = load_model(args.model_path, model_arch=args.model_arch, num_classes=args.num_classes)

    Pruner(
        model=model,
        test_data=test_data,
        train_data=train_data,
        ignore_layers=args.ignore_layers,
        retrain_epochs=args.retrain_epochs,
        poly_degree=args.poly_degree,
        max_classes=args.max_classes,
        learning_rate=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=args.nesterov,
        seed=args.seed,
        augment=args.augment
    ).prune()


def _layer_indices(text: str) -> list[int]:
    """Parse layer indices separated by commas, such as '3,12' (an argparse type)."""
    try:
        return [int(part) for part in text.split(',') if part.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"invalid layer indices {text!r}: use whole numbers separated by commas, e.g. 3,12"
        ) from error


def _number(kind: Callable[[str], NumberT], check: Callable[[NumberT], bool],
            requirement: str) -> Callable[[str], NumberT]:
    """Build an argparse type: a number of the given kind that passes a check.

    Args:
        kind: int or float.
        check: Condition the number must meet.
        requirement: The condition in words, for the error message.

    Returns:
        A function that parses one command-line value, and raises argparse.ArgumentTypeError if it is not valid.
    """
    def parse(text: str) -> NumberT:
        try:
            value = kind(text)
        except ValueError as error:
            raise argparse.ArgumentTypeError(f"invalid value {text!r}: {requirement}") from error
        if not check(value):
            raise argparse.ArgumentTypeError(f"invalid value {text!r}: {requirement}")
        return value
    return parse


def parse_arguments(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse and check the command-line arguments.

    Args:
        argv: Arguments to parse; defaults to the command line.

    Returns:
        The parsed arguments; ignore_layers is a sorted list of unique layer indices.
    """
    available_models = get_available_models()
    parser = argparse.ArgumentParser(
        description='Prune a trained network with CoSeP: Complementary Separability Pruning via Class-Separability\n'
                    'Clustering (BMVC 2026, https://arxiv.org/abs/2505.13225). The README explains every option.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Fine-tuning after each pruned layer: 2 epochs on 25% of the training data, batch 128, SGD with learning rate 0.001,
momentum 0.9 and weight decay 5e-4 by default. Use the momentum, weight decay and Nesterov setting of the model's own
training run: the paper's VGG-16 used --nesterov, its ResNet-56 --weight_decay 1e-4. For natural images, --augment
shifts and flips the fine-tuning images (never the calibration data).

Examples:
  # A model saved whole with torch.save, and CIFAR-10 saved as TensorDatasets (.pt)
  python pruning_algorithm.py --model_path model.pt \\
    --train_data_path cifar10_train.pt --test_data_path cifar10_test.pt

  # The bundled ResNet-56 from a state_dict
  python pruning_algorithm.py --model_path resnet56.pt --model_arch resnet56 --weight_decay 1e-4 \\
    --train_data_path cifar10_train.pt --test_data_path cifar10_test.pt

  # ImageNet, with the paper's ImageNet settings
  python pruning_algorithm.py --model_path resnet50.pt --imagenet_path /data/imagenet \\
    --retrain_epochs 3 --batch_size 256 --weight_decay 1e-4
        """
    )

    model_group = parser.add_argument_group('Model')
    model_group.add_argument('--model_path', type=str, required=True, metavar='PATH',
                             help='Model file (.pt or .pth): a whole model saved with torch.save, or a state_dict of '
                                  'a bundled architecture (then give --model_arch). Required')
    model_group.add_argument('--model_arch', type=str, default=None, choices=available_models, metavar='ARCH',
                             help=f'Bundled architecture of a state_dict model file: {", ".join(available_models)}')
    model_group.add_argument('--num_classes', type=_number(int, lambda v: v >= 2, 'a whole number of at least 2'),
                             default=10, metavar='N',
                             help='Number of classes of a bundled architecture (with --model_arch; default: 10)')

    # Data loading options (either .pt files OR ImageNet path)
    data_group = parser.add_argument_group('Data (either both .pt files, or --imagenet_path)')
    data_group.add_argument('--train_data_path', type=str, default=None, metavar='PATH',
                            help='Training dataset: a torch Dataset of (image, label) pairs saved with torch.save')
    data_group.add_argument('--test_data_path', type=str, default=None, metavar='PATH',
                            help='Test dataset, saved the same way; used to report the accuracy')
    data_group.add_argument('--imagenet_path', type=str, default=None, metavar='DIR',
                            help='ImageNet folder with train/ and val/ (or validation/), instead of the .pt files')

    method = parser.add_argument_group('Method')
    method.add_argument('--poly_degree', type=_number(int, lambda v: v >= 2, 'a whole number of at least 2'),
                        default=2, metavar='P',
                        help='Degree p of the polynomial fitted to the MSS curve (paper, Eq. 13). It sets the overall '
                             'pruning level: p = 2 keeps about half of each layer, higher degrees keep fewer '
                             'components (default: 2)')
    method.add_argument('--max_classes', type=_number(int, lambda v: v == 0 or v >= 2, '0, or at least 2'),
                        default=100, metavar='M',
                        help='With more classes than M, each layer builds its profiles from the M classes it '
                             'separates best (paper, Appendix A); 0 always uses every class (default: 100)')
    method.add_argument('--ignore_layers', type=_layer_indices, nargs='+', default=[], metavar='INDICES',
                        help='Indices of layers not to prune, e.g. 3,12 (or 3 12). Layers are numbered from 0, '
                             'as in the log')
    method.add_argument('--seed', type=_number(int, lambda v: v >= 0, 'a whole number, 0 or more'), default=0,
                        metavar='N',
                        help='Seed of the random steps: which training images are used and their order, t-SNE and '
                             'k-medoids, the shifts and flips of --augment, and dropout. The same seed and settings '
                             'give the same pruned model on the same machine; on a GPU the last digits can differ '
                             '(default: 0)')

    tuning = parser.add_argument_group('Fine-tuning after each pruned layer (paper, Appendix C)')
    tuning.add_argument('--retrain_epochs', type=_number(int, lambda v: v >= 0, 'a whole number, 0 or more'),
                        default=2, metavar='N', help='Epochs of fine-tuning after each pruned layer (default: 2)')
    tuning.add_argument('--train_fraction', type=_number(float, lambda v: 0 < v <= 1, 'a number in (0, 1]'),
                        default=0.25, metavar='F',
                        help='Fraction of the training data to use, sampled per class: the calibration data and the '
                             'fine-tuning data (default: 0.25)')
    tuning.add_argument('--batch_size', type=_number(int, lambda v: v >= 1, 'a whole number, 1 or more'),
                        default=128, metavar='N', help='Batch size; the training data are shuffled (default: 128)')
    tuning.add_argument('--lr', type=_number(float, lambda v: v > 0, 'a positive number'), default=0.001,
                        metavar='LR',
                        help='SGD learning rate; low on purpose, as the network is already trained. Lower it if the '
                             'accuracy drops during the first fine-tuning round (default: 0.001)')
    tuning.add_argument('--momentum', type=_number(float, lambda v: v >= 0, 'a number, 0 or more'), default=0.9,
                        metavar='M', help='SGD momentum (default: 0.9)')
    tuning.add_argument('--weight_decay', type=_number(float, lambda v: v >= 0, 'a number, 0 or more'),
                        default=5e-4, metavar='WD', help='SGD weight decay (default: 5e-4)')
    tuning.add_argument('--nesterov', action=argparse.BooleanOptionalAction, default=False,
                        help='Nesterov momentum (default: off)')
    tuning.add_argument('--augment', action=argparse.BooleanOptionalAction, default=False,
                        help='Augment the fine-tuning batches: shift each image by up to 4 pixels (padding with 0) '
                             'and flip half of them horizontally. For natural images; the calibration data are never '
                             'augmented (default: off)')

    output = parser.add_argument_group('Output')
    output.add_argument('--verbose', action=argparse.BooleanOptionalAction, default=True,
                        help='Plots, debug messages on the console and progress bars (default: on). The log file '
                             'always has the debug messages')

    args = parser.parse_args(argv)
    args.ignore_layers = sorted({index for group in args.ignore_layers for index in group})

    # Validate: either .pt files or ImageNet path must be provided
    if args.imagenet_path is None:
        if args.train_data_path is None or args.test_data_path is None:
            parser.error("Either --imagenet_path OR both --train_data_path and --test_data_path are required")
    elif args.train_data_path is not None or args.test_data_path is not None:
        parser.error("Use either --imagenet_path or --train_data_path and --test_data_path, not both")
    if args.nesterov and args.momentum == 0:
        parser.error("--nesterov needs a momentum above 0")

    return args


if __name__ == '__main__':
    main()
