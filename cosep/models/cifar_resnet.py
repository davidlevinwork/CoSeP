"""Pre-activation ResNet for CIFAR-10: the ResNet-56 of the paper's experiments."""
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Pre-activation basic residual block for CIFAR ResNets.

    BatchNorm, ReLU and a 3x3 convolution, twice. The shortcut has no parameters: it is average-pooled when the block
    downsamples and zero-padded with channels when the block widens.

    Args:
        in_planes: Number of input channels.
        planes: Number of output channels.
        stride: Stride of the first convolution (2 to halve the resolution).
    """

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)

        self.stride = stride
        self.in_planes = in_planes
        self.planes = planes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the block.

        Args:
            x: Input feature map, shape (N, in_planes, H, W).

        Returns:
            Output feature map, shape (N, planes, H / stride, W / stride).
        """
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))

        # Handle shortcut for dimension mismatch
        shortcut = x
        if self.stride != 1:
            shortcut = F.avg_pool2d(shortcut, self.stride)
        if self.in_planes != self.planes:
            # Pad channels with zeros
            pad = (self.planes - self.in_planes) // 2
            shortcut = F.pad(shortcut, (0, 0, 0, 0, pad, pad))

        out += shortcut
        return out


class CifarResNet(nn.Module):
    """Pre-activation ResNet for 32x32 images.

    A 3x3 convolution, three stages of BasicBlocks with 16, 32 and 64 channels, then BatchNorm, ReLU, global average
    pooling and a 1x1 convolution as the classifier.

    Args:
        num_blocks: Number of blocks in each of the three stages.
        num_classes: Number of output classes.
    """

    def __init__(self, num_blocks: Sequence[int], num_classes: int = 10) -> None:
        super().__init__()
        self.in_planes = 16

        self.init_conv = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)

        self.block1 = self._make_layer(16, num_blocks[0], stride=1)
        self.block2 = self._make_layer(32, num_blocks[1], stride=2)
        self.block3 = self._make_layer(64, num_blocks[2], stride=2)

        self.final_bn = nn.BatchNorm2d(64)
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1, bias=True)

    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        """One stage: `num_blocks` blocks with `planes` output channels, the first with the given stride."""
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify a batch of images.

        Args:
            x: Images, shape (N, 3, 32, 32).

        Returns:
            Class scores (logits), shape (N, num_classes).
        """
        out = self.init_conv(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = F.relu(self.final_bn(out))
        out = F.avg_pool2d(out, out.size()[3])
        out = self.final_conv(out)
        return out.view(out.size(0), -1)


def resnet56(num_classes: int = 10) -> CifarResNet:
    """Build a ResNet-56 for CIFAR: three stages of nine blocks.

    Args:
        num_classes: Number of output classes.

    Returns:
        The model, with freshly initialised weights.
    """
    return CifarResNet([9, 9, 9], num_classes)
