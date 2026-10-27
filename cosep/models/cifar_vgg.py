"""VGG-16 for CIFAR-10: the VGG-16 of the paper's experiments.

Ported from https://github.com/SeHwanJoo/cifar10-vgg16.
"""

import torch
import torch.nn as nn


class ConvBNRelu(nn.Module):
    """A convolution (3x3 by default), ReLU, then BatchNorm (the order of the original Keras model), and dropout.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Size of the convolution kernel.
        dropout_rate: Probability of dropping a channel, when dropout is used.
        use_dropout: Apply channel dropout (nn.Dropout2d) after the BatchNorm.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 dropout_rate: float = 0.4, use_dropout: bool = True) -> None:
        super().__init__()
        self.use_dropout = use_dropout

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=1, padding='same', bias=True)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=dropout_rate) if use_dropout else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the block.

        Args:
            x: Input feature map, shape (N, in_channels, H, W).

        Returns:
            Output feature map, shape (N, out_channels, H, W).
        """
        x = self.conv(x)
        x = self.relu(x)
        x = self.bn(x)
        if self.use_dropout and self.dropout is not None:
            x = self.dropout(x)
        return x


class VGG16(nn.Module):
    """VGG-16 adapted to CIFAR-10 (32x32 images).

    Thirteen convolution blocks (ConvBNRelu) in five stages of 64, 128, 256, 512 and 512 channels, each stage ending
    in 2x2 max pooling, then dropout, a 512-unit linear layer with BatchNorm and ReLU, and the output layer. Dropout
    rates are 0.3-0.5, as in the original model, which was trained with weight decay 5e-4.

    Args:
        num_classes: Number of output classes.
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()

        # Block 1: 64 filters
        self.conv1 = ConvBNRelu(3, 64, dropout_rate=0.3)
        self.conv2 = ConvBNRelu(64, 64, use_dropout=False)
        self.pool1 = nn.MaxPool2d(2, 2)

        # Block 2: 128 filters
        self.conv3 = ConvBNRelu(64, 128, dropout_rate=0.4)
        self.conv4 = ConvBNRelu(128, 128, use_dropout=False)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Block 3: 256 filters
        self.conv5 = ConvBNRelu(128, 256, dropout_rate=0.4)
        self.conv6 = ConvBNRelu(256, 256, dropout_rate=0.4)
        self.conv7 = ConvBNRelu(256, 256, use_dropout=False)
        self.pool3 = nn.MaxPool2d(2, 2)

        # Block 4: 512 filters
        self.conv11 = ConvBNRelu(256, 512, dropout_rate=0.4)
        self.conv12 = ConvBNRelu(512, 512, dropout_rate=0.4)
        self.conv13 = ConvBNRelu(512, 512, use_dropout=False)
        self.pool4 = nn.MaxPool2d(2, 2)

        # Block 5: 512 filters
        self.conv14 = ConvBNRelu(512, 512, dropout_rate=0.4)
        self.conv15 = ConvBNRelu(512, 512, dropout_rate=0.4)
        self.conv16 = ConvBNRelu(512, 512, use_dropout=False)
        self.pool5 = nn.MaxPool2d(2, 2)

        # Classifier
        self.classifier_dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(512, 512)
        self.bn = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify a batch of images.

        Args:
            x: Images, shape (N, 3, 32, 32).

        Returns:
            Class scores (logits), shape (N, num_classes).
        """
        # Block 1
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)

        # Block 2
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)

        # Block 3
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.pool3(x)

        # Block 4
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.pool4(x)

        # Block 5
        x = self.conv14(x)
        x = self.conv15(x)
        x = self.conv16(x)
        x = self.pool5(x)

        # Classifier
        x = self.classifier_dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.bn(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)

        return x


def vgg16(num_classes: int = 10) -> VGG16:
    """Build a VGG-16 for CIFAR.

    Args:
        num_classes: Number of output classes.

    Returns:
        The model, with freshly initialised weights.
    """
    return VGG16(num_classes=num_classes)
