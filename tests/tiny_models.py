"""Small networks used by the pruning tests."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyNet(nn.Module):
    """A small network with every structure the pruner must handle.

    Layers, in order: stem (first layer) -> residual block (res_conv1 is inner, res_conv2 writes into the residual
    sum) -> expand -> depthwise -> project (an inverted-residual block without a skip) -> two dense layers whose
    outputs are concatenated -> fc1 -> BatchNorm1d -> fc2 (last layer).
    """

    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.stem = nn.Conv2d(3, 8, 3, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(8)
        self.res_conv1 = nn.Conv2d(8, 12, 3, padding=1, bias=False)
        self.res_bn1 = nn.BatchNorm2d(12)
        self.res_conv2 = nn.Conv2d(12, 8, 3, padding=1, bias=False)
        self.res_bn2 = nn.BatchNorm2d(8)
        self.expand = nn.Conv2d(8, 16, 1, bias=False)
        self.expand_bn = nn.BatchNorm2d(16)
        self.depthwise = nn.Conv2d(16, 16, 3, padding=1, groups=16, bias=False)
        self.depthwise_bn = nn.BatchNorm2d(16)
        self.project = nn.Conv2d(16, 12, 1, bias=False)
        self.project_bn = nn.BatchNorm2d(12)
        self.dense1 = nn.Conv2d(12, 8, 3, padding=1)
        self.dense2 = nn.Conv2d(20, 8, 3, padding=1)
        self.fc1 = nn.Linear(28, 16)
        self.fc1_bn = nn.BatchNorm1d(16)
        self.fc2 = nn.Linear(16, num_classes)

    def forward(self, x):
        x = F.relu(self.stem_bn(self.stem(x)))
        x = F.relu(x + self.res_bn2(self.res_conv2(F.relu(self.res_bn1(self.res_conv1(x))))))
        h = F.relu(self.expand_bn(self.expand(x)))
        h = F.relu(self.depthwise_bn(self.depthwise(h)))
        x = self.project_bn(self.project(h))
        x = torch.cat([x, F.relu(self.dense1(x))], 1)
        x = torch.cat([x, F.relu(self.dense2(x))], 1)
        x = torch.flatten(F.adaptive_avg_pool2d(x, 1), 1)
        return self.fc2(F.relu(self.fc1_bn(self.fc1(x))))


class FlattenNet(nn.Module):
    """Convolutions followed by a flatten of a 2x2 map: each channel feeds four inputs of the linear layer."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 8, 3, padding=1)
        self.fc1 = nn.Linear(8 * 2 * 2, 5)
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.adaptive_max_pool2d(F.relu(self.conv2(x)), 2)
        return self.fc2(F.relu(self.fc1(torch.flatten(x, 1))))


def randomise_batchnorm(model: nn.Module, seed: int = 0) -> nn.Module:
    """Give every BatchNorm layer non-trivial statistics and affine parameters."""
    g = torch.Generator().manual_seed(seed)
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            with torch.no_grad():
                m.running_mean.copy_(torch.rand(m.num_features, generator=g) - 0.5)
                m.running_var.copy_(torch.rand(m.num_features, generator=g) + 0.5)
                m.weight.copy_(torch.rand(m.num_features, generator=g) + 0.5)
                m.bias.copy_(torch.rand(m.num_features, generator=g) - 0.5)
    return model
