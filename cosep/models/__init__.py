"""Example model architectures: the CIFAR-10 VGG-16 and ResNet-56 of the paper's experiments."""
from cosep.models.cifar_resnet import resnet56
from cosep.models.cifar_vgg import VGG16

__all__ = ['VGG16', 'resnet56']
