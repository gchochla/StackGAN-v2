"""Pre-trained visual feature extractors."""

__all__ = ['set_parameter_requires_grad', 'C2INormalize', 'GoogleNet_dim',
           'Inception_dim', 'ResNet_dim', 'VGG_dim', 'GoogleNetAvgpool',
           'ResNetAvgpool', 'Inception3Avgpool', 'VGGAvgpool']

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import (  # pylint: disable=unused-import
    googlenet, inception_v3, resnet18, resnet34,
    resnet50, resnet101, resnet152, vgg16_bn, vgg11_bn,
    vgg13_bn, vgg19_bn
)

EXTRACTOR_MAPPING = {'googlenet': GoogleNetAvgpool, 'vgg': VGGAvgpool,
                     'inception': Inception3Avgpool, 'resnet': ResNetAvgpool}

def set_parameter_requires_grad(model, requires_grad=False):
    """Sets requires_grad for all the parameters in a model.

    Args:
        model(nn model): model to alter.
        requires_grad(bool): whether the model
            requires grad.
    """
    if not requires_grad:
        for param in model.parameters():
            param.requires_grad_(False)

class C2INormalize(nn.Module):
    """Normalizes images of CUB as they are loaded
    and produced (i.e. [-1, 1]) so they are as
    ImageNet models expect them. Operates on batches
    rather than on individual images. Make into a
    nn.Module so as to use in forward(). Note that values
    are solely based on ImageNet and Module has no trainable
    parameters.

    Attributes:
        mean(torch.nn.parameter.Parameter): supposed mean to aleviate.
        std(torch.nn.parameter.Parameter): supposed std to alleviate.
    """

    def __init__(self):
        """Init."""
        super().__init__()
        # Parameter to follow module's device
        # but require_grad False to keep const
        self.mean = nn.Parameter(torch.tensor((-0.03, -0.088, -0.188))[None, :, None, None],  # pylint: disable=not-callable
                                 requires_grad=False)
        self.std = nn.Parameter(torch.tensor((0.458, 0.448, 0.450))[None, :, None, None],  # pylint: disable=not-callable
                                requires_grad=False)

    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward propagation.

        Args:
            x(nn.Tensor): input batch.

        Returns:
            normalized x.
        """
        return (x - self.mean) / self.std

GoogleNet_dim = 1024
ResNet_dim = 2048  # NOTE: only for default
VGG_dim = 2048
Inception_dim = 2048

class ResNetAvgpool(nn.Module):
    """ResNet feature extractor.

    Features are extracted by Average Pooling layer of ResNets.
    Note that requires_grad is kept as `True` (because we use
    feature extractors as submodules in classifiers) and
    torch.no_grad() should be used to disable differentiation.

    Attributes:
        features(nn.Module): net doing the feature extraction.

    """

    def __init__(self, resnet_id=50):
        """Init.

        Args:
            resnet_id(int, optional): ResNet version to load,
                default=`50` (i.e. ResNet50).
        """

        super().__init__()

        assert resnet_id in (18, 34, 50, 101, 152), 'Invalid ResNet id'

        resnet = globals()['resnet{}'.format(resnet_id)](pretrained=True, progress=False)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.inp_trans = C2INormalize()

    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward propagation.

        Args:
            x(torch.Tensor): input image.

        Returns:
            512-dimensional representation if ResNet{18, 34},
            2048-dimensional representation else.
        """

        x = self.inp_trans(x)
        x = self.features(x)
        x = x.view(x.size(0), -1)  # 512, 512, 2048, 2048, 2048
        return x

class VGGAvgpool(nn.Module):
    """VGG feature extractor.

    Features are extracted from Average pooling layer of VGGs.
    Note that requires_grad is kept as `True` (because we use
    feature extractors as submodules in classifiers) and
    torch.no_grad() should be used to disable differentiation.

    Attributes:
        features(nn.Module): net doing the feature extraction.
        inp_trans(nn.Module): input transformation.
    """


    def __init__(self, vgg_year=16):
        """Init.

        Args:
            vgg_year(int, optional): VGG of which year,
                default=`16` (as in 2016).
        """

        super().__init__()

        assert vgg_year in (11, 13, 16, 19), 'Invalid VGG year'

        vgg = globals()['vgg{}_bn'.format(vgg_year)](pretrained=True, progress=False)
        self.features = nn.Sequential(*list(vgg.children())[:2])
        self.inp_trans = C2INormalize()

    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward propagation.

        Args:
            x(torch.Tensor): input image.

        Returns:
            2048-dimensional representation.
        """

        x = self.inp_trans(x)
        x = self.features(x)
        x = x.view(x.size(0), -1)  # 2048
        return x

class Inception3Avgpool(nn.Module):
    """Inception v3 feature extractor.

    Features are extracted from Average pooling layer of Inception v3.
    Note that requires_grad is kept as `True` (because we use
    feature extractors as submodules in classifiers) and
    torch.no_grad() should be used to disable differentiation.

    Attributes:
        full_model(nn.Module): the whole is kept because
            of F functions used in original forward impl.
        inp_trans(nn.Module): input transformation.
    """

    def __init__(self):
        """Init."""
        super().__init__()
        self.features = inception_v3(pretrained=True, progress=False)
        self.inp_trans = C2INormalize()

    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward propagation.

        Args:
            x(torch.Tensor): input image.

        Returns:
            2048-dimensional representation.
        """

        x = self.inp_trans(x)
        x = self.features.Conv2d_1a_3x3(x)
        x = self.features.Conv2d_2a_3x3(x)
        x = self.features.Conv2d_2b_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.features.Conv2d_3b_1x1(x)
        x = self.features.Conv2d_4a_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.features.Mixed_5b(x)
        x = self.features.Mixed_5c(x)
        x = self.features.Mixed_5d(x)
        x = self.features.Mixed_6a(x)
        x = self.features.Mixed_6b(x)
        x = self.features.Mixed_6c(x)
        x = self.features.Mixed_6d(x)
        x = self.features.Mixed_6e(x)
        x = self.features.Mixed_7a(x)
        x = self.features.Mixed_7b(x)
        x = self.features.Mixed_7c(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)  # 2048
        return x

class GoogleNetAvgpool(nn.Module):
    """GoogleNet feature extractor.

    Features are extracted from Average pooling layer of GoogleNet.
    Note that requires_grad is kept as `True` (because we use
    feature extractors as submodules in classifiers) and
    torch.no_grad() should be used to disable differentiation.

    Attributes:
        features(nn.Module): net doing the feature extraction.
        inp_trans(nn.Module): input transformation.
    """

    def __init__(self):
        """Init."""

        super().__init__()
        glenet = googlenet(pretrained=True, progress=False)
        self.features = nn.Sequential(*list(glenet.children())[:-2])
        self.inp_trans = C2INormalize()

    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward propagation.

        Args:
            x(torch.Tensor): input image.

        Returns:
            1024-dimensional representation.
        """

        x = self.inp_trans(x)
        x = self.features(x)
        x = x.view(x.size(0), -1)  # 1024
        return x
