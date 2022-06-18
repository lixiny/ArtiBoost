from collections import OrderedDict
from typing import Dict

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from anakin.utils.builder import BACKBONE, MODEL
from anakin.utils.logger import logger
from anakin.utils.misc import CONST, enable_lower_param, param_size
from torch.nn import functional as F

__all__ = ["ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152"]

model_urls = {
    "ResNet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "ResNet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "ResNet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "ResNet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "ResNet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.
    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def forward(self, x: torch.Tensor):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, bn_layer=nn.BatchNorm2d):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = bn_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = bn_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, bn_layer=nn.BatchNorm2d):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = bn_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = bn_layer(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = bn_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, interm_feat=True, **kwargs):
        super(ResNet, self).__init__()
        if kwargs["FREEZE_BATCHNORM"]:
            self.bn_layer = FrozenBatchNorm2d
        else:
            self.bn_layer = nn.BatchNorm2d
        self.inplanes = 64
        if "INTERM_FEAT" in kwargs:
            self.interm_feat = kwargs["INTERM_FEAT"]
        else:
            self.interm_feat = interm_feat
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = self.bn_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.features = 512 * block.expansion

        self.output_channel = self.inplanes

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, self.bn_layer):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                self.bn_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.bn_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def load_pretrained(self):
        logger.info(f"Loading {type(self).__name__} pretrained models")
        self.load_state_dict(model_zoo.load_url(model_urls[type(self).__name__]))
        logger.info(f"{type(self).__name__} has {param_size(self)}M parameters")

    def forward(self, **kwargs) -> Dict:
        x = kwargs["image"]
        features = OrderedDict()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        features["res_layer1"] = x

        x = self.layer2(x)
        features["res_layer2"] = x

        x = self.layer3(x)
        features["res_layer3"] = x

        x = self.layer4(x)
        features["res_layer4"] = x

        x = x.mean(3).mean(2)
        x = x.view(x.size(0), -1)
        features["res_layer4_mean"] = x

        if self.interm_feat:
            return features

        x = self.fc(x)

        out = {"res_output": x}
        out.update(features)
        return out


@BACKBONE.register_module
class ResNet18(ResNet):
    @enable_lower_param
    def __init__(self, **cfg):
        super().__init__(BasicBlock, [2, 2, 2, 2], **cfg)
        if cfg["PRETRAINED"]:
            self.load_pretrained()


@BACKBONE.register_module
class ResNet34(ResNet):
    @enable_lower_param
    def __init__(self, **cfg):
        super().__init__(BasicBlock, [3, 4, 6, 3], **cfg)
        if cfg["PRETRAINED"]:
            self.load_pretrained()


@BACKBONE.register_module
class ResNet50(ResNet):
    @enable_lower_param
    def __init__(self, **cfg):
        super().__init__(Bottleneck, [3, 4, 6, 3], **cfg)
        if cfg["PRETRAINED"]:
            self.load_pretrained()


@BACKBONE.register_module
class ResNet101(ResNet):
    @enable_lower_param
    def __init__(self, **cfg):
        super().__init__(Bottleneck, [3, 4, 23, 3], **cfg)
        if cfg["PRETRAINED"]:
            self.load_pretrained()


@BACKBONE.register_module
class ResNet152(ResNet):
    @enable_lower_param
    def __init__(self, **cfg):
        super().__init__(Bottleneck, [3, 8, 36, 3], **cfg)
        if cfg["PRETRAINED"]:
            self.load_pretrained()
