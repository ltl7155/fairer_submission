from functools import partial
from typing import Type, Any, Callable, Union, List, Optional

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
# from ..transforms._presets import ImageClassification
# from ..utils import _log_api_usage_once
# from torchvision.models._api import WeightsEnum, Weights
# from torchvision.models._meta import _IMAGENET_CATEGORIES
# from torchvision.models._utils import handle_legacy_interface, _ovewrite_named_param

from layers.gate_layer import GateLayer


# __all__ = [
#     "ResNet",
#     # "resnet18",
# ]
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x



def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        gate=None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)

        self.gate1 = GateLayer(planes,planes,[1, -1, 1, 1])

        
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.gate2 = GateLayer(planes,planes,[1, -1, 1, 1])

        self.downsample = downsample
        self.stride = stride
        self.gate = gate

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.gate1(out)

        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.gate2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        if self.gate is not None:
            out = self.gate(out)


        return out




class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        skip_gate = True,
    ) -> None:
        super().__init__()
        # _log_api_usage_once(self)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        
        gate = skip_gate
        self.gate = gate
        if gate:
            # self.gate_skip1 = GateLayer(64,64,[1, -1, 1, 1])
            self.gate_skip64 = GateLayer(64*4,64*4,[1, -1, 1, 1])
            self.gate_skip128 = GateLayer(128*4,128*4,[1, -1, 1, 1])
            self.gate_skip256 = GateLayer(256*4,256*4,[1, -1, 1, 1])
            self.gate_skip512 = GateLayer(512*4,512*4,[1, -1, 1, 1])
            if block == BasicBlock:
                self.gate_skip64 = GateLayer(64, 64, [1, -1, 1, 1])
                self.gate_skip128 = GateLayer(128, 128, [1, -1, 1, 1])
                self.gate_skip256 = GateLayer(256, 256, [1, -1, 1, 1])
                self.gate_skip512 = GateLayer(512, 512, [1, -1, 1, 1])
        else:
            self.gate_skip64 = None
            self.gate_skip128 = None
            self.gate_skip256 = None
            self.gate_skip512 = None
        
        
        
        self.layer1 = self._make_layer(block, 64, layers[0],gate = self.gate_skip64)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0],gate=self.gate_skip128)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1],gate=self.gate_skip256)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2],gate=self.gate_skip512)
        self.avgpool =Identity()# nn.AdaptiveAvgPool2d((1, 1))
        self.fc =Identity()# nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
        gate = None,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer,
                gate = gate
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    gate = gate,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x#x.view(-1, 512, 8, 8)

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet(
    block: Type[Union[BasicBlock]],
    layers: List[int],
    weights,
    progress: bool,
    **kwargs: Any,
) -> ResNet:

    model = ResNet(block, layers, **kwargs)


    return model


# _COMMON_META = {
#     "min_size": (1, 1),
#     "categories": _IMAGENET_CATEGORIES,
# }
class ResNet18_Encoder(nn.Module):
    def __init__(self, pretrained,**kwargs):
        super().__init__()
        self.resnet = _resnet(BasicBlock, [2, 2, 2, 2], weights=None, progress=False, **kwargs)
        self.resnet.fc = Identity()
        self.resnet.avgpool = Identity()

    def forward(self, x):
        outputs = self.resnet(x)
        return outputs.view(-1, 512, 8, 8)

class ResNet10_Encoder(nn.Module):
    def __init__(self, pretrained,**kwargs):
        super().__init__()
        self.resnet = _resnet(BasicBlock, [1, 1, 1, 1], weights=None, progress=False, **kwargs)
        self.resnet.fc = Identity()
        self.resnet.avgpool = Identity()

    def forward(self, x):
        outputs = self.resnet(x)
        return outputs.view(-1, 512, 8, 8)
    
class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 1, dropout: float = 0.5) -> None:
        super().__init__()
#         _log_api_usage_once(self)
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return torch.sigmoid(x)



# # @handle_legacy_interface(weights=("pretrained", ResNet18_Weights.IMAGENET1K_V1))
# def resnet18(pretrained=False, weights = None, progress: bool = True, **kwargs: Any) -> ResNet:
#     """ResNet-18 from `Deep Residual Learning for Image Recognition <https://arxiv.org/pdf/1512.03385.pdf>`__.
#     """
#     # weights = ResNet18_Weights.verify(weights)
#
#     return _resnet(BasicBlock, [2, 2, 2, 2], weights, progress, **kwargs)

class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 1)
        self.relu = nn.ReLU()
        self.avg = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, x):
        x = self.avg(x).view(-1, 512)
        x = self.fc1(x)
        x = self.relu(x)
        outputs = self.fc2(x)
        return torch.sigmoid(outputs)
    
    
import torch.nn as nn
import torch
# class AlexNet_cifar(nn.Module):

#     def __init__(self, num_classes=1):
#         super(AlexNet_cifar, self).__init__()
        
#         self.features = nn.Sequential(
#               nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2),
#               nn.ReLU(inplace=True),
#               nn.MaxPool2d(kernel_size=2, stride=2),
#               nn.Conv2d(64, 192, kernel_size=3, padding=1),
#               nn.ReLU(inplace=True),
#               nn.MaxPool2d(kernel_size=2, stride=2),
#               nn.Conv2d(192, 384, kernel_size=3, padding=1),
#               nn.ReLU(inplace=True),
#               nn.Conv2d(384, 256, kernel_size=3, padding=1),
#               nn.ReLU(inplace=True),
#               nn.Conv2d(256, 256, kernel_size=3, padding=1),
#               nn.ReLU(inplace=True),
#               nn.MaxPool2d(kernel_size=2, stride=2),
#         )

        
#         self.classifier = nn.Sequential(
#             nn.Dropout(),
#             nn.Linear(256 * 4 * 4, 4096),
#             nn.ReLU(inplace=True),
#             nn.Dropout(),
#             #nn.Linear(4096, 4096),
#             #nn.ReLU(inplace=True),
#             nn.Linear(4096, num_classes),
#         )

#     def forward(self, x):
#         feat_conv1 = self.features[0](x)
#         feat_conv1_relu = self.features[1](feat_conv1)
# #         print(feat_conv1_relu1.size())
#         feat_pool1 = self.features[2](feat_conv1_relu)
#         feat_conv2 = self.features[3](feat_pool1)
#         feat_conv2_relu = self.features[4](feat_conv2)
# #         print(feat_conv2_relu2.size())
#         feat_pool2 = self.features[5](feat_conv2_relu)
#         feat_conv3 = self.features[6](feat_pool2)
#         feat_conv3_relu = self.features[7](feat_conv3)
# #         print(feat_conv3_relu3.size())
#         feat_conv4 = self.features[8](feat_conv3_relu)
#         feat_conv4_relu = self.features[9](feat_conv4)
# #         print(feat_conv4_relu4.size())
#         feat_conv5 = self.features[10](feat_conv4_relu)
#         feat_conv5_relu = self.features[11](feat_conv5)
# #         print(feat_conv5_relu5.size())
#         feat_pool5 = self.features[12](feat_conv5_relu)
# #         print(feat_pool5.size())
        
#         x = feat_pool5.view(feat_pool5.size(0), -1)
#         y = self.classifier(x)
#         return torch.sigmoid(y)


class Net_cifar(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.sigmoid(x)
