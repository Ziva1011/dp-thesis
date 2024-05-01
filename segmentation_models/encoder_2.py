import torch.nn as nn
from torch import Tensor

from typing import Any, Callable, List, Optional, Type, Union

from segmentation_models_pytorch.base import modules
from segmentation_models_pytorch.encoders._base import EncoderMixin

try:
    from inplace_abn import InPlaceABN
except ImportError:
    InPlaceABN = None


class BasicBlock(nn.Module):
    def __init__(self,
        in_channels,
        out_channels,
        kernel_size,
        downsample: Optional[nn.Module] = None,
        padding=1,
        stride=1,
        dilation=1,
        use_batchnorm=True,
    ) -> None:
        super().__init__()
        
        if use_batchnorm == "inplace" and InPlaceABN is None:
            raise RuntimeError(
                "In order to use `use_batchnorm='inplace'` inplace_abn package must be installed. "
                + "To install see: https://github.com/mapillary/inplace_abn"
            )
        
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=True,
        )
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, padding=dilation, dilation=dilation, bias=True)
        self.bn2 = nn.BatchNorm3d(out_channels)

        self.downsample = downsample
     
        self.stride = stride


    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out



class ResNetEncoder(nn.Module, EncoderMixin):
    def __init__(self, pretrained=True, in_channels=3, depth=5, output_stride=32, replace_stride_with_dilation: Optional[List[bool]] = None):
        super().__init__()
        self._depth = depth
        out_channels = (1, 64, 64, 128, 256, 512)
        self._out_channels = out_channels[:depth+1]
        self._in_channels = in_channels
        self.dilation = 1
        self.inplanes = 64

        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )

        kwargs = dict(
            in_chans=in_channels,
            features_only=True,
            output_stride=output_stride,
            pretrained=pretrained,
            out_indices=tuple(range(depth)),
        )

        # not all models support output stride argument, drop it by default
        if output_stride == 32:
            kwargs.pop("output_stride")


        self.conv = nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3)
        self.bn = nn.BatchNorm3d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1 )
        self.layer1 = self._make_layer(blocks=3, in_channels=64, out_channels=64, stride=1)
        self.layer2 = self._make_layer(blocks=4, in_channels=64, out_channels=128, stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(blocks=6, in_channels=128, out_channels=256, stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(blocks=3, in_channels=256, out_channels=512, stride=2, dilate=replace_stride_with_dilation[2])

    def get_stages(self):
        return [
            nn.Identity(),
            nn.Sequential(self.conv, self.bn, self.relu),
            nn.Sequential(self.maxpool, self.layer1),
            self.layer2,
            self.layer3,
            self.layer4,
        ]

    def forward(self, x):
        stages = self.get_stages()

        features = []
        for i in range(self._depth + 1):
            x = stages[i](x)
            features.append(x)

        return features

    def _make_layer(
        self,
        blocks: int,
        in_channels,
        out_channels,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:

        downsample = None

        if dilate:
            self.dilation *= stride
            stride = 1

        if stride != 1 or self.inplanes != in_channels:
            downsample = nn.Sequential (
                nn.Conv3d(in_channels, out_channels, kernel_size = 1, stride=stride, bias=False, dilation=self.dilation),
                nn.BatchNorm3d(out_channels)
            )
        layers = []
        layers.append(
           BasicBlock(in_channels, out_channels, kernel_size=3, downsample=downsample, stride=stride, dilation=self.dilation)
        )
        
        for _ in range(1, blocks):
            layers.append(
                BasicBlock(out_channels, out_channels, kernel_size=3, dilation=self.dilation)
            )
            

        return nn.Sequential(*layers)



