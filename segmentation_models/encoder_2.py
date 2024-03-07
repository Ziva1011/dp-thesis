import torch.nn as nn
from torch import Tensor

from segmentation_models_pytorch.base import modules

try:
    from inplace_abn import InPlaceABN
except ImportError:
    InPlaceABN = None


class BasicBlock(nn.Module):
    def __init__(self,
        in_channels,
        out_channels,
        kernel_size,
        padding=1,
        stride=1,
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
            padding=padding,
            dilation=1,
            bias=False,
        )
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, padding=padding, dilation=1, bias=False)
        if stride != 1:
            self.downsample = nn.Sequential (
                nn.Conv3d(in_channels, out_channels, kernel_size = 1, stride=2, dilation=1, bias=False),
                nn.BatchNorm3d(out_channels)
            )
        self.stride = stride


    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn(out)

        if self.stride != 1:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out



class ResNetEncoder(nn.Module):
    def __init__(self, pretrained=True, in_channels=3, depth=5, output_stride=32):
        super().__init__()
        self._depth = depth
        self._out_channels = (1, 64, 64, 128, 256, 512)
        self._in_channels = in_channels
        self.output_stride = 32


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


        self.conv = nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn = nn.BatchNorm3d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.layer1 = self._make_layer(blocks=3, in_channels=64, out_channels=64, stride=1)
        self.layer2 = self._make_layer(blocks=4, in_channels=64, out_channels=128, stride=2)
        self.layer3 = self._make_layer(blocks=6, in_channels=128, out_channels=256, stride=2)
        self.layer4 = self._make_layer(blocks=3, in_channels=256, out_channels=512, stride=2)

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

        layers = []
        layers.append(
           BasicBlock(in_channels, out_channels, kernel_size=3, stride=stride)
        )
        
        for _ in range(1, blocks):
            layers.append(
                BasicBlock(out_channels, out_channels, kernel_size=3)
            )
            

        return nn.Sequential(*layers)



