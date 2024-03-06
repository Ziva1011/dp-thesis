import torch.nn as nn
from torch import Tensor

from segmentation_models_pytorch.base import modules

try:
    from inplace_abn import InPlaceABN
except ImportError:
    InPlaceABN = None

class BasicBlock(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding=1,
        stride=1,
        use_batchnorm=True,
    ):

        if use_batchnorm == "inplace" and InPlaceABN is None:
            raise RuntimeError(
                "In order to use `use_batchnorm='inplace'` inplace_abn package must be installed. "
                + "To install see: https://github.com/mapillary/inplace_abn"
            )

        conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=1,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        if use_batchnorm == "inplace":
            bn = InPlaceABN(out_channels, activation="leaky_relu", activation_param=0.0)
            relu = nn.Identity()

        elif use_batchnorm and use_batchnorm != "inplace":
            bn = nn.BatchNorm3d(out_channels)

        else:
            bn = nn.Identity()

        conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=1,bias=not (use_batchnorm))

        downsample = nn.Sequential (
            nn.Conv3d(in_channels, out_channels, kernel_size = 1, stride=2, dilation=1,bias=not (use_batchnorm)),
            nn.BatchNorm3d(out_channels)
        )

        maxpool = nn.MaxPool3d(kernel_size = 1, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)

        if (in_channels== out_channels):
            conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=1,bias=not (use_batchnorm))
        
            super(BasicBlock, self).__init__(conv, bn, relu, conv2, bn)
        else:
            conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=2,
            padding=padding,
            dilation=1,
            bias=not (use_batchnorm),
        )
            conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, stride=1, padding=padding, dilation=1,bias=not (use_batchnorm))
        
            super(BasicBlock, self).__init__(conv, bn, relu, conv2, bn, maxpool)

class BasicBlock2(nn.Module):
    def __init__(self,
        in_channels,
        out_channels,
        kernel_size,
        padding=1,
        stride=1,
        use_batchnorm=True,
    ):
        super().__init__()
        self.stride = stride
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
            bias=not (use_batchnorm),
        )
        self.relu = nn.ReLU(inplace=True)

        self.bn = nn.BatchNorm3d(out_channels)

        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, padding=padding, dilation=1,bias=not (use_batchnorm))

        self.downsample = nn.Sequential (
            nn.Conv3d(in_channels, out_channels, kernel_size = 1, stride=2, dilation=1,bias=not (use_batchnorm)),
            nn.BatchNorm3d(out_channels)
        )

        self.maxpool = nn.MaxPool3d(kernel_size = 1, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)


    def forward(self, x: Tensor) -> Tensor:

        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn(out)

        if self.stride != 1:
            x = self.downsample(x)

        out += x
        out = self.relu(out)

        return out


class TransposeX2(nn.Sequential):
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        super().__init__()
        layers = [
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        ]

        if use_batchnorm:
            layers.insert(1, nn.BatchNorm3d(out_channels))

        super().__init__(*layers)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_blocks, use_batchnorm=True):
        super().__init__()

        self.block = nn.Sequential()

        self.block.add_module("0",BasicBlock2(in_channels, out_channels, kernel_size=3, use_batchnorm=use_batchnorm))

        for i in range(n_blocks-1):
            self.block.add_module(str(i+1), BasicBlock2(out_channels, out_channels, kernel_size=3, stride= 2, use_batchnorm=use_batchnorm))

    def forward(self, x):
        x = self.block(x)

        return x




class ResNetEncoder2(nn.Module):
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


        self.conv = nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm3d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        #self.layer1 = EncoderBlock(64, 64, 3, use_batchnorm=True),
        self.layer1 = self._make_layer(3, 64, 64, stride=1)
        self.layer2 = self._make_layer(4, 64, 128, stride=2)
        self.layer3 = self._make_layer(6, 128, 256, stride=2)
        self.layer4 = self._make_layer(3, 256, 512, stride=2)

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

    # def forward(self, x):
    #     identity = x
    #     #stages = self.get_stages()
    #     x = self.conv(x)
    #     x = self.bn(x)
    #     x = self.relu(x)
    #     x = self.maxpool(x)
    #     x = self.layer1(x)
    #     x = self.layer2(x)
    #     x = self.layer3(x)
    #     x = self.layer4(x)

    #     features = x
    #     #x = self.layer1(x)
    #     features = [
    #         identity,
    #     ] + [features]

    #     return features
    
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
           BasicBlock2(in_channels, out_channels, kernel_size=3, stride=stride)
        )
        
        for _ in range(1, blocks):
            layers.append(
                BasicBlock2(out_channels, out_channels, kernel_size=3)
            )
            

        return nn.Sequential(*layers)
        # features = []
        # for i in range(self._depth + 1):
        #     x = stages[i](x)
        #     features.append(x)

        # return features


class ResnetEncoder(nn.Module):
    def __init__(
        self, pretrained=True, in_channels=3, depth=5, output_stride=32, use_batchnorm=True
        
    ):
        super().__init__()

        # remove first skip
        #encoder_channels = encoder_channels[1:]
        # reverse channels to start from head of encoder
        #encoder_channels = encoder_channels[::-1]

        #channels = list(encoder_channels) + [prefinal_channels]
        # conv1 = nn.Conv3d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # bn1 = nn.BatchNorm3d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # relu = nn.ReLU(inplace=True)
        # maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        # layer1 = nn.Sequential(
        # layer2 = 
        # layer3 = 
        # layer4 = 
        
        self._out_channels = (3, 64, 64, 128, 256, 512)

        # self.conv3d = nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3,bias=False)
        # self.batch = nn.BatchNorm3d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # self.relu = nn.ReLU(inplace=True)
        # self.max = nn.MaxPool3d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        # self.block1 = EncoderBlock(64, 64, 3, use_batchnorm=use_batchnorm)
        
        self.model = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3,bias=False),
            nn.BatchNorm3d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False),
            #[EncoderBlock(encoder_channels, 64, use_batchnorm=use_batchnorm) for i in range(n_blocks)]
            EncoderBlock(64, 64, 3, use_batchnorm=use_batchnorm),
            EncoderBlock(64, 128, 4, use_batchnorm=use_batchnorm),
            EncoderBlock(128, 256, 6, use_batchnorm=use_batchnorm),
            EncoderBlock(256, 512, 3, use_batchnorm=use_batchnorm),
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


        self._in_channels = in_channels
        # self._out_channels = [in_channels,] + self.model.feature_info.channels()
        self._depth = depth
        self._output_stride = output_stride
        #super(LinknetEncoder, self).__init__(conv1, bn1, relu, maxpool, blocks)

    def forward(self, x):

        features = self.model(x)
        features = [
            x,
        ] + [features]
        return features


    @property
    def out_channels(self):
        return self._out_channels

    @property
    def output_stride(self):
        return min(self._output_stride, 2**self._depth)

