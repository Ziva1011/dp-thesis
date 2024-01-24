import torch.nn as nn

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
        padding=0,
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

        super(BasicBlock, self).__init__(conv, bn, relu, conv, bn)



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

        self.block.add_module("0",BasicBlock(in_channels, out_channels, kernel_size=1, use_batchnorm=use_batchnorm))

        for i in range(n_blocks-1):
            self.block.add_module(str(i+1), BasicBlock(out_channels, out_channels, kernel_size=1, use_batchnorm=use_batchnorm))

    def forward(self, x, skip=None):
        x = self.block(x)
        if skip is not None:
            x = x + skip
        return x


class LinknetEncoder(nn.Module):
    def __init__(
        self,
        encoder_channels,
        prefinal_channels=32,
        n_blocks=5,
        use_batchnorm=True,
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
        self.out_channels = [1, 64, 64, 128, 256, 512]
        self.blocks = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            nn.BatchNorm3d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False),
            #[EncoderBlock(encoder_channels, 64, use_batchnorm=use_batchnorm) for i in range(n_blocks)]
            EncoderBlock(64, 64, 3, use_batchnorm=use_batchnorm),
            EncoderBlock(64, 128, 4, use_batchnorm=use_batchnorm),
            EncoderBlock(128, 256, 6, use_batchnorm=use_batchnorm),
            EncoderBlock(256, 512, 3, use_batchnorm=use_batchnorm),
        )

        #super(LinknetEncoder, self).__init__(conv1, bn1, relu, maxpool, blocks)


    def forward(self, *features):
        features = features[1:]  # remove first skip
        features = features[::-1]  # reverse channels to start from head of encoder

        x = features[0]
        skips = features[1:]

        for i, encoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = encoder_block(x, skip)

        return x
