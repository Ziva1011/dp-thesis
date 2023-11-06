import torch
import glob

from segmentation_models_pytorch.base import (
    SegmentationHead,
    SegmentationModel,
    ClassificationHead,
)
from acsconv.operators import ACSConv
from torch.nn import functional as F
from acsconv.operators.functional import acs_conv_f
#from acsconv.operators import _ACSConv

from acsconv.converters import ACSConverter

model_1 = torch.nn.Sequential(torch.nn.ConvTranspose2d(1, 64, kernel_size=4, stride=(2, 2, 2), padding=(1, 1, 1)))
model_2 = torch.nn.ConvTranspose3d(1, 64, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1))


#model_2d=TransposedLayerModel()

#model_3d = TransposedLayerModel3D()
model_3d = ACSConverter(model_1)
data = torch.randn(1, 1, 128, 128, 64)
out = model_3d(data)
#x = ACSConverter(model_1)
arguments = type(model_1[0]).__init__.__code__.co_varnames[1:]

arguments = [
    a
    for a in arguments
    if a
    not in [
        "device",
        "dtype",
        "factory_kwargs",
        "kernel_size_",
        "stride_",
        "padding_",
        "dilation_",
    ]
]

kwargs = {}
for k in arguments:
    if k != 'stride':
        kwargs[k] = getattr(model_1[0], k)
    else:
        kwargs[k] = getattr(model_1[0], k)[0]


class ACSTransposeConv(ACSConv):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,   
        padding=0,
        dilation=1,
        groups=1,
        acs_kernel_split=None,
        bias=True,
        padding_mode="zeros",
        output_padding=0,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            True,
            output_padding,
            groups,
            bias,
            padding_mode,
        )
        if acs_kernel_split is None:
            if self.out_channels % 3 == 0:
                self.acs_kernel_split = (
                    self.out_channels // 3,
                    self.out_channels // 3,
                    self.out_channels // 3,
                )
            if self.out_channels % 3 == 1:
                self.acs_kernel_split = (
                    self.out_channels // 3 + 1,
                    self.out_channels // 3,
                    self.out_channels // 3,
                )
            if self.out_channels % 3 == 2:
                self.acs_kernel_split = (
                    self.out_channels // 3 + 1,
                    self.out_channels // 3 + 1,
                    self.out_channels // 3,
                )
        else:
            self.acs_kernel_split = acs_kernel_split

    def forward(self, x):
        return acs_conv_f(
            x,
            self.weight,
            self.bias,
            self.kernel_size,
            self.dilation,
            self.padding,
            self.stride,
            self.groups,
            self.out_channels,
            self.acs_kernel_split,
            output_padding=self.output_padding,
            transposed=True,
            conv_func=F.conv_transpose3d,
        )

layer = ACSTransposeConv(**kwargs)

print(model_1)
