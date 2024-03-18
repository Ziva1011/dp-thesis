from typing import Optional, Union

from segmentation_models_pytorch.base import (
    SegmentationModel,
    ClassificationHead,
)

#from .encoder import get_encoder
#from segmentation_models__pytorch.segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.encoders import get_encoder
from .decoder import LinknetDecoder

from ..encoder_2 import ResNetEncoder
import torch.nn as nn
from segmentation_models_pytorch.base.modules import Activation
from ..heads import SegmentationHead


class Linknet(SegmentationModel):
    """Linknet_ is a fully convolution neural network for image semantic segmentation. Consist of *encoder*
    and *decoder* parts connected with *skip connections*. Encoder extract features of different spatial
    resolution (skip connections) which are used by decoder to define accurate segmentation mask. Use *sum*
    for fusing decoder blocks with skip connections.

    Note:
        This implementation by default has 4 skip connections (original - 3).

    Args:
        encoder_name: Name of the classification model that will be used as an encoder (a.k.a backbone)
            to extract features of different spatial resolution
        encoder_depth: A number of stages used in encoder in range [3, 5]. Each stage generate features
            two times smaller in spatial dimensions than previous one (e.g. for depth 0 we will have features
            with shapes [(N, C, H, W),], for depth 1 - [(N, C, H, W), (N, C, H // 2, W // 2)] and so on).
            Default is 5
        encoder_weights: One of **None** (random initialization), **"imagenet"** (pre-training on ImageNet) and
            other pretrained weights (see table with available weights for each encoder_name)
        decoder_use_batchnorm: If **True**, BatchNorm2d layer between Conv2D and Activation layers
            is used. If **"inplace"** InplaceABN will be used, allows to decrease memory consumption.
            Available options are **True, False, "inplace"**
        in_channels: A number of input channels for the model, default is 3 (RGB images)
        classes: A number of classes for output mask (or you can think as a number of channels of output mask)
        activation: An activation function to apply after the final convolution layer.
            Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**,
                **callable** and **None**.
            Default is **None**
        aux_params: Dictionary with parameters of the auxiliary output (classification head). Auxiliary output is build
            on top of encoder if **aux_params** is not **None** (default). Supported params:
                - classes (int): A number of classes
                - pooling (str): One of "max", "avg". Default is "avg"
                - dropout (float): Dropout factor in [0, 1)
                - activation (str): An activation function to apply "sigmoid"/"softmax"
                    (could be **None** to return logits)

    Returns:
        ``torch.nn.Module``: **Linknet**

    .. _Linknet:
        https://arxiv.org/abs/1707.03718
    """

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = True,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, callable]] = None,
        aux_params: Optional[dict] = None,
    ):
        super().__init__()

        if encoder_name.startswith("mit_b"):
            raise ValueError("Encoder `{}` is not supported for Linknet".format(encoder_name))

        # self.encoder = get_encoder(
        #    encoder_name,
        #    in_channels=in_channels,
        #    depth=encoder_depth,
        #    weights=encoder_weights,
        # )

        # self.encoder = ResnetEncoder(
        #     in_channels=in_channels,
        #     depth=encoder_depth,
        #     output_stride=32,
        #     use_batchnorm=decoder_use_batchnorm,
        # )
        self.encoder = ResNetEncoder(
            in_channels=in_channels,
            depth=encoder_depth,
            output_stride=32)

        self.decoder = LinknetDecoder(
            encoder_channels=self.encoder._out_channels,
            n_blocks=encoder_depth,
            prefinal_channels=32,
            use_batchnorm=decoder_use_batchnorm,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=32, 
            out_channels=classes, 
            activation=activation, 
            kernel_size=1
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(in_channels=self.encoder._out_channels[-1], **aux_params)
        else:
            self.classification_head = None

        self.name = "link-{}".format(encoder_name)
        self.initialize()
