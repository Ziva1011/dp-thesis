#%%
import torch
import glob
from acsconv.converters import ACSConverter
import segmentation_models_pytorch as smp
from segmentation_models.linknet import Linknet
import segmentation_models_pytorch as smp
import torchvision.models as models


from monai.data import (
    Dataset,
    DataLoader,
    list_data_collate,
    pad_list_data_collate,
    ArrayDataset,
)

from opacus import validators
from monai.utils import first 
from torchinfo import summary

#%%

# %%
data = torch.randn(1, 1, 128, 128, 64)

#model_2d = smp.FPN(in_channels=1, classes=3, encoder_weights=None)
# model_2d = smp.PSPNet(in_channels=1, classes=3)
# model_2d = smp.PAN(in_channels=1, classes=3)
# model_2d = smp.DeepLabV3(in_channels=1, classes=3)
# model_2d = smp.DeepLabV3Plus(in_channels=1, classes=3)

#%%
#model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
res_net= models.resnet18(pretrained=True)
# %%
#model_2d = smp.Linknet(in_channels=1, classes=3)
#summary(model_2d, input_size=(1, 1, 128, 128))
#model_2d = smp.PSPNet(in_channels=1, classes=3)
#model_2d = smp.FPN(in_channels=1, classes=3, encoder_weights=None)
model_2d = smp.Linknet(in_channels=1, classes=3)
model_3d = Linknet(in_channels=1, classes=3)
out = model_3d(data)
#model = ACSConverter(model_2d)

# model= VNet(
#             spatial_dims=3, 
#             in_channels=1, 
#             out_channels=3)
# model = validators.ModuleValidator.fix(model)
    


#%%
#model.train()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

#%%


#%%
#input, target = (x.to(device), y.to(device))
#out = model(data)
# %%
