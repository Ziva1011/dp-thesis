#%%
import torch
import glob
from acsconv.converters import ACSConverter
import segmentation_models_pytorch as smp
from monai.data import (
    Dataset,
    DataLoader,
    list_data_collate,
    pad_list_data_collate,
    ArrayDataset,
)
from monai.utils import first 
#%%

# %%
train_files = glob.glob("./data/liver_seg/train/*.nii")
train_labels = glob.glob("./data/liver_seg_labels/train/*.nii")
train_files = [
        {"img": img, "seg": seg} for img, seg in zip(train_files, train_labels)
    ]

#%%
train_ds = Dataset(data=train_files, transform=None)
train_dl = DataLoader(
    train_ds,
    batch_size=4,
    shuffle=True,
    num_workers=2,
    collate_fn=list_data_collate,
    pin_memory=torch.cuda.is_available(),
)
# %%
model_2d = smp.Linknet(in_channels=1, classes=3)
model = ACSConverter(model_2d)


#%%
dict= first(iter(train_dl))
model.train()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

x= dict["img"]
y= dict["seg"]

#%%
#input, target = (x.to(device), y.to(device))
out = model(x)
# %%
