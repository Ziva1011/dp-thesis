# %%
from pathlib import Path
from monai import data as mdt
from monai.data.image_dataset import ImageDataset
from monai import transforms as mtf
from matplotlib import pyplot as plt
from tqdm import tqdm
import pandas as pd
import glob

from random import seed
from numpy.random import seed as npseed
from torch.random import manual_seed as tseed

from monai.data import Dataset, DataLoader, list_data_collate, pad_list_data_collate, ArrayDataset
from monai.transforms import (
    RandFlip,
    Compose,
    LoadImage,
    RandRotate90,
    RandAdjustContrast,
    ScaleIntensity,
    Lambda,
    ToTensor,
    EnsureChannelFirst,
    RandSpatialCrop,
    Resize,
)

# %%
SEED = 0
seed(SEED)
npseed(SEED)
tseed(SEED)

# %%
# normal_path = Path("/mnt/seagate-4t/shared/NormalPancreas/normal_selected/")  
# tumour_path = Path("/mnt/seagate-4t/shared/PDAC_CT/CTs/")
save_path = Path("./data/liver_seg")
save_path_labels = Path("./data/liver_seg_labels")
# # %%
# # ignore = ["HARRISON^ARON^RENA.nii", "ADAMS^CHARMAINE^ROSEMARY.nii", ""]
# bad_cts_df = pd.read_csv("/mnt/seagate-4t/shared/Problematic_CTs.csv")
# ignore = bad_cts_df.Patient_name.tolist()
# # %%
# normal_ct_paths = [
#     p for p in normal_path.rglob("*.nii") if p.is_file() and not p.stem in ignore
# ]
# tumour_ct_paths = [
#     p for p in tumour_path.rglob("*.nii") if p.is_file() and not p.stem in ignore
# ]

images = sorted(glob.glob("/media/datasets/MSD/Task03_Liver/imagesTr/liver_*.nii.gz"))
segs = sorted(glob.glob("/media/datasets/MSD/Task03_Liver/labelsTr/liver_*.nii.gz"))

# %%
transforms = Compose([
    LoadImage(image_only=True),
    EnsureChannelFirst(),
    Resize(spatial_size=(128,128,64)),
])
train_ds = ArrayDataset(images, transforms, segs, transforms)

# %%
# x_normal = train_ds[0]

# # %%
# fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
# axs[0].imshow(x_normal[0][0, :, :, 35], cmap="gray")
# axs[1].imshow(x_normal[1][0, :, :, 35], cmap="gray")
# axs[0].set_xlabel("Normal")
# axs[1].set_xlabel("Label")
# for ax in axs:
#     ax.set_yticklabels([])
#     ax.set_yticks([])
#     ax.set_xticklabels([])
#     ax.set_xticks([])
# fig.show()
#plt.savefig("output2.png")
# %%
# normal_dataloader = mdt.dataloader.DataLoader(
#     normal_dataset, num_workers=32, batch_size=1, prefetch_factor=2, shuffle=True
# )
# tumour_dataloader = mdt.dataloader.DataLoader(
#     tumour_dataset, num_workers=32, batch_size=1, prefetch_factor=2, shuffle=True
# )
train_dl = DataLoader(train_ds, batch_size=1, num_workers=2, pin_memory='True')

# %%
test_split = 0.2
val_split = 0.2
splits = ["train", "val", "test"]
# %%
writer = mdt.image_writer.NibabelWriter()


# %%
def save_all_cts_in_dataloader(save_path, dataloader, writer, name):
    test_imgs = int(round(len(dataloader) * test_split))
    val_imgs = int(round(len(dataloader) * val_split))
    train_imgs = len(dataloader) - val_imgs - test_imgs 
    assert train_imgs > 0
    for i, array in tqdm(
        enumerate(dataloader),
        desc=f"Saving {name} CTs",
        leave=False,
        total=len(dataloader),
    ):
        writer.set_data_array(array[0].squeeze(0), channel_dim=0)
        split: int
        if i < train_imgs:
            split = 0
        elif train_imgs <= i < val_imgs + train_imgs:
            split = 1
        else:
            split = 2
        save_folder = save_path / splits[split]
        
        #save_folder = save_path
        if not save_folder.is_dir():
            save_folder.mkdir(parents=True)
        writer.write(save_folder / f"{i}.nii")

        writer.set_data_array(array[1].squeeze(0), channel_dim=0)
        save_folder = save_path_labels / splits[split]
        if not save_folder.is_dir():
            save_folder.mkdir(parents=True)
        writer.write(save_folder / f"{i}.nii")



# %%
# save_all_cts_in_dataloader(save_path, normal_dataloader, writer, "normal")
# %%
save_all_cts_in_dataloader(save_path, train_dl, writer, "tumour")
# %%
# normal_paths = [p for p in (save_path / "normal").rglob("*.nii") if p.is_file()]
# tumour_paths = [p for p in (save_path / "tumour").rglob("*.nii") if p.is_file()]
# paths = normal_paths + tumour_paths
# labels = [0 for _ in normal_paths] + [1 for _ in tumour_paths]
# final_dataset = ImageDataset(paths, labels)
# final_dataloader = mdt.dataloader.DataLoader(
#     final_dataset, num_workers=16, batch_size=4, prefetch_factor=4, shuffle=True
# )
# %%