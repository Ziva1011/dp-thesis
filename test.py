# %%
# %%
import hydra
from pathlib import Path
import glob
import os
import tempfile
from skimage.io import imread
from torch.utils import data
import nibabel as nib
import numpy as np
import torch
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss
from tqdm import tqdm, trange
from torchmetrics.classification import MulticlassF1Score
import matplotlib.pyplot as plt


from dptraining.config import Config
from dptraining.config.config_store import load_config_store
from dptraining.datasets.nifti.creator import NiftiSegCreator

from torchinfo import summary

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
)
from monai.utils import first
import SimpleITK as sitk

import wandb

from unet import UNet
from trainer import Trainer


load_config_store()
print(Path.cwd())

class SegmentationDataSet(data.Dataset):
    def __init__(self,
                 inputs: list,
                 targets: list,
                 transform=None
                 ):
        self.inputs = inputs
        self.targets = targets
        self.transform = transform
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.long

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self,
                    index: int):
        # Select the sample
        input_ID = self.inputs[index]
        target_ID = self.targets[index]

        # Load input and target
        x, y = sitk.ReadImage(input_ID), sitk.ReadImage(target_ID)
        x = sitk.GetArrayFromImage(x).astype(np.float32)
        y = sitk.GetArrayFromImage(y).astype(np.float32)
        # Preprocessing
        if self.transform is not None:
            x, y = self.transform(x, y)

        # Typecasting
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        return x, y

@hydra.main(version_base=None, config_path=Path.cwd() / "configs")

def main(config: Config):
    print(config)
    # %%
    # train_ds, val_ds, test_ds = NiftiSegCreator.make_datasets(
    #     config, (None, None, None)
    # )
    # train_dl, val_dl, test_dl = NiftiSegCreator.make_dataloader(
    #     train_ds, val_ds, test_ds, {}, {}, {}
    # )
    # # train_dl, val_dl, test_dl = make_loader_from_config(config)
    # # train_dl_torch = torch.from_numpy(train_dl)
    # print(type(train_dl))
    # x, y = next(iter(train_dl))
    images = sorted(glob.glob("/media/datasets/MSD/Task03_Liver/imagesTr/liver_*.nii.gz"))
    segs = sorted(glob.glob("/media/datasets/MSD/Task03_Liver/labelsTr/liver_*.nii.gz"))

    training_dataset = SegmentationDataSet(inputs=images,
                                       targets=segs,
                                       transform=None)

    training_dataloader = data.DataLoader(dataset=training_dataset,
                                        batch_size=2,
                                        shuffle=True)
    
    x, y = next(iter(training_dataloader))
    # binary_loss = "DiceLoss"
    print("Hello")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # train_config = {}
    # # train_config.update({
    # #     # get device to train on
    # #     'device': device,
    # #     'loss': {'name': binary_loss, 'weight': np.random.rand(2).astype(np.float32), 'pos_weight': 3.},
    # #     'eval_metric': {'name': val_metric}
    # # })

    # loaders = {"train": train_dl, "val": val_dl}

    # loss_criterion = get_loss_criterion(train_config)
    # eval_criterion = get_evaluation_metric(train_config)
    # model = get_class("UNet3D", modules=["pytorch3dunet.unet3d.model"])
    # model = model.to(device)
    # y_p = model(x.to(device))

    # formatter = DefaultTensorboardFormatter()
    # optimizer = torch.optim.Adam(
    #     model.parameters(), lr=0.003, betas=(0.9, 0.999), weight_decay=0
    # )
    # unet = UNetTrainer(
    #     model=model,
    #     optimizer=optimizer,
    #     lr_scheduler=None,
    #     loss_criterion=loss_criterion,
    #     eval_criterion=eval_criterion,
    #     loaders=loaders,
    #     tensorboard_formatter=formatter,
    #     resume=None,
    #     pre_trained=None,
    # )
    # unet.train()


if __name__ == "__main__":
    main()
# %%
