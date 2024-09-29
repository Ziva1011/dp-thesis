# %%
import hydra
from pathlib import Path
import glob

# imports from  pytorch
import torch
import segmentation_models_pytorch as smp

# from segmentation_models_pytorch.losses import DiceLoss
# from torchmetrics.classification import MulticlassF1Score
# from torchinfo import summary
from deepee import ModelSurgeon
from kernel_norm import KernelNorm3d


from segmentation_models.linknet.linknet import Linknet
from segmentation_models.fpn.fpn import FPN
from segmentation_models.psp.psp import PSPNet
from segmentation_models.pan.pan import PAN
from segmentation_models.deeplabv3.deeplab import DeepLabV3, DeepLabV3Plus


# imports from dptraining
from dptraining.config import Config
from dptraining.config.config_store import load_config_store


# imports from monai
from monai.data import (
    Dataset,
    DataLoader,
    list_data_collate,
)
from monai.losses import DiceLoss, GeneralizedDiceLoss
from monai.transforms import (
    RandFlipd,
    Compose,
    LoadImaged,
    RandRotate90d,
    EnsureChannelFirstd,
    ScaleIntensityRanged,
    Resized, 
    RandSpatialCropd
)
from monai.utils import first
from monai.networks.nets import (
    DynUNet,
    SegResNetVAE,
    AttentionUnet,
    VNet,
    DiNTS,
    TopologyInstance,
    UNet,
    BasicUNetPlusPlus,
)

# imports from opacus
from opacus import PrivacyEngine
from opacus.data_loader import wrap_collate_with_empty
from opacus import validators
from opacus.validators import ModuleValidator

import matplotlib.pyplot as plt
from acsconv.converters import ACSConverter

from unet import Unet
from trainer import Trainer
from tester import Tester

import wandb

load_config_store()
print(Path.cwd())



def wrap_collate_with_empty(*, collate_fn, sample_empty_shapes, dtypes):
    def collate(batch):
        if len(batch) > 0:
            return collate_fn(batch)
        else:
            return {
                key: torch.zeros(sample_empty_shapes[key], dtype=dtypes[key]).unsqueeze(
                    0
                )
                for key in sample_empty_shapes.keys()
            }

    return collate


@hydra.main(version_base=None, config_path=Path.cwd() / "configs")
def main(config: Config):
    # print(config)
    wandb.init(
        # set the wandb project where this run will be logged
        project="dpSegmentation",

        # track hyperparameters and run metadata
        config={
        "learning_rate": 0.01,
        "architecture": "Unet",
        "epochs": 100,
        "loss": "Dice",
        }
    )

    # Transforms
    train_transforms = Compose(
        [
            # LoadImage(image_only=True),
            # EnsureChannelFirst(),
            LoadImaged(keys=["img", "seg"]),
            EnsureChannelFirstd(keys=["img", "seg"]),
            ScaleIntensityRanged(
                keys=["img"],
                a_min=-57,
                a_max=164,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            # Resized(spatial_size=(128, 64, 64), keys=["img"]),
            # Resized(spatial_size=(128, 64, 64), keys=["seg"], mode="nearest"),
            RandSpatialCropd(roi_size=(64, 64, 64), random_size=False, keys=["img", "seg"]),
            RandFlipd(keys=["img", "seg"], prob=0.1, spatial_axis=1),
            # RandAdjustContrast(prob=0.5, gamma=[0.5,0.6]),
            RandRotate90d(keys=["img", "seg"], prob=0.1, spatial_axes=(0, 1)),
        ]
    )

    val_transforms = Compose(
        [
            LoadImaged(keys=["img", "seg"]),
            EnsureChannelFirstd(keys=["img", "seg"]),
            # Resized(spatial_size=(128, 64, 64), keys=["img"]),
            # Resized(spatial_size=(128, 64, 64), keys=["seg"], mode="nearest"),
            ScaleIntensityRanged(
                keys=["img"], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True
            ),
            # CropForegroundd(keys=["image", "label"], source_key="image"),
        ]
    )
 
    train_files = glob.glob("./data2/liver_seg/train/*.nii")
    train_labels = glob.glob("./data2/liver_seg_labels/train/*.nii")

    val_files = glob.glob("./data2/liver_seg/val/*.nii")
    val_labels = glob.glob("./data2/liver_seg_labels/val/*.nii")

    test_files = glob.glob("./data2/liver_seg/test/*.nii")
    test_labels = glob.glob("./data2/liver_seg_labels/test/*.nii")

    train_files = [
        {"img": img, "seg": seg} for img, seg in zip(train_files, train_labels)
    ]
    val_files = [{"img": img, "seg": seg} for img, seg in zip(val_files, val_labels)]
    test_files = [{"img": img, "seg": seg} for img, seg in zip(test_files, test_labels)]

    train_ds = Dataset(data=train_files, transform=train_transforms)
    train_dl = DataLoader(
        train_ds,
        batch_size=4,
        shuffle=True,
        num_workers=4,
        collate_fn=list_data_collate,
        pin_memory=True,
        prefetch_factor=2,
    )

    val_ds = Dataset(data=val_files, transform=val_transforms)
    val_dl = DataLoader(
        val_ds, batch_size=1, num_workers=4, collate_fn=list_data_collate
    )

    test_ds = Dataset(data=test_files, transform=val_transforms)
    test_dl = DataLoader(
        test_ds, batch_size=1, num_workers=4, collate_fn=list_data_collate
    )

    private = config.private
    architecture = config.archi
    print("ARCHITECTURE: {} {} \n".format(architecture, private))

    if architecture == "dynUnet":
        learning_rate = 0.002
        model = DynUNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=3,
            kernel_size=[3, 3, 3, 3, 3, 3],
            strides=[1, 2, 2, 2, 2, [2, 2, 1]],
            upsample_kernel_size=[2, 2, 2, 2, [2, 2, 1]],
            norm_name="instance",
            deep_supervision=False,
            res_block=True,
        )

    elif architecture == "unet":
        learning_rate = 0.002
        model = Unet(
            in_channels=1,
            out_channels=3,
            n_blocks=4,
            start_filters=32,
            activation="mish",
            normalization="batch",
            conv_mode="same",
            dim=3,
        )

    elif architecture == "attention":
        learning_rate = 0.0001
        model = AttentionUnet(
            spatial_dims=3,
            in_channels=1,
            out_channels=3,
            channels=[64, 32, 16, 8, 4],
            strides=[1, 2, 2, 2, 2],
            dropout=0.2,
        )

    elif architecture == "vnet":
        learning_rate = 0.002
        model = VNet(spatial_dims=3, in_channels=1, out_channels=3)

    elif architecture == "unetMonai":
        learning_rate = 0.002
        model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=3,
            channels=[64, 32, 16, 8, 4],
            strides=[1, 2, 2, 2, 2],
        )

    elif architecture == "dints":
        learning_rate = 0.002
        model = DiNTS(dints_space=TopologyInstance(), in_channels=1, num_classes=3)

    elif architecture == "resnet":
        learning_rate = 0.002

        class NewModel(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, x):
                return self.model(x)[0]

        model = SegResNetVAE(
            input_image_size=(128, 128, 64), in_channels=1, out_channels=3
        )

    elif architecture == "unet++":
        learning_rate = 0.0002
        model_2d = smp.UnetPlusPlus(in_channels=1, classes=3)
        model = ACSConverter(model_2d)

    elif architecture == "unetpp":
        learning_rate = 0.0002
        model = BasicUNetPlusPlus(spatial_dims=3, in_channels=1, out_channels=3)

    elif architecture == "linknet":
        learning_rate = 0.002
        model = Linknet(in_channels=1, classes=3)

    elif architecture == "fpn":
        learning_rate = 0.002
        model = FPN(in_channels=1, classes=3, encoder_weights=None)

    elif architecture == "psp":
        learning_rate = 0.002
        model = PSPNet(in_channels=1, classes=3, encoder_weights=None)

    elif architecture == "pan":
        learning_rate = 0.002
        model = PAN(in_channels=1, classes=3)

    elif architecture == "deep":
        learning_rate = 0.002
        model = DeepLabV3(in_channels=1, classes=3)

    elif architecture == "deepPlus":
        learning_rate = 0.002
        model = DeepLabV3Plus(in_channels=1, classes=3)


    # Testing model
    # x = torch.randn(size=(1, 1, 512, 512, 512), dtype=torch.float32)
    # with torch.no_grad():
    #     out = model(x)
    # print(f'Out: {out.shape}')

    # print(model.parameters)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    model.train()

    def bn_to_kn(*_, **__):
        return KernelNorm3d(kernel_size=3, stride=3, padding=1)

    surgeon = ModelSurgeon(converter=bn_to_kn)
    model = surgeon.operate(model)

    #criterion
    # criterion = torch.nn.CrossEntropyLoss()
    # criterion = DiceLoss(mode, classes=None, log_loss=False, from_logits=True, smooth=0.0, ignore_index=None, eps=1e-07)
    criterion = DiceLoss(
        to_onehot_y=True, reduction="mean", softmax=True, weight=[1, 50, 10000]
    )

    # optimizer
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    optimizer = torch.optim.NAdam(model.parameters(), lr=learning_rate)


    trainSteps = len(train_ds)
    epochs = 100


    if private:

        privacy_engine = PrivacyEngine(accountant="gdp")
        model, optimizer, train_dl = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_dl,
            target_epsilon=8,
            target_delta=1e-2,
            epochs=epochs,
            max_grad_norm=1,
            grad_sample_mode="functorch",
        )
        train_dl.collate_fn = wrap_collate_with_empty(
            collate_fn=list_data_collate,
            sample_empty_shapes={
                x: train_dl.dataset[0][x].shape for x in ["img", "seg"]
            },
            dtypes={x: train_dl.dataset[0][x].dtype for x in ["img", "seg"]},
        )

    print("[INFO] training the network...")
    
    #Training loop
    trainer = Trainer(
        model=model,
        device=device,
        criterion=criterion,
        optimizer=optimizer,
        training_DataLoader=train_dl,
        validation_DataLoader=val_dl,
        lr_scheduler=None,
        epochs=epochs,
        epoch=0,
        notebook=False,
    )

    # # start training
    training_losses, validation_losses, lr_rates = trainer.run_trainer()
    print(training_losses, validation_losses)

    
    tester = Tester(
        model=model,
        device=device,
        criterion=criterion,
        optimizer=optimizer,
        validation_DataLoader=test_dl,
        lr_scheduler=None,
        notebook=False,
    )

    # # start training
    test_losses = tester.run_trainer()
    print(test_losses)
    
    ## Images Print
    x = iter(test_dl)
    dict = first(x)

    img = dict["img"]
    seg_gt = dict["seg"]
    # img= img.squeeze
    # img = rand_rotate(img)
    # img, seg_gt = first(train_dl)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              model.cuda()
    # img=img.cuda().float()
    (img, seg_gt) = (img.to(device), seg_gt.to(device))
    img = img.float()
    seg_gt = seg_gt.long()

    model.eval()



    seg_pred = model(img)

    slice_num = 50

    img = img.cpu().detach().numpy()
    seg_pred = torch.argmax(seg_pred, dim=(1), keepdims=True)

    seg_pred = seg_pred.cpu().detach().numpy()
    seg_gt = seg_gt.cpu().detach().numpy()

    fig, axs = plt.subplots(nrows=3, sharex=True, figsize=(3, 5))
    axs[0].imshow(img[0, 0, :, :, slice_num], cmap="gray")
    axs[1].imshow(seg_gt[0, 0, :, :, slice_num], cmap="gray")
    axs[2].imshow(seg_pred[0, 0, :, :, slice_num], cmap="gray")
    # plt.imshow(seg_pred[0,1,:,:,30], alpha=0.4)
    plt.savefig(f"img_out/{architecture}_p.png")
    
    
    

    

if __name__ == "__main__":
    main()
# %%
