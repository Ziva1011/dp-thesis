# %%
import hydra
from pathlib import Path
import glob

# imports from  pytorch
import torch
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
import segmentation_models_pytorch as smp

# from segmentation_models_pytorch.losses import DiceLoss
from torchmetrics.classification import MulticlassF1Score
from torchinfo import summary


from segmentation_models.linknet.linknet import Linknet
from segmentation_models.fpn.fpn import FPN
from segmentation_models.psp.psp import PSPNet
from segmentation_models.pan.pan import PAN
from segmentation_models.deeplabv3.deeplab import DeepLabV3, DeepLabV3Plus


# imports from dptraining
from dptraining.config import Config
from dptraining.config.config_store import load_config_store
from dptraining.datasets.nifti.creator import NiftiSegCreator


# imports from monai
from monai.data import (
    Dataset,
    DataLoader,
    list_data_collate,
    pad_list_data_collate,
    ArrayDataset,
)
from monai.losses import DiceLoss, GeneralizedDiceLoss

# from monai.metrics import get_confusion_matrix, ConfusionMatrixMetric, compute_confusion_matrix_metric
from monai.transforms import (
    RandFlipd,
    Compose,
    LoadImaged,
    RandRotate90d,
    RandAdjustContrast,
    ScaleIntensity,
    EnsureChannelFirstd,
    RandSpatialCrop,
    Resize,
    LoadImage,
    EnsureChannelFirst,
    ScaleIntensityRanged,
)
from monai.utils import first
from monai.networks.nets import(
    DynUNet,
    SegResNetVAE,
    AttentionUnet,
    VNet,
    DiNTS,
    TopologyInstance,
    UNet,
    BasicUNetPlusPlus
)

#imports from opacus
from opacus import PrivacyEngine
from opacus.utils.uniform_sampler import UniformWithReplacementSampler
from opacus.data_loader import wrap_collate_with_empty
from opacus import validators

import wandb
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
from acsconv.converters import ACSConverter

from unet import Unet
from trainer import Trainer

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

    # wandb.init(
    #     # set the wandb project where this run will be logged
    #     project="dpSegmentation",

    #     # track hyperparameters and run metadata
    #     config={
    #     "learning_rate": 0.01,
    #     "architecture": "Unet",
    #     "epochs": 100,
    #     "loss": "Dice",
    #     }
    # )

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
            # #Resize(spatial_size=(128,128,50)),
            # #RandSpatialCrop((128, 128, 50), random_size=False),
            RandFlipd(keys=["img", "seg"], prob=0.1, spatial_axis=1),
            # RandAdjustContrast(prob=0.5, gamma=[0.5,0.6]),
            RandRotate90d(keys=["img", "seg"], prob=0.1, spatial_axes=(0, 1)),
        ]
    )

    val_transforms = Compose(
    [
        LoadImaged(keys=["img", "seg"]),
        EnsureChannelFirstd(keys=["img", "seg"]),

        ScaleIntensityRanged(keys=["img"], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
        #CropForegroundd(keys=["image", "label"], source_key="image"),
    ]
)
    # rand_rotate = RandRotate90(prob=1,spatial_axes=[2,3])

    # train_ds, val_ds, test_ds = NiftiSegCreator.make_datasets(
    #     config, (None, None, None)
    # )
    # train_dl, val_dl, test_dl = NiftiSegCreator.make_dataloader(
    #     train_ds, val_ds, test_ds, {}, {}, {}
    # )

    # images = sorted(glob.glob("/media/datasets/MSD/Task03_Liver/imagesTr/liver_*.nii.gz"))
    # segs = sorted(glob.glob("/media/datasets/MSD/Task03_Liver/labelsTr/liver_*.nii.gz"))
    
    train_files = glob.glob("./data2/liver_seg/train/*.nii")
    train_labels = glob.glob("./data2/liver_seg_labels/train/*.nii")
    val_files = glob.glob("./data2/liver_seg/val/*.nii")
    val_labels = glob.glob("./data2/liver_seg_labels/val/*.nii")

    # train_ds = ArrayDataset(train_files, transforms, train_labels, transforms)
    # train_dl = DataLoader(train_ds, batch_size=1, num_workers=2, pin_memory='True')

    # val_ds = ArrayDataset(val_files, transforms, val_labels, transforms)
    # val_dl = DataLoader(val_ds, batch_size=1, num_workers=2, pin_memory='True')

    train_files = [
        {"img": img, "seg": seg} for img, seg in zip(train_files, train_labels)
    ]
    val_files = [{"img": img, "seg": seg} for img, seg in zip(val_files, val_labels)]

    train_ds = Dataset(data=train_files, transform=train_transforms)
    train_dl = DataLoader(
        train_ds,
        batch_size=4,
        shuffle=True,
        num_workers=2,
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
    )

    val_ds = Dataset(data=val_files, transform=val_transforms)
    val_dl = DataLoader(
        val_ds, batch_size=1, num_workers=4, collate_fn=list_data_collate
    )

    # name_of_script = sys.argv[0]
    # architecture = sys.argv[1]
    # private = sys.argv[2]
    private = True
    #architecture = config.model.name
    #print(architecture)
    architecture = 'dynUnet'
    print("ARCHITECTURE: {} {} \n".format(architecture, private) )


    if (architecture =='dynUnet'):
        learning_rate = 0.0002
        model= DynUNet(
            spatial_dims=3, 
            in_channels=1, 
            out_channels=3, 
            kernel_size=[3, 3, 3, 3, 3, 3],
            strides=[1, 2, 2, 2, 2, [2, 2, 1]],
            upsample_kernel_size=[2, 2, 2, 2, [2, 2, 1]],
            norm_name="instance",
            deep_supervision=False,
            res_block=True)
    
    elif (architecture=='unet'):
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

    elif (architecture=='attention'):
        learning_rate = 0.0001
        model = AttentionUnet(
            spatial_dims=3, in_channels=1, out_channels=3, channels=[64,32,16,8,4], strides= [1, 2, 2, 2, 2], dropout= 0.2
        )
        model = validators.ModuleValidator.fix(model)

    elif (architecture=='vnet'):
        learning_rate = 0.002
        model= VNet(
            spatial_dims=3, 
            in_channels=1, 
            out_channels=3)
        model = validators.ModuleValidator.fix(model)
    
    elif (architecture=='unetMonai'):
        learning_rate = 0.002
        model= UNet(
            spatial_dims=3, in_channels=1, out_channels=3, channels=[64,32,16,8,4], strides= [1, 2, 2, 2, 2] 
        )
    

    elif (architecture=="dints"):
        learning_rate = 0.002
        model = DiNTS(dints_space=TopologyInstance(), in_channels=1, num_classes=3)

    elif (architecture=="resnet"):
        learning_rate = 0.002
        class NewModel(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            def forward(self, x):
                return self.model(x)[0]
        model = SegResNetVAE(input_image_size=(128,128,64), in_channels=1, out_channels=3)
    
    elif (architecture=="unet++"):
        learning_rate = 0.0002
        model_2d = smp.UnetPlusPlus(in_channels=1, classes=3)
        model = ACSConverter(model_2d)
    
    elif (architecture=="unetpp"):
        learning_rate = 0.0002
        model = BasicUNetPlusPlus(spatial_dims=3, in_channels=1, out_channels=3)


    elif (architecture=="linknet"):
        learning_rate = 0.002
        model = Linknet(in_channels=1, classes=3)

    elif (architecture=="fpn"):
        learning_rate = 0.002
        model = FPN(in_channels=1, classes=3, encoder_weights=None)

    elif (architecture=="psp"):
        learning_rate = 0.002
        model = PSPNet(in_channels=1, classes=3, encoder_weights=None)

    elif (architecture=="pan"):
        learning_rate = 0.002
        model = PAN(in_channels=1, classes=3)

    elif (architecture=="deep"):
        learning_rate = 0.002
        model = DeepLabV3(in_channels=1, classes=3)

    elif (architecture=="deepPlus"):
        learning_rate = 0.002
        model = DeepLabV3Plus(in_channels=1, classes=3)

    
    batch_size = 4
    summary(model, input_size=(batch_size, 1, 128, 128, 64))

    # Testing model
    # x = torch.randn(size=(1, 1, 512, 512, 512), dtype=torch.float32)
    # with torch.no_grad():
    #     out = model(x)
    # print(f'Out: {out.shape}')

    # Option 1: criterion for loss and optimization
    # lossFunc = BCEWithLogitsLoss()
    # opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # print(model.parameters)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    model.train()

    # criterion
    # criterion = torch.nn.CrossEntropyLoss()
    # ßßmode = "multiclass"
    # criterion = DiceLoss(mode, classes=None, log_loss=False, from_logits=True, smooth=0.0, ignore_index=None, eps=1e-07)
    criterion = DiceLoss(to_onehot_y=True, reduction="mean", softmax=True, weight=[1,10,50])
    #criterion = GeneralizedDiceLoss(include_background=True, to_onehot_y=True, reduction="mean", sigmoid=True)

    # optimizer
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    optimizer = torch.optim.NAdam(model.parameters(), lr=learning_rate)

    # f1 = F1Score(task="multiclass", num_classes=3)

    trainSteps = len(train_ds)
    epochs = 100

    if (private):
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
            sample_empty_shapes={x: train_dl.dataset[0][x].shape for x in ["img", "seg"]},
            dtypes={x: train_dl.dataset[0][x].dtype for x in ["img", "seg"]},
        )

    ##First Training loop
    print("[INFO] training the network...")
    # dict= first(train_dl)

    # for e in trange(epochs):
    #     # set the model in training mode
    #     model.train()
    #     # initialize the total training and validation loss
    #     totalTrainLoss = 0
    #     totalTestLoss = 0
    #     f1=[0,0,0]

    # loop over the training set
    #     for i, dict in tqdm(enumerate(train_dl), total=len(train_dl), leave=False):
    #     # send the input to the device
    #         #x, y = first(train_dl)
    #         x= dict["img"]
    #         y= dict["seg"]

    #         (x, y) = (x.to(device), y.to(device))
    #         x = x.float()
    #         #y = y.squeeze(0).long()

    #         #has to be a long for the one hot encoding
    #         y = y.long()

    #         # perform a forward pass and calculate the training loss
    #         pred = model(x)
    #         # pred = torch.argmax(pred, dim=1, keepdim=True)

    #         #one hot encode the label
    #         z= torch.nn.functional.one_hot(y[0,0], num_classes=3)
    #         z=torch.permute(z,(3,0,1,2))
    #         z = z.view(1,3,128,128,100)

    #         #calculate loss
    #         loss = criterion(pred, y)

    #         #metric = get_confusion_matrix(pred, z)
    #         tp, fp, fn, tn = smp.metrics.get_stats(pred.long(), z, mode='multiclass', num_classes=3)
    #         f1 = 2*tp/(2*tp+fp+fn)
    #         # f1 = f1+smp.metrics.f1_score(tp, fp, fn, tn, reduction="none").cpu().numpy()

    #         #F1 score for multiclass
    #         # mcf1s = MulticlassF1Score(num_classes=3, average=None).to(device)
    #         # f1= (f1+mcf1s(pred, z).cpu().numpy())
    #         # metric = ConfusionMatrixMetric(include_background=True, metric_name="f1 score", compute_sample=False, reduction= "mean_channel", get_not_nans=False)
    #         # #.numpy() to convert it to an array
    #         # f1 = f1 + metric(pred,z).cpu().numpy()

    #         #f1.append(mcf1s(pred, y))
    #         #wandb.log({"f1_score": f1.data.item()})
    #         #wandb.log({"f1_score": f1_score})

    #         # first, zero out any previously accumulated gradients, then
    #         # perform backpropagation, and then update model parameters
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         # add the loss to the total training loss so far
    #         totalTrainLoss += loss
    #         # switch off autograd

    #     avgTrainLoss = totalTrainLoss / trainSteps
    #     # update our training history
    #     f1 = f1/len(train_dl)
    #     print(f1)
    #     # print the model training and validation information
    #     print("[INFO] EPOCH: {}/{}".format(e + 1, epochs))

    # #     # wandb.log({"f1_background": f1[0]})
    # #     # wandb.log({"f1_liver": f1[1]})
    # #     # wandb.log({"f1_tumor": f1[2]})
    # #     # wandb.log({"train_loss": avgTrainLoss.item()})

    #     print("Train loss: {:.6f}".format(avgTrainLoss))

    # Second training loop
    # trainer = Trainer(
    #     model=model,
    #     device=device,
    #     criterion=criterion,
    #     optimizer=optimizer,
    #     training_DataLoader=train_dl,
    #     validation_DataLoader=val_dl,
    #     lr_scheduler=None,
    #     epochs=epochs,
    #     epoch=0,
    #     notebook=False,
    # )

    # # # start training
    # training_losses, validation_losses, lr_rates = trainer.run_trainer()
    # print(training_losses, validation_losses)

    ## Images Print

    x= iter(val_dl)
    dict = first(x)
    

    img =   dict["img"]
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
    plt.savefig("output.png")
    
    

if __name__ == "__main__":
    main()
# %%
