# %%
import hydra
from pathlib import Path
import glob

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
    RandSpatialCrop,
    Resize,
)
from monai.utils import first

import wandb

from unet import UNet
from trainer import Trainer

load_config_store()
print(Path.cwd())


@hydra.main(version_base=None, config_path=Path.cwd() / "configs")
def main(config: Config):
    #print(config)

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

    #Transforms
    transforms = Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        #Resize(spatial_size=(128,128,50)),
        #RandSpatialCrop((128, 128, 50), random_size=False),
        #RandFlip(prob=0.5,spatial_axis=1),
        #RandAdjustContrast(prob=0.5, gamma=[0.5,0.9]),
        #ToTensor,
        #Lambda(lambda x: x.squeeze()),
        #RandRotate90(prob=1,spatial_axes=[2,3])
    ])
    # rand_rotate = RandRotate90(prob=1,spatial_axes=[2,3])

    # train_ds, val_ds, test_ds = NiftiSegCreator.make_datasets(
    #     config, (None, None, None)
    # )
    # train_dl, val_dl, test_dl = NiftiSegCreator.make_dataloader(
    #     train_ds, val_ds, test_ds, {}, {}, {}
    # )





    images = sorted(glob.glob("/media/datasets/MSD/Task03_Liver/imagesTr/liver_*.nii.gz"))
    segs = sorted(glob.glob("/media/datasets/MSD/Task03_Liver/labelsTr/liver_*.nii.gz"))

    train_files = glob.glob("./data/liver_seg/train/*.nii")
    train_labels = glob.glob("./data/liver_seg_labels/train/*.nii")

    train_ds = ArrayDataset(train_files, transforms, train_labels, transforms)
    train_dl = DataLoader(train_ds, batch_size=1, num_workers=2, pin_memory='True')

   
    #train_files = [{"img": img, "seg": seg} for img, seg in zip(images[:20], segs[:20])]
    #val_files = [{"img": img, "seg": seg} for img, seg in zip(images[-20:], segs[-20:])]

    # train_ds = Dataset(data=train_files, transform=transforms)
    # train_dl = DataLoader(
    #     train_ds,
    #     batch_size=1,
    #     shuffle=True,
    #     num_workers=4,
    #     collate_fn=list_data_collate,
    #     pin_memory=torch.cuda.is_available(),
    # )

    model = UNet(
        in_channels=1,
        out_channels=3,
        n_blocks=2,
        start_filters=8,
        activation="mish",
        normalization="batch",
        conv_mode="same",
        dim=3,
    )

    # batch_size = 16
    # summary(model, input_size=(batch_size, 1, 128, 128, 50))

    # Testing model
    # x = torch.randn(size=(1, 1, 512, 512, 512), dtype=torch.float32)
    # with torch.no_grad():
    #     out = model(x)
    # print(f'Out: {out.shape}')

    # Option 1: criterion for loss and optimization
    # lossFunc = BCEWithLogitsLoss()
    # opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    #print(model.parameters)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    # inputs, labels = inputs.to(device), labels.to(device)

    # criterion
    #criterion = torch.nn.CrossEntropyLoss()
    mode = "multiclass"
    criterion = DiceLoss(mode, classes=None, log_loss=False, from_logits=True, smooth=0.0, ignore_index=None, eps=1e-07)

    # optimizer
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    optimizer = torch.optim.NAdam(model.parameters(), lr=0.002)

    #f1 = F1Score(task="multiclass", num_classes=3)

    trainSteps = len(train_ds)
    epochs= 100

    ##First Training loop
    print("[INFO] training the network...")

    #img, seg_gt = first(train_dl)
    #print(img.shape)
    # print(im.shape, seg.shape)
    for e in trange(epochs):
        # set the model in training mode
        model.train()
        # initialize the total training and validation loss
        totalTrainLoss = 0
        totalTestLoss = 0
        f1=[0,0,0]
        # loop over the training set
        for i, (x,y) in tqdm(enumerate(train_dl), total=len(train_dl), leave=False):
        # send the input to the device
            #x, y = first(train_dl)
            #y= dict["seg"]
            (x, y) = (x.to(device), y.to(device))
            x = x.float()
            y = y.squeeze(1).long()
            #y = y.long()
            # perform a forward pass and calculate the training loss
            pred = model(x)
            # pred = torch.argmax(pred, dim=1, keepdim=True)
            z= torch.nn.functional.one_hot(y[0], num_classes=3)
            z = z.view(1,3,128,128,50)
            loss = criterion(pred, y)

            #tp, fp, fn, tn = smp.metrics.get_stats(z, pred.round().long(), mode='multilabel', threshold=0.5, num_classes=3)
            #f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="macro-imagewise")
            
            mcf1s = MulticlassF1Score(num_classes=3, average=None).to(device)
            f1= (f1+mcf1s(pred, y).cpu().numpy())

            #f1.append(mcf1s(pred, y))
            #wandb.log({"f1_score": f1.data.item()})
            #wandb.log({"f1_score": f1_score})

            # first, zero out any previously accumulated gradients, then
            # perform backpropagation, and then update model parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # add the loss to the total training loss so far
            totalTrainLoss += loss
    #         # switch off autograd

        avgTrainLoss = totalTrainLoss / trainSteps
        # update our training history
        f1 = f1/len(train_dl)
        print(f1)
        # print the model training and validation information
        print("[INFO] EPOCH: {}/{}".format(e + 1, epochs))

        # wandb.log({"f1_background": f1[0]})
        # wandb.log({"f1_liver": f1[1]})
        # wandb.log({"f1_tumor": f1[2]})
        # wandb.log({"train_loss": avgTrainLoss})

        print("Train loss: {:.6f}".format(avgTrainLoss))


    ##Second training loop
    # trainer = Trainer(
    #     model=model,
    #     device=device,
    #     criterion=criterion,
    #     optimizer=optimizer,
    #     training_DataLoader=train_dl,
    #     validation_DataLoader=None,
    #     lr_scheduler=None,
    #     epochs=100,
    #     epoch=0,
    #     notebook=False,
    # )

    # #start training
    # training_losses, validation_losses, lr_rates = trainer.run_trainer()
    # print(training_losses, validation_losses)

    ## Images Print
    img, seg_gt = first(train_dl)
    #img= img.squeeze
    #img = rand_rotate(img)
    # img, seg_gt = first(train_dl)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              model.cuda()
    #img=img.cuda().float()
    (img, seg_gt) = (img.to(device), seg_gt.to(device))
    img = img.float()
    seg_gt = seg_gt.squeeze(1).long()
    model.eval()
    seg_pred = model(img)

    slice_num = 35
    img=img.cpu().detach().numpy()
    seg_pred = seg_pred.cpu().detach().numpy()
    seg_gt = seg_gt.cpu().detach().numpy()
    fig, axs = plt.subplots(nrows=3, sharex=True, figsize=(3, 5))
    axs[0].imshow(img[0,0,:,:,slice_num], cmap="gray")
    axs[1].imshow(seg_gt[0,:,:,slice_num], cmap="gray")
    axs[2].imshow(seg_pred[0,2,:,:,slice_num],cmap="gray")
    # plt.imshow(seg_pred[0,1,:,:,30], alpha=0.4)
    plt.savefig("output.png")



if __name__ == "__main__":
    main()
# %%
