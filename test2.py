# %%
import hydra
from pathlib import Path
import torch
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from segmentation_models_pytorch.losses import DiceLoss
from tqdm import tqdm, trange
from torchmetrics.classification import MulticlassF1Score

from dptraining.config import Config
from dptraining.config.config_store import load_config_store
from dptraining.datasets.nifti.creator import NiftiSegCreator
import segmentation_models_pytorch as smp

import wandb

from unet import UNet
from trainer import Trainer

load_config_store()
print(Path.cwd())


@hydra.main(version_base=None, config_path=Path.cwd() / "configs")
def main(config: Config):
    #print(config)

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
    train_ds, val_ds, test_ds = NiftiSegCreator.make_datasets(
        config, (None, None, None)
    )
    train_dl, val_dl, test_dl = NiftiSegCreator.make_dataloader(
        train_ds, val_ds, test_ds, {}, {}, {}
    )

    # for idx in range (len(train_dl)):
    #     next(iter(train_dl))
    # train_dl_torch = torch.from_numpy(train_dl)

    # x, y = next(iter(train_dl))
    # x = torch.squeeze(x)
    # x = torch.unsqueeze(x, 0)
    # print(x.shape)

    model = UNet(
        in_channels=1,
        out_channels=3,
        n_blocks=2,
        start_filters=8,
        activation="relu",
        normalization="batch",
        conv_mode="same",
        dim=3,
    )

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
    # device = torch.device("cpu")

    # criterion
    #criterion = torch.nn.CrossEntropyLoss()
    mode = "multiclass"
    criterion = DiceLoss(mode, classes=None, log_loss=False, from_logits=True, smooth=0.0, ignore_index=None, eps=1e-07)

    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    # optimirzer = torch.optim.Adam(model.parameters(), lr=0.01)

    #f1 = F1Score(task="multiclass", num_classes=3)

    trainSteps = len(train_ds)
    f1=[]
    # print("[INFO] training the network...")
    for e in trange(100):
        # set the model in training mode
        model.train()
        # initialize the total training and validation loss
        totalTrainLoss = 0
        totalTestLoss = 0
        # loop over the training set
        for i, (x, y) in tqdm(enumerate(train_dl), total=len(train_dl), leave=False):
        # send the input to the device
            (x, y) = (x.to(device), y.to(device))
            x = x.float()
            y = y.squeeze(1).long()
            # perform a forward pass and calculate the training loss
            pred = model(x)
            # pred = torch.argmax(pred, dim=1, keepdim=True)

            loss = criterion(pred, y)

            #tp, fp, fn, tn = smp.metrics.get_stats(y, pred, mode='multilabel', threshold=0.5)
            #f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
            mcf1s = MulticlassF1Score(num_classes=3).to(device)
            if e ==0:
                f1.append(mcf1s(pred, y).cpu().numpy())
            else:
                f1[i]= (f1[i]+mcf1s(pred, y).cpu().numpy())/2
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
        # print the model training and validation information
        print("[INFO] EPOCH: {}/{}".format(e + 1, 1))
        wandb.log({"f1_score": f1[0]})
        wandb.log({"train_loss": avgTrainLoss})
        print("Train loss: {:.6f}".format(avgTrainLoss))


    # trainer = Trainer(
    #     model=model,
    #     device=device,
    #     criterion=criterion,
    #     optimizer=optimizer,
    #     training_DataLoader=train_dl,
    #     validation_DataLoader=val_dl,
    #     lr_scheduler=None,
    #     epochs=100,
    #     epoch=0,
    #     notebook=False,
    # )

    # #start training
    # training_losses, validation_losses, lr_rates = trainer.run_trainer()
    # #print(training_losses, validation_losses)


if __name__ == "__main__":
    main()
# %%
