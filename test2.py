#%%
import hydra
from pathlib import Path
import torch
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam

from dptraining.config import Config
from dptraining.config.config_store import load_config_store
from dptraining.datasets.nifti.creator import NiftiSegCreator


from unet import UNet
from trainer import Trainer

load_config_store()
print(Path.cwd())

@hydra.main(version_base=None, config_path=Path.cwd()/"configs")
def main(config: Config):
    print(config)

    train_ds, val_ds, test_ds = NiftiSegCreator.make_datasets(config, (None, None, None))
    train_dl, val_dl, test_dl = NiftiSegCreator.make_dataloader(train_ds, val_ds, test_ds, {}, {},{})
    #train_dl_torch = torch.from_numpy(train_dl)
    x, y = next(iter(train_dl))
    print(x.dtype)
   
    
    model= UNet(in_channels=1,
             out_channels=4,
             n_blocks=2,
             start_filters=8,
             activation='relu',
             normalization='batch',
             conv_mode='same',
             dim=3)
    
    # Testing model
    # x = torch.randn(size=(1, 1, 512, 512, 512), dtype=torch.float32)
    # with torch.no_grad():
    #     out = model()
    # print(f'Out: {out.shape}')

    # Option 1: criterion for loss and optimization
    # lossFunc = BCEWithLogitsLoss()
    # opt = Adam(model.parameters(), lr=1e-3)

    
    #device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    #inputs, labels = inputs.to(device), labels.to(device)
    device = torch.device("cpu")
    # criterion
    criterion = torch.nn.CrossEntropyLoss()

    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)



    # trainer
    trainer = Trainer(
        model=model,
        device=device,
        criterion=criterion,
        optimizer=optimizer,
        training_DataLoader=train_dl,
        validation_DataLoader=val_dl,
        lr_scheduler=None,
        epochs=1,
        epoch=0,
        notebook=False,
    )


    # start training
    training_losses, validation_losses, lr_rates = trainer.run_trainer()
    print(training_losses, validation_losses)

if __name__ == '__main__':
    main()
# %%
