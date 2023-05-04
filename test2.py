#%%
import hydra
from pathlib import Path
import torch

from dptraining.config import Config
from dptraining.config.config_store import load_config_store
from dptraining.datasets.nifti.creator import NiftiSegCreator

from unet import UNet

load_config_store()
print(Path.cwd())

@hydra.main(version_base=None, config_path=Path.cwd()/"configs")
def main(config: Config):
    print(config)

    train_ds, val_ds, test_ds = NiftiSegCreator.make_datasets(config, (None, None, None))
    train_dl, val_dl, test_dl = NiftiSegCreator.make_dataloader(train_ds, val_ds, test_ds, {}, {},{})
    #train_dl_torch = torch.from_numpy(train_dl)
    print(next(iter(train_dl)))
   
    
    model= UNet(in_channels=1,
             out_channels=4,
             n_blocks=2,
             start_filters=8,
             activation='relu',
             normalization='batch',
             conv_mode='same',
             dim=3)
    x = torch.randn(size=(1, 1, 512, 512, 512), dtype=torch.float32)
    print(x[0][0][0])
    with torch.no_grad():
        out = model(x)
    print(f'Out: {out.shape}')

if __name__ == '__main__':
    main()
# %%
