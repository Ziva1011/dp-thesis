import hydra
from pathlib import Path
import torch

from dptraining.config import Config
from dptraining.config.config_store import load_config_store
from dptraining.datasets.nifti.creator import NiftiSegCreator


load_config_store()
print(Path.cwd())

@hydra.main(version_base=None, config_path=Path.cwd()/"configs")
def main(config: Config):
    print(config)

    train_ds, val_ds, test_ds = NiftiSegCreator.make_datasets(config, (None, None, None))
    train_dl, val_dl, test_dl = NiftiSegCreator.make_dataloader(train_ds, val_ds, test_ds, {}, {},{})
    #train_dl_torch = torch.from_numpy(train_dl)
    print(type(train_dl))

if __name__ == '__main__':
    main()