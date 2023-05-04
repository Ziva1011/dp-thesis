# %%
import hydra
from pathlib import Path
import torch

from dptraining.config import Config
from dptraining.config.config_store import load_config_store
from dptraining.datasets.nifti.creator import NiftiSegCreator

from dptraining.datasets import make_loader_from_config

from pytorch3dunet.unet3d.trainer import UNetTrainer
from pytorch3dunet.unet3d.model import get_model
from pytorch3dunet.unet3d.utils import DefaultTensorboardFormatter
from pytorch3dunet.unet3d.losses import get_loss_criterion
from pytorch3dunet.unet3d.metrics import get_evaluation_metric
from pytorch3dunet.unet3d.utils import get_class

from unet import UNet

load_config_store()
print(Path.cwd())


@hydra.main(version_base=None, config_path=Path.cwd() / "configs")
def main(config: Config):
    print(config)
    # %%
    train_ds, val_ds, test_ds = NiftiSegCreator.make_datasets(
        config, (None, None, None)
    )
    train_dl, val_dl, test_dl = NiftiSegCreator.make_dataloader(
        train_ds, val_ds, test_ds, {}, {}, {}
    )
    # train_dl, val_dl, test_dl = make_loader_from_config(config)
    # train_dl_torch = torch.from_numpy(train_dl)
    print(type(train_dl))
    x, y = next(iter(train_dl))

    binary_loss = "DiceLoss"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_config = {}
    # train_config.update({
    #     # get device to train on
    #     'device': device,
    #     'loss': {'name': binary_loss, 'weight': np.random.rand(2).astype(np.float32), 'pos_weight': 3.},
    #     'eval_metric': {'name': val_metric}
    # })

    loaders = {"train": train_dl, "val": val_dl}

    loss_criterion = get_loss_criterion(train_config)
    eval_criterion = get_evaluation_metric(train_config)
    model = get_class("UNet3D", modules=["pytorch3dunet.unet3d.model"])
    model = model.to(device)
    y_p = model(x.to(device))

    formatter = DefaultTensorboardFormatter()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.003, betas=(0.9, 0.999), weight_decay=0
    )
    unet = UNetTrainer(
        model=model,
        optimizer=optimizer,
        lr_scheduler=None,
        loss_criterion=loss_criterion,
        eval_criterion=eval_criterion,
        loaders=loaders,
        tensorboard_formatter=formatter,
        resume=None,
        pre_trained=None,
    )
    unet.train()


if __name__ == "__main__":
    main()
# %%
