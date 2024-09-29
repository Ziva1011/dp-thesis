import numpy as np
import torch
import wandb
from torchmetrics.classification import MulticlassF1Score


class Tester:
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        validation_DataLoader: torch.utils.data.Dataset = None,
        lr_scheduler: torch.optim.lr_scheduler = None,

        notebook: bool = False,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.validation_DataLoader = validation_DataLoader
        self.device = device
        self.notebook = notebook

        self.test_loss = []
        self.learning_rate = []

    def run_trainer(self):
        if self.notebook:
            from tqdm.notebook import tqdm, trange
        else:
            from tqdm import tqdm, trange


        self._test()

        """Learning rate scheduler block"""
        if self.lr_scheduler is not None:
            if (
                self.validation_DataLoader is not None
                and self.lr_scheduler.__class__.__name__ == "ReduceLROnPlateau"
            ):
                self.lr_scheduler.batch(
                    self.test_loss[i]
                )  # learning rate scheduler step with validation loss
            else:
                self.lr_scheduler.batch()  # learning rate scheduler step

        return self.test_loss

    # %%
   
    def _test(self):
        if self.notebook:
            from tqdm.notebook import tqdm, trange
        else:
            from tqdm import tqdm, trange

        self.model.eval()  # evaluation mode
        test_losses = []  # accumulate the losses here
        batch_iter = tqdm(
            enumerate(self.validation_DataLoader),
            "Test",
            total=len(self.validation_DataLoader),
            leave=False,
        )

        f1 = [0, 0, 0]

        for i, dict in batch_iter:
            x = dict["img"]
            y = dict["seg"]
            input, target = x.to(self.device), y.to(
                self.device
            )  # send to device (GPU or CPU)
            with torch.inference_mode():
                input = input.float()
                out = self.model(input)
                if type(out) is tuple or type(out) is list:
                    out = torch.tensor(out[0])
                loss = self.criterion(out, target)
                loss_value = loss.item()
                test_losses.append(loss_value)

                target = target.squeeze(0).long()

                mcf1s = MulticlassF1Score(num_classes=3, average=None).to(self.device)
                f1 = f1 + mcf1s(out, target).cpu().numpy()

                batch_iter.set_description(f"Test: (loss {loss_value:.4f})")

        self.test_loss.append(np.mean(test_losses))
        f1 = f1 / len(batch_iter)
        print("Test_f1:", f1)
        batch_iter.close()
