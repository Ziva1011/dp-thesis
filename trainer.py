import numpy as np
import torch
import wandb
from torchmetrics.classification import MulticlassF1Score

class Trainer:
    def __init__(self,
                 model: torch.nn.Module,
                 device: torch.device,
                 criterion: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 training_DataLoader: torch.utils.data.Dataset,
                 validation_DataLoader: torch.utils.data.Dataset = None,
                 lr_scheduler: torch.optim.lr_scheduler = None,
                 epochs: int = 100,
                 epoch: int = 0,
                 notebook: bool = False
                 ):

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.training_DataLoader = training_DataLoader
        self.validation_DataLoader = validation_DataLoader
        self.device = device
        self.epochs = epochs
        self.epoch = epoch
        self.notebook = notebook

        self.training_loss = []
        self.validation_loss = []
        self.learning_rate = []

    def run_trainer(self):

        if self.notebook:
            from tqdm.notebook import tqdm, trange
        else:
            from tqdm import tqdm, trange

        progressbar = trange(self.epochs, desc='Progress')
        for i in progressbar:
            """Epoch counter"""
            self.epoch += 1  # epoch counter

            """Training block"""
            self._train()

            """Validation block"""
            if self.validation_DataLoader is not None:
                self._validate()

            """Learning rate scheduler block"""
            if self.lr_scheduler is not None:
                if self.validation_DataLoader is not None and self.lr_scheduler.__class__.__name__ == 'ReduceLROnPlateau':
                    self.lr_scheduler.batch(self.validation_loss[i])  # learning rate scheduler step with validation loss
                else:
                    self.lr_scheduler.batch()  # learning rate scheduler step
            #wandb.log({"lr": self.learning_rate[i], "train_loss": self.training_loss[i]})
            #wandb.log({"val_loss": self.validation_loss[i]})
        return self.training_loss, self.validation_loss, self.learning_rate
#%%
    def _train(self):

        if self.notebook:
            from tqdm.notebook import tqdm, trange
        else:
            from tqdm import tqdm, trange

        self.model.train()  # train mode
        train_losses = []  # accumulate the losses here
        batch_iter = tqdm(enumerate(self.training_DataLoader), 'Training', total=len(self.training_DataLoader),
                          leave=False)
        f1=[0,0,0]

        for i, dict in batch_iter:
            x= dict["img"]  
            y= dict["seg"]
            input, target = x.to(self.device), y.to(self.device)  # send to device (GPU or CPU)
            
            out = self.model(input)  # one forward pass

            loss = self.criterion(out, target)  # calculate loss
            self.optimizer.zero_grad()  # zerograd the parameters
            
            loss_value = loss.item()
            train_losses.append(loss_value)
            loss.backward()  # one backward pass
            self.optimizer.step()  # update the parameters
            
            target = target.squeeze(1).long() #because multiclass receives size (N, ...)
            mcf1s = MulticlassF1Score(num_classes=3, average=None).to(self.device)
            f1= (f1+mcf1s(out, target).cpu().numpy())

            batch_iter.set_description(f'Training: (loss {loss_value:.4f})')  # update progressbar

        self.training_loss.append(np.mean(train_losses))
        wandb.log({"train_loss": np.mean(train_losses)})
        self.learning_rate.append(self.optimizer.param_groups[0]['lr'])
        f1 = f1/len(batch_iter)
        print(f1)
        wandb.log({"f1_background": f1[0]})
        wandb.log({"f1_liver": f1[1]})
        wandb.log({"f1_tumor": f1[2]})

        batch_iter.close()

    def _validate(self):

        if self.notebook:
            from tqdm.notebook import tqdm, trange
        else:
            from tqdm import tqdm, trange

        self.model.eval()  # evaluation mode
        valid_losses = []  # accumulate the losses here
        batch_iter = tqdm(enumerate(self.validation_DataLoader), 'Validation', total=len(self.validation_DataLoader),
                          leave=False)
        
        f1=[0,0,0]

        for i,  dict in batch_iter:
            x= dict["img"]  
            y= dict["seg"]
            input, target = x.to(self.device), y.to(self.device)  # send to device (GPU or CPU)
            with torch.no_grad():
                input = input.float()
                out = self.model(input)
                loss = self.criterion(out, target)
                loss_value = loss.item()
                valid_losses.append(loss_value)

                target = target.squeeze(0).long()

                mcf1s = MulticlassF1Score(num_classes=3, average=None).to(self.device)
                f1= (f1+mcf1s(out, target).cpu().numpy())

                batch_iter.set_description(f'Validation: (loss {loss_value:.4f})')

        self.validation_loss.append(np.mean(valid_losses))
        f1 = f1/len(batch_iter)
        print('Validation_f1:',f1)
        wandb.log({"val_loss": np.mean(valid_losses)})
        batch_iter.close()

