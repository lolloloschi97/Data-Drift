import sys

import torch
from timeit import default_timer as timer
from config import *
from torch import nn
from Dropout import enable_MCD
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Callable, Tuple


class MSLELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, actual):
        return self.mse(torch.log(pred + 1), torch.log(actual + 1))


class MaeDataset(Dataset):

    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)[:, None]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def training_loop(
        num_epochs: int,
        optimizer: torch.optim,
        lr_scheduler: torch.optim.lr_scheduler,
        log_interval: int,
        model: nn.Module,
        loader_train: DataLoader,
        loader_val: DataLoader,
        patience: int = 10,
        verbose: bool = True) -> Dict:
    """Executes the training loop.

        Args:
            num_epochs: the number of epochs.
            optimizer: the optimizer to use.
            lr_scheduler: the scheduler for the learning rate.
            log_interval: intervall to print on tensorboard.
            model: the mode to train.
            loader_train: the data loader containing the training data.
            loader_val: the data loader containing the validation data.
            patience: integer for definin patience of trianing
            verbose: if true print the value of loss.

        Returns:
            A dictionary with the statistics computed during the train:
            the values for the train loss for each epoch.
            the values for the train accuracy for each epoch.
            the values for the validation accuracy for each epoch.
            the time of execution in seconds for the entire loop.
    """
    criterion = MSLELoss()
    loop_start = timer()

    losses_values_train = []
    losses_values_val = []

    train_mae_values = []
    val_mae_values = []

    last_loss = sys.maxsize
    trigger_times = 0

    for epoch in range(1, num_epochs + 1):
        time_start = timer()
        loss_train, mae_train = training(model, loader_train, 'cpu',
                                      optimizer, criterion)
        loss_val, mae_val = validate(model, loader_val, 'cpu', criterion)
        time_end = timer()

        losses_values_train.append(loss_train)
        train_mae_values.append(mae_train)

        losses_values_val.append(loss_train)
        val_mae_values.append(mae_val)

        lr = optimizer.param_groups[0]['lr']

        if verbose:
            print(f'Epoch: {epoch} '
                  f' Lr: {lr:.8f} '
                  f' Loss: Train = [{loss_train:.4f}] - Val = [{loss_val:.4f}] '
                  f' MAE: Train = [{mae_train:.2f}] - Val = [{mae_val:.2f}] '
                  f' Time one epoch (s): {(time_end - time_start):.4f} ')

        # Increases the internal counter
        if lr_scheduler:
            lr_scheduler.step()

        # Early stopping
        if loss_val > last_loss:
            trigger_times += 1

            if trigger_times >= patience:
                loop_end = timer()
                time_loop = loop_end - loop_start
                if verbose:
                    print(f'Time for {epoch} epochs (s): {(time_loop):.3f}')

                return {'loss_values_train': losses_values_train,
                        'loss_values_val': losses_values_val,
                        'train_mae_values': train_mae_values,
                        'val_mae_values': val_mae_values,
                        'time': time_loop}

    loop_end = timer()
    time_loop = loop_end - loop_start
    if verbose:
        print(f'Time for {num_epochs} epochs (s): {(time_loop):.3f}')

    return {'loss_values_train': losses_values_train,
            'loss_values_val': losses_values_val,
            'train_mae_values': train_mae_values,
            'val_mae_values': val_mae_values,
            'time': time_loop}


# Train one epoch
def training(model: nn.Module,
          train_loader: DataLoader,
          device: torch.device,
          optimizer: torch.optim,
          criterion: Callable[[torch.Tensor, torch.Tensor], float]) -> Tuple[float, float]:
    """Trains a neural network for one epoch.

    Args:
        model: the model to train.
        train_loader: the data loader containing the training data.
        device: the device to use to train the model.
        optimizer: the optimizer to use to train the model.
        criterion: the loss to optimize.
        log_interval: the log interval.
        epoch: the number of the current epoch.

    Returns:
        the cross entropy Loss value on the training data.
        the accuracy on the training data.
    """
    error_sum = 0
    samples_train = 0
    loss_train = 0

    model.train()
    for idx_batch, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        scores = model(x)

        loss = criterion(scores, y)
        loss_train += loss.item() * len(y)
        samples_train += len(y)

        loss.backward()
        optimizer.step()
        error_sum += torch.abs(scores - y).sum()

    loss_train /= samples_train
    mae = error_sum / samples_train
    return loss_train, mae


# Validate one epoch
def validate(model: nn.Module,
             data_loader: DataLoader,
             device: torch.device,
             criterion: Callable[[torch.Tensor, torch.Tensor], float]) -> Tuple[float, float]:
    """Evaluates the model.

    Args:
        model: the model to evalaute.
        data_loader: the data loader containing the validation or test data.
        device: the device to use to evaluate the model.
        criterion: the loss function.

    Returns:
        the loss value on the validation data.
        the accuracy on the validation data.
    """

    error_sum = 0
    samples_val = 0
    loss_val = 0
    model = model.eval()

    with torch.no_grad():
        for idx_batch, (x, y) in enumerate(data_loader):
            x, y = x.to(device), y.to(device)
            scores = model(x)

            ###
            loss = criterion(scores, y)
            loss_val += loss.item() * len(y)
            samples_val += len(y)
            error_sum += torch.abs(scores - y).sum()

    loss_val /= samples_val
    mae = error_sum / samples_val
    return loss_val, mae

def compute_prediction(model, dataset, is_vae = True, has_real = True, mcd = False):
    res = []
    model.eval()

    if mcd: # Enable MC Dropout
        model = enable_MCD(model)

    with torch.no_grad():
        if has_real: #check if it has y to predict
            for idx_batch, (el, y) in enumerate(DataLoader(dataset,1, shuffle = False)):
                res.append(model(el).numpy())
            return res
        else:
            distribution = []
            for idx_batch, el in enumerate(DataLoader(dataset,1, shuffle = False)):
                pred, distr = model(el)
                res.append(pred)
                distribution.append(distr)
            return res, distribution
# Model

class MLP(torch.nn.Module):
    def __init__(self, n_features, hidden_width=1024, n_additional_hidden_layers=4, dense_rate=2, p=0.3, embedding = False):
        super(MLP, self).__init__()

        final_dim = int(hidden_width / (dense_rate ** n_additional_hidden_layers))
        self.first = torch.nn.Linear(n_features, hidden_width)
        self.activation = torch.relu
        self.last = torch.nn.Linear(final_dim, 1)
        self.dropout = torch.nn.Dropout(p=p)

        self.additional_hidden_layers = torch.nn.ModuleList(
            [torch.nn.Linear(int(hidden_width / (2 ** (i - 1))), int(hidden_width / (2 ** i))) for i in
             range(1, n_additional_hidden_layers + 1)])
        self.embedding = embedding

    def forward(self, x):
        x = self.first.forward(x)  # call linear classifier --> computes scores (z)
        x = self.activation(x)  # apply actovation functions on z --> sigma(z) = h
        for layer in self.additional_hidden_layers:  # iteration over hidden layers(initialized as LC)
            x = layer.forward(x)
            x = self.dropout(x)
            x = self.activation(x)
        if self.embedding:
            return x
        else:
            x = self.last.forward(x)
            return x

