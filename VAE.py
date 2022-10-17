import copy
import sys

import torch
import numpy as np
from timeit import default_timer as timer
from torch import nn
from torch.utils.data import DataLoader, Dataset
from typing import Dict, Callable, Tuple, Union
import torch.nn.functional as F

Nonetype = type(None)

### Betascheduler

class Betascheduler(object):
    def __init__(self,n_epochs,batch_iteration, start = 0.0, stop = 1.0,  n_cycle=4, ratio=0.5 ):
        self.batch_iteration = batch_iteration
        self.n_epochs = n_epochs
        self.n_iter = n_epochs * batch_iteration
        self.current_epoch = int(0)
        self.current_iteration = int(0)
        self.start = start
        self.stop = stop
        self.n_cycle = n_cycle
        self.ratio = ratio
        self.schedule = self.frange_cycle_linear()
        self.beta = start


    def frange_cycle_linear(self):
        Schedule = np.ones(self.n_iter + 1) * self.stop
        period = self.n_iter/self.n_cycle
        step = (self.stop-self.start)/(period*self.ratio) # linear schedule

        for c in range(self.n_cycle):
            v, i = self.start, 0
            while v <= self.stop and (int(i+c*period) < self.n_iter):
                Schedule[int(i+c*period)] = v
                v += step
                i += 1
        return Schedule

    def get_current_beta(self):
        beta =  self.schedule[self.current_iteration]
        return beta

    def get_current_epoch(self):
        epoch = self.current_iteration // self.batch_iteration
        return epoch


    def step(self):
         self.current_iteration += 1
         self.beta = self.get_current_beta()
         if self.get_current_epoch() > self.current_epoch:
             self.current_epoch +=1

    def reset(self):
        self.beta = self.start
        self.current_epoch = int(0)
        self.current_iteration = int(0)

    def summary(self):
        print(f'''--------------------------------------------------------
Betascheduler:
current beta = {self.beta}
current iteration = {self.current_iteration}
current epoch = {self.current_epoch}
--------------------------------------------------------''')


# loss functions

def final_loss_1(reconstructed, q_z, inputs, beta, criterion):
    """
    This function will add the reconstruction loss (BCELoss) and the
    KL-Divergence.
    KL-Divergence = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    :param reconstructed: decoder output of model1
    :param q_z: Learned normal distribution, .loc (mu) .scale(sigma)
    :param beta: weight of kl divergence wrt rec loss
    """
    # reconstruction loss
    REC = criterion(reconstructed, inputs.detach())

    # KL loss
    mu = q_z.loc
    logvar = (q_z.scale ** 2).log()
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) # 0.5 * sum( var + mu**2 -1 - ln(var))

    # final loss
    LOSS = (1- beta)* REC + beta * KLD  # https://github.com/haofuml/cyclical_annealing  #LOSS =  REC + KLD

    return LOSS, KLD, REC

def final_loss_3(reconstructed, q_z, inputs, beta, criterion):
    """
    This function will add the reconstruction loss (BCELoss) and the
    KL-Divergence.
    KL-Divergence = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    :param reconstructed: decoder output of model1
    :param q_z: Learned normal distribution, .loc (mu) .scale(sigma)
    :param past_mse: mse of the previous batch
    """

    # reconstruction loss
    REC = criterion(reconstructed, inputs.detach())

    # KL loss
    mu = q_z.loc
    logvar = (q_z.scale ** 2).log()
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) # 0.5 * sum( var + mu**2 -1 - ln(var))

    # final loss
    LOSS = REC/(2*beta) + KLD   #LOSS =  REC + KLD

    return LOSS, KLD, REC


# Dataset class

class VaeDataset(Dataset):

  def __init__(self,x):

    self.x = torch.tensor(x,dtype=torch.float32)

  def __len__(self):
    return len(self.x)

  def __getitem__(self,idx):
    return self.x[idx]


# training functions


def vae_training_loop(
                  num_epochs: int,
                  optimizer: torch.optim,
                  lr_scheduler: torch.optim.lr_scheduler,
                  model: nn.Module,
                  loader_train: DataLoader,
                  loader_val: DataLoader,
                  betascheduler_train: Betascheduler,
                  betascheduler_val: Betascheduler,
                  criterion: Union[Callable[[torch.Tensor, torch.Tensor], float], Nonetype],
                  patience: int = 50,
                  strategy1: bool = True,
                  verbose: bool = True,) -> Dict:

    """Executes the training loop.

        Args:
            num_epochs: the number of epochs.
            optimizer: the optimizer to use.
            lr_scheduler: the scheduler for the learning rate.
            model: the mode to train.
            loader_train: the data loader containing the training data.
            loader_val: the data loader containing the validation data.
            betascheduler_train: define a cyclic annealing for beta in training data losses to balance the two losses
            betascheduler_val: define a cyclic annealing for beta in validation data losses
            patience: integer define the patience of the training
            strategy1: if 1 use the final_loss1 and beta scheduler
            verbose: if true print the value of loss.
            criterion: nn loss

        Returns:
            A dictionary with the statistics computed during the train:
            the values for the train loss for each epoch.
            the values for the train accuracy for each epoch.
            the values for the validation accuracy for each epoch.
            the time of execution in seconds for the entire loop.
    """

    loop_start = timer()
    losses_values_train = []
    losses_kld_train = []
    losses_rec_train = []
    losses_mae_train = []
    mse_train = sys.maxsize

    losses_values_val = []
    losses_kld_val = []
    losses_rec_val = []
    losses_mae_val = []
    mse_val = sys.maxsize

    last_loss = sys.maxsize
    trigger_times = 0

    for epoch in range(1, num_epochs + 1):
        time_start = timer()
        if strategy1:
            loss_train, kld_train, rec_train, mae_train = training(model, loader_train, 'cpu', optimizer, criterion, betascheduler_train, strategy1, mse_train)
            loss_val, kld_val, rec_val, mae_val = validate(model, loader_val, 'cpu', criterion, betascheduler_val, strategy1, mse_val)
        else:
            loss_train, kld_train, rec_train, mse_train, mae_train = training(model, loader_train, 'cpu', optimizer, criterion,
                                                                betascheduler_train, strategy1, mse_train)
            loss_val, kld_val, rec_val,mse_val, mae_val = validate(model, loader_val, 'cpu', criterion, betascheduler_val, strategy1,
                                                  mse_val)


        time_end = timer()

        losses_values_train.append(loss_train)
        losses_kld_train.append(kld_train)
        losses_rec_train.append(rec_train)
        losses_mae_train.append(mae_train)

        losses_values_val.append(loss_val)
        losses_kld_val.append(kld_val)
        losses_rec_val.append(rec_val)
        losses_mae_val.append(mae_val)

        lr = optimizer.param_groups[0]['lr']

        if verbose:
            print(f'Epoch: {epoch} '
                  f' Lr: {lr:.8f} '
                  f' Loss: Train = [{loss_train:.4f}] - Val = [{loss_val:.4f}] '
                  f' KLD: Train = [{kld_train:.4f}] - Val = [{kld_val:.4f}] '
                  f' REC: Train = [{rec_train:.4f}] - Val = [{rec_val:.4f}] '
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
                            'loss_kld_train': losses_kld_train,
                            'loss_kld_val': losses_kld_val,
                            'loss_rec_train': losses_rec_train,
                            'loss_rec_val': losses_rec_val,
                            'loss_mae_train': losses_mae_train,
                            'loss_mae_val': losses_mae_val,
                            'time': time_loop}
                else:
                    last_loss = loss_val
                    trigger_times = 0

    loop_end = timer()
    time_loop = loop_end - loop_start
    if verbose:
        print(f'Time for {num_epochs} epochs (s): {(time_loop):.3f}')

    return {'loss_values_train': losses_values_train,
            'loss_values_val': losses_values_val,
            'loss_kld_train': losses_kld_train,
            'loss_kld_val': losses_kld_val,
            'loss_rec_train': losses_rec_train,
            'loss_rec_val': losses_rec_val,
            'loss_mae_train': losses_mae_train,
            'loss_mae_val': losses_mae_val,
            'time': time_loop}

# Train one epoch
def training(model: nn.Module,
          train_loader: DataLoader,
          device: torch.device,
          optimizer: torch.optim,
          criterion: Union[Callable[[torch.Tensor, torch.Tensor], float], Nonetype],
          beta_scheduler: Betascheduler,
          strategy1: bool,
          mse: float) -> Tuple[float, float]:

    """Trains a neural network for one epoch.

    Args:
        model: the model to train.
        train_loader: the data loader containing the training data.
        device: the device to use to train the model.
        optimizer: the optimizer to use to train the model.
        criterion: the loss to optimize.
        beta_scheduler: either a scheduler or balancer for beta in kl and rec losses
        strategy1: if true final_loss1 and model1
        mse: mse in previous epoch

    Returns:
        the Loss value on the training data.
        the average mse of the epoch
    """

    loss_train = 0
    kld_loss = 0
    rec_loss = 0
    mae_score = 0
    samples_train = 0
    model.train()

    for idx_batch, x in enumerate(train_loader):
        x = x.to(device)
        x.requires_grad = True
        optimizer.zero_grad()

        reconstruction, q_z = model(x)

        if not strategy1:
            current_mse = torch.mean((x.detach() - reconstruction.detach()) ** 2)
            gamma = min(mse,current_mse)
            mse = gamma
            loss, kld, rec = final_loss_3(reconstruction,q_z,x, gamma, criterion)

        else:
            loss, kld, rec = final_loss_1(reconstruction, q_z, x, beta_scheduler.beta, criterion)
            beta_scheduler.step()

        kld_loss += kld.item() * len(x)
        rec_loss += rec.item() * len(x)
        loss_train += loss.item() * len(x)
        mae_score += torch.mean(torch.abs((x.detach() - reconstruction.detach()))) * len(x)
        samples_train += len(x)

        loss.backward()
        optimizer.step()

    loss_train /= samples_train
    kld_loss /= samples_train
    rec_loss /= samples_train
    mae_score /= samples_train

    if not strategy1:
    #    mse_loss /= samples_train
        return loss_train, kld_loss, rec_loss , mse, mae_score

    else:
        return loss_train, kld_loss, rec_loss, mae_score


# Validate one epoch
def validate(model: nn.Module,
             data_loader: DataLoader,
             device: torch.device,
             criterion: Union[Callable[[torch.Tensor, torch.Tensor], float],Nonetype],
             beta_scheduler: Betascheduler,
             strategy1: bool,
             mse: float) -> Tuple[float, float]:
    """Evaluates the model.

    Args:
        model: the model to evalaute.
        data_loader: the data loader containing the validation or test data.
        device: the device to use to evaluate the model.
        criterion: the loss function.
        mse: loss in the past epoch

    Returns:
        the loss value on the validation data.
        the accuracy on the validation data.
    """

    loss_val = 0
    kld_loss = 0
    rec_loss = 0
    mae_score = 0
    samples_val = 0

    model = model.eval()

    with torch.no_grad():
        for idx_batch, x in enumerate(data_loader):
            x = x.to(device)
            reconstruction, q_z = model(x)

            if not strategy1:
                current_mse = torch.mean((x - reconstruction) ** 2)
                gamma = min(mse, current_mse)
                mse = gamma
                #mse_loss += current_mse * len(x)

            if not strategy1:
                loss, kld, rec = final_loss_3(reconstruction, q_z, x, gamma, criterion)

            else:
                loss, kld, rec = final_loss_1(reconstruction, q_z, x, beta_scheduler.beta, criterion)
                beta_scheduler.step()

            kld_loss += kld.item() * len(x)
            rec_loss += rec.item() * len(x)
            loss_val += loss.item() * len(x)
            mae_score += torch.mean(torch.abs((x.detach() - reconstruction.detach()))) * len(x)
            samples_val += len(x)

    loss_val /= samples_val
    kld_loss /= samples_val
    rec_loss /= samples_val
    mae_score /= samples_val
    if not strategy1:
        #mse_loss /= samples_val
        return loss_val, kld_loss, rec_loss, mse, mae_score
    else:
        return loss_val, kld_loss, rec_loss, mae_score

#Model 1

class Vae_Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, n_hidden = 1):
        super(Vae_Encoder,self).__init__()

        # encoder
        self.enc1 = nn.Linear(input_dim, hidden_dim)
        self.enc_hidden_layers = torch.nn.ModuleList([torch.nn.Linear(hidden_dim, hidden_dim) for i in range(n_hidden)])
        self.enc2 = nn.Linear(hidden_dim, latent_dim * 2)
        self.latent_dim = latent_dim
        self.activation = F.relu

    def forward(self,x):

        # encoding
        x = self.enc1.forward(x)
        x = self.activation(x)
        for layer in self.enc_hidden_layers:  # iteration over hidden layers(initialized as LC)
            x = layer.forward(x)
            x = self.activation(x)
        x = self.enc2.forward(x).view(-1,2,self.latent_dim) # [n, latent_dim * 2] --> [n,2, latent_dim]

        mu = x[:,0,:]
        log_var = x[:,1,:]

        return torch.distributions.Normal(loc=mu, scale=torch.exp(log_var))



class Vae_Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, n_hidden=1):
        super(Vae_Decoder, self).__init__()

        # decoder
        self.dec1 = nn.Linear(latent_dim, hidden_dim)
        self.dec_hidden_layers = torch.nn.ModuleList([torch.nn.Linear(hidden_dim, hidden_dim) for i in range(n_hidden)])
        self.dec2 = nn.Linear(hidden_dim, input_dim)
        self.latent_dim = latent_dim
        self.activation = F.relu
        self.out_activation = torch.sigmoid

    def forward(self,x):

        # decoding
        x = self.dec1.forward(x)
        x = self.activation(x)
        for layer in self.dec_hidden_layers:  # iteration over hidden layers(initialized as LC)
            x = layer.forward(x)
            x = self.activation(x)
        x = self.dec2.forward(x)
        reconstructed = self.out_activation(x)
        return reconstructed

class Vae(nn.Module):
    def __init__(self, encoder, decoder):
        super(Vae,self).__init__()

        self.encoder = encoder
        self.decoder = decoder

    def reparameterise(self, q_z):
        if self.training:
            z = q_z.rsample()
        else:
            z = q_z.loc
        return z

    def forward(self,x):
        # encoding
        q_z = self.encoder.forward(x)
        z = self.reparameterise(q_z)
        reconstructed = self.decoder(z)
        return reconstructed, q_z






