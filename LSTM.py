import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.optim.lr_scheduler import _LRScheduler
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader
from torchmetrics import Metric
import pytorch_forecasting.metrics.point as metrics
import numpy as np
import pandas as pd
from functools import partial
from einops import rearrange
import random


class LSTM_model(nn.Module):
    def __init__(self, input_shape=(1, 1), opt="Adam", activation='prelu', 
                 add_layer=1, epochs=200, batch_size=20, weight_decay=0.005, 
                 lrate=0.001, nodes_n=100, norm=False, split=0.0, unroll=False, ahead=10,
                 lstm_dropout=0.2, trg=None, drop=False, device="cuda"):
        super(LSTM_model, self).__init__()
        
        # Hyperparameters
        self.input_shape = input_shape
        self.seq_len = input_shape[0]
        self.activation = activation
        self.add_layer = add_layer
        self.nodes_n = nodes_n
        self.ahead = ahead
        self.device = device
        self.norm = norm
        self.last_drop = drop


        # LSTM layer
        self.lstm = nn.LSTM(input_size=input_shape[-1], 
                            hidden_size=nodes_n, 
                            num_layers=add_layer, 
                            dropout=lstm_dropout if add_layer > 1 else 0,
                            batch_first=True, 
                            bidirectional=False)
        
        # Layer Normalization
        if self.norm:
            self.layer_norm = nn.LayerNorm(nodes_n)
        
        # # Final Dense Layer
        # self.lstm_dense = nn.Linear(nodes_n, nodes_n)
        # self.tanh = nn.Tanh()
        self.output_dense = nn.Linear(nodes_n, input_shape[-1])
        
        # Activation
        if activation == 'tanh':
            self.activation_fn = nn.Tanh()
        elif activation == 'prelu':
            self.activation_fn = nn.PReLU()
        elif activation == 'relu':
            self.activation_fn = nn.ReLU()
        elif activation == 'softmax':
            self.activation_fn = nn.Softmax2d()
        elif activation == 'mish':
            self.activation_fn = nn.Mish()
        elif activation == "gelu":
            self.activation_fn = nn.GELU()
        elif activation == "silu":
            self.activation_fn = nn.SiLU()
        elif activation == "none":
            self.activation_fn = None
            
        
        # Loss Function
        self.loss_fn = CustomLoss(ahead=self.ahead, device=self.device)

        # Optional: Dropout after LSTM
        self.dropout = nn.Dropout(p=lstm_dropout)

    def forward(self, x):
        """
        Forward pass with variable-length sequence handling.
        
        Args:
            inputs: Tensor of shape (batch_size, seq_len, input_dim)
            input_lengths: List or tensor of sequence lengths before padding
            
        Returns:
            final_output: Processed output tensor
        """
        output, (hidden, cell) = self.lstm(x)
        outputs = output[:, -self.ahead:, :] # probably the last 10 time steps is not good strategy
        
        if self.last_drop:
            outputs = self.dropout(outputs)

        # Apply Layer Normalization and Dense layer if needed
        if self.norm:
            outputs = self.layer_norm(outputs)
        outputs = self.output_dense(outputs)
        
        # Final Activation
        if self.activation_fn:
            final_output = self.activation_fn(outputs)
            return final_output
        return outputs



class CustomLoss:
    def __init__(self, device="cuda", ahead=10):
        self.ahead = ahead
        self.device = device
        print(f"Loss is calculated for the last {self.ahead} time steps")
        self.mae = nn.L1Loss(reduction = 'mean') #sum mean none
        self.mae2 = nn.L1Loss(reduction = 'none')
        self.mse = nn.MSELoss(reduction = 'mean')
        self.mape = metrics.MAPE(reduction = 'mean')
        self.mase = metrics.MASE(reduction = 'mean')
        self.cos = nn.CosineSimilarity(dim=-1)

    def mae_loss(self, y_true, y_pred):
        y_true_last = y_true[:, -self.ahead:, :]
        y_pred_last = y_pred[:, -self.ahead:, :]
        return self.mae(y_pred_last, y_true_last)

    def EuclideanPerFeature(pred, actual, f):
        eu_per_f = {}
        for period in range(len(pred)):
            for day in range(0, len(pred[period])):
                day_dist = abs(pred[period][day][f] - actual[period][day][f])
                if day in eu_per_f:
                    eu_per_f[day].append(day_dist)
                else:
                    eu_per_f[day] = [day_dist]
        return eu_per_f

    def custom_mae_loss(self, y_true, y_pred, threshold=0.1, penalty=2):
        y_true_last = y_true[:, -self.ahead:, :]
        y_pred_last = y_pred[:, -self.ahead:, :]
        loss = torch.abs(torch.mean(y_true_last - y_pred_last, dim=1))
        mask = loss > threshold
        loss[mask] = loss[mask] * penalty
        return torch.mean(loss)

    def multivar_loss(self, y_true, y_pred):
        y_true_last = y_true[:, -self.ahead:, :]
        y_pred_last = y_pred[:, -self.ahead:, :]
        true_mean = torch.mean()
        return self.mae(y_pred_last, y_true_last)
        
    def mape_loss(self, y_true, y_pred):
        return self.mape(y_pred, y_true)

    def mase_loss(self, y_true, y_pred, length=28):
        y_true_last = y_true[:, -self.ahead:, :]
        y_pred_last = y_pred[:, -self.ahead:, :]
        return self.mase(y_pred_last, y_true_last, y_pred_last, length)

    def short_distance(self, y_true, y_pred):
        y_pred_last = y_pred[:, -self.ahead:, :]
        y_true_last = y_true[:, -self.ahead:, :]
        pred_trend = torch.sum(y_pred_last[:, -1, :] - y_pred_last[:, 0, :], -1)/self.ahead
        true_trend = torch.sum(y_true_last[:, -1, :] - y_true_last[:, 0, :], -1)/self.ahead
        return self.mae(pred_trend, true_trend)

    def cosine_simillaruty_metrics(self, y_true, y_pred):
        y_true_last = y_true[:, -self.ahead, :]
        y_pred_last = y_pred[:, -self.ahead, :]
        return self.cos(y_pred_last, y_true_last)

    def single_mae_loss(self, y_true, y_pred):
        return self.mae(y_true, y_pred)

    def nextDay_mae_loss(self, y_true, y_pred):
        y_true_last = y_true[:, -self.ahead, :]
        y_pred_last = y_pred[:, -self.ahead, :]
        return self.mae(y_pred_last, y_true_last)

    def variance_loss(self, y_true, y_pred):
        var_true = torch.var(y_true[:, -self.ahead:, :23], 1) ** 0.5
        var_pred = torch.var(y_pred[:, -self.ahead:, :23], 1) ** 0.5
        return self.mae(var_pred, var_true)

    def var_error_loss(self, y_true, y_pred):
        y_true_last = y_true[:, -self.ahead, :]
        y_pred_last = y_pred[:, -self.ahead, :]
        var_error = torch.var(torch.abs(y_true_last - y_pred_last)) ** 0.5
        return var_error

    def central_loss(self, y_true, y_pred):
        mean_true = torch.mean(y_true[:, -self.ahead:, :], 1)
        mean_pred = torch.mean(y_pred[:, -self.ahead:, :], 1)
        return self.mae(mean_pred, mean_true)

    def main_strains_loss(self, y_true, y_pred):
        y_true_last = torch.cat((y_true[:, -self.ahead:, 4], y_true[:, -self.ahead:, 11], y_true[:, -self.ahead:, 14]), dim=-1)
        y_pred_last = torch.cat((y_pred[:, -self.ahead:, 4], y_pred[:, -self.ahead:, 11], y_pred[:, -self.ahead:, 14]), dim=-1)
        return self.mae(y_pred_last, y_true_last)
        
    def minor_strains_loss(self, y_true, y_pred):
        y_true_last = torch.cat((y_true[:, -self.ahead:, 0], y_true[:, -self.ahead:, 2], y_true[:, -self.ahead:, 12]), dim=-1)
        y_true_last = torch.cat((y_true_last[:, :], y_true[:, -self.ahead:, 15]), dim=-1)
        y_pred_last = torch.cat((y_pred[:, -self.ahead:, 0], y_pred[:, -self.ahead:, 2], y_true[:, -self.ahead:, 12]), dim=-1) 
        y_pred_last = torch.cat((y_pred_last[:, :], y_pred[:, -self.ahead:, 15]), dim=-1)
        return self.mae(y_pred_last, y_true_last)

    def tech_loss(self, y_true, y_pred):
        y_true_last = y_true[:, -self.ahead:, 25:]
        y_pred_last = y_pred[:, -self.ahead:, 25:]
        return self.mae(y_pred_last, y_true_last)

    def seq_loss(self, y_true, y_pred):
        y_true_last = y_true[:, -self.ahead:, 21:23]
        y_pred_last = y_pred[:, -self.ahead:, 21:23]
        return self.mae(y_pred_last, y_true_last)

    def media_loss(self, y_true, y_pred):
        y_true_last = y_true[:, -self.ahead:, 23:25]
        y_pred_last = y_pred[:, -self.ahead:, 23:25]
        return self.mae(y_pred_last, y_true_last)
    
    def strain_mae_loss(self, y_true, y_pred):
        y_true_last = y_true[:, -self.ahead:, :23]
        y_pred_last = y_pred[:, -self.ahead:, :23]
        return self.mae(y_pred_last, y_true_last)

    def all_tech_loss(self, y_true, y_pred):
        y_true_last = y_true[:, -self.ahead:, 23:]
        y_pred_last = y_pred[:, -self.ahead:, 23:]
        return self.mae(y_pred_last, y_true_last)

    def bc_loss(self, y_true, y_pred):
        y_true_last = y_true[:, -self.ahead:, :21]
        y_pred_last = y_pred[:, -self.ahead:, :21]
        bc = torch.mean(torch.sum(torch.abs(y_true_last - y_pred_last), dim=-1) / torch.sum(y_true_last + y_pred_last, dim=-1))
        return bc

    def penaltie_mae_loss(self, y_true, y_pred):
        y_true_last = y_true[:, -self.ahead:, :]
        y_pred_last = y_pred[:, -self.ahead:, :]
        error_scores = torch.from_numpy(np.arange(self.ahead, 0, -1)).to(torch.device(self.device)).float()
        y_true_scored = (y_true_last.permute(0, 2, 1) * error_scores).to(torch.device(self.device)).float()
        y_pred_scored = (y_pred_last.permute(0, 2, 1) * error_scores).to(torch.device(self.device)).float()
        return self.mae(y_pred_scored, y_true_scored)        
        
    def mse_loss(self, y_true, y_pred):
        y_true_last = y_true[:, -self.ahead:, :]
        y_pred_last = y_pred[:, -self.ahead:, :]
        return self.mse(y_pred_last, y_true_last)

    def last_timestep_mae(self, y_true, y_pred):
        y_true_last = y_true[:, -self.ahead, :]
        y_pred_last = y_pred[:, -self.ahead, :]
        return self.mae(y_pred_last, y_true_last)

    def sum_loss(self, y_true, y_pred):
        y_true_last = y_true[:, -self.ahead, :]
        y_pred_last = y_pred[:, -self.ahead, :]
        total_loss = y_true_last - y_pred_last
        return self.mae(y_pred_last, y_true_last)

    def mae_full(self, y_true, y_pred):
        return self.mae(y_pred, y_true)

    def mse_full(self, y_true, y_pred):
        return self.mse(y_pred, y_true)
    

def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight' in name:  # Apply initialization only to weight tensors
                if param.dim() >= 2:  # Ensure the tensor is at least 2D
                    nn.init.kaiming_uniform_(param.data, mode='fan_in', nonlinearity='relu')
                    print("Kaiming uniform is applied.")
                else:
                    nn.init.uniform_(param.data, -0.08, 0.08)  # Fallback for 1D tensors
            elif 'bias' in name:  # Initialize biases
                nn.init.constant_(param.data, 0)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_optimizer(opt_name, model_parameters, lr=0.001, weight_decay=0.005, momentum=0.9):
    if opt_name.lower() == "adam":
        optimizer = Adam(model_parameters, lr=lr)
    elif opt_name.lower() == "sgd":
        optimizer = torch.optim.SGD(model_parameters, lr=lr, momentum=momentum, nesterov=True)
    elif opt_name.lower() == "adamw":
        optimizer = torch.optim.AdamW(model_parameters, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {opt_name}")
    return optimizer

def create_scheduler(optimizer, step_size=10, gamma=0.5):
    """
    Creates a learning rate scheduler that reduces the learning rate by a factor every step_size epochs.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer for which to schedule the learning rate.
        step_size (int): Number of epochs between learning rate reductions.
        gamma (float): Factor by which the learning rate is reduced.

    Returns:
        torch.optim.lr_scheduler.StepLR: A learning rate scheduler.
    """
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    return scheduler


class DynamicStepScheduler(_LRScheduler):
    def __init__(self, optimizer, initial_step_size=10, gamma=0.5, warm=False, last_epoch=-1, verbose=False):
        """
        Custom scheduler to reduce learning rate by a factor and dynamically increase step size.

        Args:
            optimizer (torch.optim.Optimizer): Wrapped optimizer.
            initial_step_size (int): Initial step size for reducing the learning rate.
            gamma (float): Factor to multiply the learning rate at each step.
            last_epoch (int): The index of the last epoch. Defaults to -1.
            verbose (bool): If True, prints learning rate updates. Defaults to False.
        """
        self.step_size = initial_step_size
        self.initial_step_size = initial_step_size
        self.gamma = gamma
        self.beta = 1/gamma
        self.warm = warm
        self.update = False
        super(DynamicStepScheduler, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self.warm == True:
            if (self.last_epoch + 1) == self.step_size and self.last_epoch >= 0:
                # Increase step size and reduce learning rate
                # self.update = True
                # if self.update
                #     return [group['lr'] * self.beta for group in self.optimizer.param_groups]
                # else:
                return [group['lr'] * self.gamma for group in self.optimizer.param_groups]
        else:
            if (self.last_epoch + 1) % self.step_size == 0 and self.last_epoch >= 0:
                # Increase step size and reduce learning rate
                self.step_size += 2 * self.initial_step_size
                return [group['lr'] * self.gamma for group in self.optimizer.param_groups]
        return [group['lr'] for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        # Closed form for logging (optional, for verbose=True)
        return [base_lr * (self.gamma ** ((self.last_epoch + 1) // self.step_size))
                for base_lr in self.base_lrs]


