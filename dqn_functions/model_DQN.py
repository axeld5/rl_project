import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy as np
import copy

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_sup_layers):
        super().__init__()
        self.first_layer = nn.Linear(input_size, hidden_size)
        self.mid_layers = [nn.ReLU()]
        for _ in range(n_sup_layers):
            self.mid_layers.append(nn.Linear(hidden_size, hidden_size))
            self.mid_layers.append(nn.ReLU())
        self.mid_layer = nn.Sequential(*self.mid_layers)
        self.last_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.mid_layer(self.first_layer(x))
        x = self.last_layer(x)
        return x
