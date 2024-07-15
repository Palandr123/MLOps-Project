import random
import numpy as np

import torch
from torch import nn


class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_units, output_size, seed):
        random.seed(int(seed))
        np.random.seed(int(seed))
        torch.manual_seed(int(seed))
        super(SimpleNN, self).__init__()
        self.hidden = nn.Linear(int(input_size), int(hidden_units))
        self.relu = nn.ReLU()
        self.output = nn.Linear(int(hidden_units), int(output_size))

    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        return x


# Fully connected ResNet implementation
class FullyConnectedResNet(nn.Module):
    def __init__(self, input_size, hidden_units, output_size, num_blocks, seed):
        random.seed(int(seed))
        np.random.seed(int(seed))
        torch.manual_seed(int(seed))
        super(FullyConnectedResNet, self).__init__()
        self.input_layer = nn.Linear(int(input_size), int(hidden_units))

        self.blocks = nn.ModuleList(
            [ResNetBlock(int(hidden_units), int(hidden_units)) for _ in range(int(num_blocks))]
        )
        self.relu = nn.ReLU()

        self.output_layer = nn.Linear(int(hidden_units), int(output_size))

    def forward(self, x):
        x = self.relu(self.input_layer(x))
        for block in self.blocks:
            x = block(x)
        x = self.output_layer(x)
        return x


class ResNetBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ResNetBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.fc2 = nn.Linear(output_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        x += residual
        x = self.relu(x)
        return x
