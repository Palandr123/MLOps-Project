import random
import numpy as np

import torch
from torch import nn


class SimpleNN(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_units1,
        hidden_units2,
        hidden_units3,
        output_size,
        num_regions,
        regions_embed_dim,
        region_idx,
        seed,
    ):
        random.seed(int(seed))
        np.random.seed(int(seed))
        torch.manual_seed(int(seed))
        super(SimpleNN, self).__init__()
        self.embed_region = nn.Embedding(num_regions, regions_embed_dim)
        self.hidden1 = nn.Linear(int(input_size)-1+regions_embed_dim, int(hidden_units1))
        self.hidden2 = nn.Linear(int(hidden_units1), int(hidden_units2))
        self.hidden3 = nn.Linear(int(hidden_units2), int(hidden_units3))
        self.relu = nn.ReLU()
        self.output = nn.Linear(int(hidden_units3), int(output_size))
        self.region_idx = region_idx

    def forward(self, x):
        mask = torch.ones(x.shape[1], dtype=torch.bool)
        mask[self.region_idx] = False
        region = self.embed_region(x[:, self.region_idx].int())
        x = x[:, mask]
        x = self.hidden1(torch.cat([x, region], dim=1))
        x = self.relu(x)
        x = self.hidden2(x)
        x = self.relu(x)
        x = self.hidden3(x)
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
            [
                ResNetBlock(int(hidden_units), int(hidden_units))
                for _ in range(int(num_blocks))
            ]
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
