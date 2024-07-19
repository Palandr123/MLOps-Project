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
        num_wmis,
        wmis_embed_dim,
        wmi_idx,
        num_vds,
        vds_embed_dim,
        vds_idx,
        num_models,
        models_embed_dim,
        model_idx,
        seed,
    ):
        random.seed(int(seed))
        np.random.seed(int(seed))
        torch.manual_seed(int(seed))
        super(SimpleNN, self).__init__()
        self.embed_region = nn.Embedding(int(num_regions), int(regions_embed_dim))
        self.embed_wmi = nn.Embedding(int(num_wmis), int(wmis_embed_dim))
        self.embed_vds = nn.Embedding(int(num_vds), int(vds_embed_dim))
        #self.embed_model = nn.Embedding(int(num_models), int(models_embed_dim))
        self.hidden1 = nn.Linear(int(input_size)-3+int(regions_embed_dim)+int(wmis_embed_dim)+int(vds_embed_dim), int(hidden_units1))
        self.hidden2 = nn.Linear(int(hidden_units1), int(hidden_units2))
        self.hidden3 = nn.Linear(int(hidden_units2), int(hidden_units3))
        self.relu = nn.ReLU()
        self.output = nn.Linear(int(hidden_units3), int(output_size))
        self.region_idx = int(region_idx)
        self.wmi_idx = int(wmi_idx)
        self.vds_idx = int(vds_idx)
        #self.model_idx = int(model_idx)

    def forward(self, x):
        mask = torch.ones(x.shape[1], dtype=torch.bool)
        mask[self.region_idx] = False
        mask[self.wmi_idx] = False
        mask[self.vds_idx] = False
        #mask[self.model_idx] = False
        region = self.embed_region(x[:, self.region_idx].int())
        wmi = self.embed_wmi(x[:, self.wmi_idx].int())
        vds = self.embed_vds(x[:, self.vds_idx].int())
        x = x[:, mask]
        x = self.hidden1(torch.cat([x, region, wmi, vds], dim=1))
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
