import random
import numpy as np
import pandas as pd

import torch
from torch import nn
from skorch import NeuralNetRegressor


class SimpleNN(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_units,
        num_blocks,
        output_size,
        embed_dim,
        num_regions,
        region_idx,
        num_wmis,
        wmi_idx,
        num_vds,
        vds_idx,
        num_models,
        model_idx,
        seed,
    ):
        random.seed(int(seed))
        np.random.seed(int(seed))
        torch.manual_seed(int(seed))
        super(SimpleNN, self).__init__()
        self.embed_region = nn.Embedding(int(num_regions), int(embed_dim))
        self.embed_wmi = nn.Embedding(int(num_wmis), int(embed_dim))
        self.embed_vds = nn.Embedding(int(num_vds), int(embed_dim))
        self.embed_model = nn.Embedding(int(num_models), int(embed_dim))
        num_units = int(input_size)-4+4*int(embed_dim)
        modules = []
        for i in range(int(num_blocks)):
            if i == 0:
                modules.append(nn.Sequential(nn.Linear(num_units, int(hidden_units)), nn.ReLU()))
                num_units = int(hidden_units) 
            elif i == int(num_blocks) - 1:
                modules.append(nn.Linear(num_units, int(output_size)))
            else:
                modules.append(nn.Sequential(nn.Linear(num_units, num_units // 2), nn.ReLU()))
                num_units = num_units // 2
        self.linears = nn.Sequential(*modules)
        self.region_idx = int(region_idx)
        self.wmi_idx = int(wmi_idx)
        self.vds_idx = int(vds_idx)
        self.model_idx = int(model_idx)

    def forward(self, x):
        mask = torch.ones(x.shape[1], dtype=torch.bool)
        mask[self.region_idx] = False
        mask[self.wmi_idx] = False
        mask[self.vds_idx] = False
        mask[self.model_idx] = False
        region = self.embed_region(x[:, self.region_idx].int())
        wmi = self.embed_wmi(x[:, self.wmi_idx].int())
        vds = self.embed_vds(x[:, self.vds_idx].int())
        model = self.embed_model(x[:, self.model_idx].int())
        x = x[:, mask]
        x = self.linears(torch.cat([x, region, wmi, vds, model], dim=1))
        return x
    

class MLP(nn.Module):
    def __init__(self, input_size, hidden_units1, hidden_units2, hidden_units3, output_size, seed):
        random.seed(int(seed))
        np.random.seed(int(seed))
        torch.manual_seed(int(seed))
        super(MLP, self).__init__()
        self.hidden1 = nn.Linear(int(input_size), int(hidden_units1))
        self.hidden2 = nn.Linear(int(hidden_units1), int(hidden_units2))
        self.hidden3 = nn.Linear(int(hidden_units2), int(hidden_units3))
        self.relu = nn.ReLU()
        self.output = nn.Linear(int(hidden_units3), int(output_size))

    def forward(self, x):
        x = self.hidden1(x)
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
    
class NNWrapper(NeuralNetRegressor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __prepare_features(self, X):
        if isinstance(X, pd.DataFrame):
            return X.values.astype(np.float32)
        return X
    
    def __prepare_targets(self, y):
        if isinstance(y, pd.Series):
            return y.values.astype(np.float32).reshape(-1, 1)
        return y
    
    def fit(self, X, y, **kwargs):
        X = self.__prepare_features(X)
        y = self.__prepare_targets(y)
        return super().fit(X, y, **kwargs)
    
    def predict(self, X):
        X = self.__prepare_features(X)
        return super().predict(X)
    
    def score(self, X, y):
        X = self.__prepare_features(X)
        y = self.__prepare_targets(y)
        return super().score(X, y)
