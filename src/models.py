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
