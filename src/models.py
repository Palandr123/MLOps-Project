from torch import nn


class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_units, output_size):
        super(SimpleNN, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_units)
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_units, output_size)

    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        return x
