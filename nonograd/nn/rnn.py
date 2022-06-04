import torch
import torch.nn as nn
import numpy as np
from nonograd.tensor import Tensor

# https://github.com/kaustubhhiware/LSTM-GRU-from-scratch/blob/master/module.py
# https://pytorch.org/docs/stable/generated/torch.nn.GRU.html
class Model(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(Model, self).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        #Defining the layers
        # RNN Layers
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)   
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_size)
    
    def forward(self, x):
        
        batch_size = x.size(0)

        # Initializing hidden state for first input using method defined below
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(x, hidden)
        
        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)
        
        return out, hidden


class GRUCell:
    def __init__(self, input_shape, activation_shape) -> None:
        pass

    def forward(input, activation):
        pass
