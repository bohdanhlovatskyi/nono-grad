import torch
import numpy as np
from nonograd.tensor import Tensor

class XornyNet:
    def __init__(self) -> None:
        w = torch.empty(2, 2)
        torch.nn.init.xavier_uniform_(w, gain=torch.nn.init.calculate_gain('relu'))
        self.l1 = Tensor(np.array(w.data), requires_grad=True)

        w = torch.empty(2, 1)
        torch.nn.init.xavier_uniform_(w, gain=torch.nn.init.calculate_gain('relu'))
        self.l2 = Tensor(np.array(w.data), requires_grad=True)

        print("\n\n\n\n", self.l1, self.l2, "\n\n\n\n")

    def forward(self, x: 'Tensor'):
        w = x @ self.l1
        w = w.relu()
        w = w @ self.l2
        w = w.relu()
        return w
