import torch
import numpy as np
from nonograd.tensor import Tensor

class MNISTNet:
  def __init__(self):
    # epsilon = 0.12
    # self.l1 = Tensor(np.random.rand(784, 25)*epsilon*2 - epsilon, requires_grad=True)
    # self.l2 = Tensor(np.random.rand(25, 10)*epsilon*2 - epsilon, requires_grad=True)
    w = torch.empty(784, 25)
    torch.nn.init.xavier_uniform_(w, gain=torch.nn.init.calculate_gain('sigmoid'))
    self.l1 = Tensor(np.array(w.data), requires_grad=True)

    w = torch.empty(25, 10)
    torch.nn.init.xavier_uniform_(w, gain=torch.nn.init.calculate_gain('sigmoid'))
    self.l2 = Tensor(np.array(w.data), requires_grad=True)

  def forward(self, x: 'Tensor'):
    return x.dot(self.l1).sigmoid().dot(self.l2).sigmoid()