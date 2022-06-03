import numpy as np
from nonograd.tensor import Tensor

class SGD:
    def __init__(self, params, lr: float = 0.01, alpha: float = 0.001 ) -> None:
        self.lr = lr
        self.alpha = alpha
        self.params = params # list of tensors

    def step(self) -> None:
        for parameter in self.params:
            parameter.data -= (parameter.grad + self.alpha * parameter.data) * self.lr

    def zero_grad(self):
        for param in self.params:
            param.grad = np.zeros(param.grad.shape)

class Adam():
  def __init__(self, params, lr=0.001, b1=0.9, b2=0.999, eps=1e-8):
    self.params = params
    self.lr, self.b1, self.b2, self.eps, self.t = lr, b1, b2, eps, 0

    self.m = [Tensor(np.zeros(t.shape), requires_grad=False) for t in self.params]
    self.v = [Tensor(np.zeros(t.shape), requires_grad=False) for t in self.params]

  def step(self):
    self.t = self.t + 1
    a = self.lr * ((1.0 - self.b2**self.t)**0.5) / (1.0 - self.b1**self.t)
    for i, t in enumerate(self.params):
    #   print(self.m[i], self.b1 , t.grad ,(1.0 - self.b1))
    #   print(self.m[i] * self.b1 , t.grad * (1.0 - self.b1), type(self.b1))
      self.m[i] = (self.m[i] * self.b1) + Tensor((t.grad * (1.0 - self.b1)))
      self.v[i] = (self.v[i] * self.b2) + Tensor((t.grad * t.grad * (1.0 - self.b2)))

    #   t -= self.m[i] * (a / (np.sqrt(self.v[i].data) + self.eps))
    #   print(self.m[i].div(self.v[i].sqrt() + self.eps) * a)
      t.data -= (self.m[i].div(self.v[i].sqrt() + self.eps) * a).data

  def zero_grad(self):
        for param in self.params:
            param.grad = np.zeros(param.grad.shape)
