import numpy as np
from tqdm import tqdm

from examples import load_mnist
from nonograd.tensor import Tensor
from nonograd.optim import SGD
from nonograd.models.mnist_net import MNISTNet

def validation(model, Xv, Yv):
    iteration = 0
    tr = 0

    for x, y in tqdm(zip(Xv, Yv)):
        x, y = Tensor(x.reshape(1, 784)),\
               Tensor(np.eye(10)[y, :].reshape(10, 1))
        out = model.forward(x)

        if np.argmax(out.data) == np.argmax(y.data):
            tr += 1

        iteration += 1

    return tr/iteration

def Mnist(load: bool = False):
    # load dataset
    if load:
        load_mnist.init()

    # Accuracy: 0.8727 for 10 epochs

    Xt, yt, Xv, Yv = load_mnist.load()
    # create model
    model = MNISTNet()
    optim = SGD([model.l1, model.l2], lr=0.01)

    # train loop
    iteration = 0
    for epoch in range(15):
        loss = None
        for x, y in tqdm(zip(Xt, yt)):
            iteration += 1
            x, y = Tensor(x.reshape(1, 784)),\
                Tensor(np.eye(10)[y, :].reshape(1, 10))

            out = model.forward(x)
            loss = out.mse(y)
            assert loss.requires_grad

            optim.zero_grad()
            loss.backward()
            optim.step()

        acc = validation(model, Xv, Yv)
        print(f"[{epoch}] acc: {acc}; loss: {loss}")
