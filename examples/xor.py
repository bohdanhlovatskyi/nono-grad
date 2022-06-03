import numpy as np
from tqdm import tqdm

from nonograd.tensor import Tensor
from nonograd.optim import Adam
from nonograd.models.xornynet import XornyNet

def Xor():
    model = XornyNet()
    optim = Adam([model.l1, model.l2])
    Xs = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    ys = np.array([[0.,], [1.,], [1.,], [0.,]])

    # # train loop
    epochs = 100000
    for i in tqdm(range(epochs)):
        X = Tensor(Xs)
        y = Tensor(ys)
        out = model.forward(X)
        loss = out.mse(y)
        if np.isnan(loss.data):
            exit(0)

        assert loss.requires_grad
        # print(i, loss)

        optim.zero_grad()
        loss.backward()
        if i % 1000 == 0:
            print(loss)
        # print(f"iteration #{i}: loss: {loss}\n l1_grad: {model.l1.grad}, l2_grad: {model.l2.grad}, l1_value: {model.l1}, l2_value: {model.l2}")
        optim.step()

    for elm in Xs:
        out = model.forward(Tensor(elm))
        print(elm, out)
