import sys 
import torch
import unittest
import numpy as np

sys.path.append('..')
from nonograd.tensor import Tensor

class TestNono(unittest.TestCase):

    def test_mul(self):
        a = torch.tensor([2., 3.], requires_grad=True)
        b = torch.tensor([6., 2.], requires_grad=True)
        z = a.matmul(b)
        z.backward()

        ma = Tensor(np.array([2., 3.]).reshape((1, 2)), requires_grad=True)
        mb = Tensor(np.array([6., 2.]).reshape((2, 1)), requires_grad=True)
        mz = ma @ mb
        mz.backward()

        a = a.grad.cpu().detach().numpy().reshape((2, 1))
        b = b.grad.cpu().detach().numpy().reshape((2, 1))

        ma = ma.grad.data.reshape((2, 1))
        mb = mb.grad.data.reshape((2, 1))

        self.assertTrue(np.allclose(a, ma))
        self.assertTrue(np.allclose(b, mb))

    @unittest.skip("seems deprecated")
    def test_relu(self):
        a = torch.tensor([2., 3.], requires_grad=True)
        m = torch.nn.ReLU()
        b = m(a)
        z = a.matmul(b)
        z.backward()

        ma = Tensor(np.array([2., 3.]).reshape((1, 2)), requires_grad=True)
        mb = ma.relu()
        mz = ma @ mb.T
        mz.backward()

        a = a.grad.cpu().detach().numpy().reshape((2, 1))
        b = b.grad.cpu().detach().numpy().reshape((2, 1))

        ma = ma.grad.data.reshape((2, 1))
        mb = mb.grad.data.reshape((2, 1))

        self.assertTrue(np.allclose(a, ma))
        self.assertTrue(np.allclose(b, mb))

if __name__ == '__main__':
    unittest.main()
