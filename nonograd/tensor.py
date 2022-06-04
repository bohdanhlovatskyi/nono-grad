import numpy as np 
from abc import ABC, abstractmethod

DISPLAY_GRAPH = False

class Func:

    def __init__(self, ctx, op) -> None:
        # instantiation of a function is a context 
        self.ctx = ctx
        self.backward = op

class Tensor(ABC):

    def __init__(self, data, requires_grad: bool = False, depends_on = None) -> None:
        self.data = data.astype(np.float64)
        self.requires_grad = requires_grad
        self.depends_on = depends_on if depends_on else []
        self.grad: 'Tensor' = None
        if self.requires_grad:
            self.zero_grad()

    @property
    def shape(self):
        return self.data.shape

    @property
    def T(self): 
        return self.__class__(self.data.T, self.requires_grad, self.depends_on)

    def zero_grad(self):
        self.grad = self.__class__(np.zeros_like(self.data))

    def backward(self, grad: 'Tensor' = None):
        if grad is None or not grad.data.any():
            grad = self.__class__(np.ones(self.data.shape))
        
        self.grad.data += grad.data
        
        for dep in self.depends_on:
            backward_grad = dep.backward(grad.data)
            # backward_grad.data /= np.linalg.norm(backward_grad.data)
            if DISPLAY_GRAPH:
                print(dep.backward.__name__)
            dep.ctx.backward(self.__class__(backward_grad))

    def __str__(self) -> str:
        return f'{self.data}'

    @abstractmethod
    def __matmul__(self, other: 'Tensor') -> 'Tensor':
        return NotImplemented

    @abstractmethod
    def dot(self, other: 'Tensor') -> 'Tensor':
        return NotImplemented

    @abstractmethod
    def __add__(self, other: 'Tensor') -> 'Tensor':
        return NotImplemented

    @abstractmethod
    def __sub__(self, other: 'Tensor') -> 'Tensor':
        return NotImplemented

    @abstractmethod
    def __mul__(self, k: float):
        return NotImplemented

    @abstractmethod
    def sqrt(self):
        return NotImplemented

    @abstractmethod
    def __pow__(self, k: float):
        return NotImplemented

    @abstractmethod
    def div(self, other):
        return NotImplemented

    @abstractmethod
    def relu(self):
        return NotImplemented

    @abstractmethod
    def softmax(self) -> 'Tensor':
        return NotImplemented

    @abstractmethod
    def mse(self, other) -> 'Tensor':
        return NotImplemented

    @abstractmethod
    def sigmoid(self) -> 'Tensor':
        return NotImplemented

    @abstractmethod
    def mean(self):
        return NotImplemented

    @abstractmethod
    def cross_entropy(self, true_result):
        return NotImplemented

class CPUTensor(Tensor):

    def __init__(self, data, requires_grad: bool = False, depends_on = None) -> None:
        super().__init__(data=data, requires_grad=requires_grad, depends_on=depends_on)

    def __matmul__(self, other: 'Tensor') -> 'CPUTensor':
        return _matmul(self, other)

    def __add__(self, other: 'Tensor') -> 'CPUTensor':
        try:
            return Tensor(self.data + other.data,
                requires_grad=self.requires_grad or other.requires_grad)
        except:
            return Tensor(self.data + other,
                requires_grad=self.requires_grad)

    def __sub__(self, other: 'Tensor') -> 'CPUTensor':
        return CPUTensor(self.data - other.data,
             requires_grad=self.requires_grad or other.requires_grad)

    def __mul__(self, k: float):
        return CPUTensor(self.data * k, requires_grad=self.requires_grad)

    def __pow__(self, k: float):
        return CPUTensor(self.data ** k, requires_grad=self.requires_grad)

    def sqrt(self):
        return CPUTensor(self.data ** .5, requires_grad=self.requires_grad)
    
    def dot(self, other: 'Tensor') -> 'CPUTensor':
        return _matmul(self, other)

    def div(self, other):
        return CPUTensor((other.data ** -1) * self.data, requires_grad=self.requires_grad)

    def relu(self):
        return _relu(self)

    def dot(self, other: 'Tensor') -> 'CPUTensor':
        return self @ other

    def softmax(self) -> 'CPUTensor':
        return _softmax(self)

    def mse(self, other) -> 'CPUTensor':
        return _MSE(self, other)

    def sigmoid(self) -> 'CPUTensor':
        return _sigmoid(self)

    def mean(self):
        return _mean(self)
    
    def cross_entropy(self, true_result):
        return _cross_entropy(self, true_result)

# ---------------------------------------------------------------------------
# -----------------------------CPU OPS REAL----------------------------------
# ---------------------------------------------------------------------------

'''
Important to note that we can put those in other file, but due to circular
imports we need to register those later on via python metaprogramming.

As we will port this to C++, we will probably have several overloadings
of tensor (or simpl a lot of if statements)
'''

def _softmax(t: 'CPUTensor') -> 'CPUTensor':
    max_value = np.max(t.data)
    temp = t.data - max_value
    data = np.exp(temp)
    divide_by = np.sum(data)
    data = data / divide_by
    depends_on = []

    if t.requires_grad:
        def softmax_fn(grad: np.ndarray):
            der = temp * (divide_by - temp)
            return grad * der / (divide_by ** 2) 

        depends_on.append(Func(t, softmax_fn))
    
    return CPUTensor(data, t.requires_grad, depends_on)

def _mean(t: 'CPUTensor') -> 'CPUTensor':
    data = np.sum(t.data) / t.data.shape
    depends_on = []
    if t.requires_grad:
        def mean_fn(grad: np.ndarray):
            return np.ones(t.data.shape) / t.data.shape

        depends_on.append(Func(t, mean_fn))

    return CPUTensor(data, t.requires_grad, depends_on)

def _cross_entropy(t1: 'CPUTensor', t2: 'CPUTensor') -> 'CPUTensor':
    delta = 0.0001
    data = - np.mean((t2.data.T * np.log(t1.data+delta) + (1-t2.data).T * np.log(1-t1.data+delta)))
    depends_on = []
    if t1.requires_grad:
        def cross_entropy_fn(grad: np.ndarray):
            temp = t2.data.T / (t1.data+delta) - (1 - t2.data).T / (1 - t1.data + delta)
            return -data*grad*temp
        
        depends_on.append(Func(t1, cross_entropy_fn))
    return CPUTensor(data, t1.requires_grad, depends_on)

def _MSE(t1: 'CPUTensor', t2: 'CPUTensor') -> 'CPUTensor':
    temp = t2.data - t1.data
    data = np.mean(temp * temp)
    depends_on = []

    if t1.requires_grad:
        def MSE_fn(grad: np.ndarray):
            return -grad*temp/t1.shape[1]

        depends_on.append(Func(t1, MSE_fn))
    
    return CPUTensor(data, t1.requires_grad, depends_on)


def _sigmoid(t: 'CPUTensor') -> 'CPUTensor':
    data = np.zeros(t.data.shape)
    for i in range(t.data.shape[1]):
        val = t.data[0, i]
        if val < 0:
            data[0, i] = np.exp(val) / (1 + np.exp(val))
        else:
            data[0, i] = 1 / (1 + np.exp(-val))
    depends_on = []

    if t.requires_grad:
        def sigmoid_fn(grad: np.ndarray):
            return grad * data * (1 - data)
        
        depends_on.append(Func(t, sigmoid_fn))
    
    return CPUTensor(data, t.requires_grad, depends_on)

def _matmul(t1: 'CPUTensor', t2: 'CPUTensor') -> 'CPUTensor':
    data = t1.data @ t2.data
    if data.shape == ():
        data = np.array([data]).reshape((1, 1))
    depends_on = []

    if t1.requires_grad:
        def matmul_fn1(grad: np.ndarray) -> np.ndarray:
            return grad @ t2.data.T

        depends_on.append(Func(t1, matmul_fn1))

    if t2.requires_grad:
        def matmul_fn2(grad: np.ndarray) -> np.ndarray:
            return t1.data.T @ grad
        depends_on.append(Func(t2, matmul_fn2))

    return CPUTensor(data, t1.requires_grad or t2.requires_grad, depends_on)

def _relu(t: 'CPUTensor') -> 'CPUTensor':
    data = np.maximum(0, t.data)
    depends_on = []

    if t.requires_grad:
        def relu_fn(grad: np.ndarray):
            tmp = grad * (data >= 0)
            return tmp
        
        depends_on.append(Func(t, relu_fn))
    
    return CPUTensor(data, t.requires_grad, depends_on)
