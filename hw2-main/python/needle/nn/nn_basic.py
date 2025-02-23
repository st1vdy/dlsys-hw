"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        # y = X @ A^T + b
        self.weight = Parameter(init.kaiming_uniform(fan_in=in_features, fan_out=out_features, dtype="float32", requires_grad=True))
        self.bias = Parameter(init.kaiming_uniform(fan_in=out_features, fan_out=1, dtype="float32", requires_grad=True).transpose()) if bias else None
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        y = X @ self.weight
        if self.bias is not None:
            y += self.bias.broadcast_to(y.shape)
        return y
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        return ops.reshape(X, (X.shape[0], -1))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION

class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for module in self.modules:
            x = module(x)
        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        lse = ops.logsumexp(logits, axes=(1, )).sum()
        zy = (logits * init.one_hot(logits.shape[1], y)).sum()
        loss = lse - zy
        return loss / logits.shape[0]
        ### END YOUR SOLUTION



class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(1, dim, requires_grad=True))
        self.bias = Parameter(init.zeros(1, dim, requires_grad=True))
        self.running_mean = init.zeros(dim)  # 不是parameter 不要加parameter!
        self.running_var = init.ones(dim)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            mean = ops.summation(x, (0,)) / x.shape[0]
            # self.running_xxx 需要加detach() 不然无法通过test_optim_adam_z_memory_check_1()
            self.running_mean = self.momentum * mean.detach() + (1 - self.momentum) * self.running_mean.detach()
            mean = ops.broadcast_to(mean, x.shape)
            var = ops.summation((x - mean) ** 2, (0,)) / x.shape[0]
            self.running_var = self.momentum * var.detach() + (1 - self.momentum) * self.running_var.detach()
            std = ops.power_scalar(var + self.eps, 0.5)
            std = ops.broadcast_to(std, x.shape)
            y = (x - mean) / std
            bn_x = ops.broadcast_to(self.weight, x.shape) * y + ops.broadcast_to(self.bias, x.shape)
            return bn_x
        else:
            mean = ops.broadcast_to(self.running_mean, x.shape)
            std = ops.power_scalar(ops.broadcast_to(self.running_var + self.eps, x.shape), 0.5)
            y = (x - mean) / std
            bn_x = ops.broadcast_to(self.weight, x.shape) * y + ops.broadcast_to(self.bias, x.shape)
            return bn_x
        ### END YOUR SOLUTION


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(1, dim, requires_grad=True))
        self.bias = Parameter(init.zeros(1, dim, requires_grad=True))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        mean = ops.summation(x, (1,)) / x.shape[1]
        mean = ops.reshape(mean, (mean.shape[0], 1))
        mean = ops.broadcast_to(mean, x.shape)
        var = ops.power_scalar(x - mean, 2) + self.eps
        var = ops.summation(var, (1,)) / x.shape[1]
        var = ops.reshape(var, (var.shape[0], 1))
        std = ops.power_scalar(var, 0.5)
        std = ops.broadcast_to(std, x.shape)
        y = (x - mean) / std
        ln_x = ops.broadcast_to(self.weight, x.shape) * y + ops.broadcast_to(self.bias, x.shape)
        return ln_x
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            mask = init.randb(x.shape[0], x.shape[1], p=1 - self.p)
            x *= mask
            x /= (1 - self.p)
        return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return x + self.fn(x)
        ### END YOUR SOLUTION
