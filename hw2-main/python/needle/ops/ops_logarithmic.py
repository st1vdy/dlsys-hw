from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api

class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        max_Z = array_api.max(Z, 1, keepdims=True)
        exp_Z = array_api.exp(Z - max_Z)
        lse_Z = array_api.log(array_api.sum(exp_Z, 1, keepdims=True)) + array_api.max(Z, 1, keepdims=True)
        return Z - lse_Z
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        Z = node.inputs[0]
        softmax_Z = exp(node)
        out_sum = summation(out_grad, 1).realize_cached_data()
        grad_Z = softmax_Z.realize_cached_data()
        for i in range(grad_Z.shape[0]):
            grad_Z[i, :] *= out_sum[i]
        res = out_grad - Tensor(grad_Z)
        return res
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        max_Z = array_api.max(Z, self.axes, keepdims=True)
        exp_Z = array_api.exp(Z - max_Z)
        lse_Z = array_api.log(array_api.sum(exp_Z, axis=self.axes)) + array_api.max(Z, axis=self.axes)
        return lse_Z
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        Z = node.inputs[0]
        lse_z = node
        # lse_z = logsumexp(Z, self.axes)  # also correct
        if self.axes:
            new_shape = [1] * len(Z.shape)
            for i in range(len(Z.shape)):
                if i not in self.axes:
                    new_shape[i] = Z.shape[i]
            out_grad = reshape(out_grad, new_shape)
            lse_z = reshape(lse_z, new_shape)
        return out_grad * exp(Z - lse_z)
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

