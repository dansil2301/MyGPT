import numpy as np
from numpy.random import default_rng


class Linear:
    def __init__(self, in_feature: int, out_feature: int):
        self.weights = default_rng(2).random((in_feature, out_feature))
        self.bias = default_rng(1).random(out_feature)

        self.input = None
        self.d_weights = None
        self.d_bias = None

    def forward(self, x: np.array):
        self.input = x
        return x @ self.weights + self.bias

    def backward(self, icoming_grad: np.ndarray) -> np.array:
        self.d_weights = self.input.T @ icoming_grad
        self.d_bias = np.sum(icoming_grad, axis=0)

        outgoing_grad = icoming_grad @ self.weights.T
        return outgoing_grad
