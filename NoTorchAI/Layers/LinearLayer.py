import numpy as np
from numpy.random import default_rng


class Linear:
    def __init__(self, in_feature: int, out_feature: int):
        self.weights = default_rng(42).random((in_feature, out_feature))
        self.bias = default_rng(43).random(out_feature)

        self.input = None
        self.d_weights = None
        self.d_bias = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        return x @ self.weights + self.bias

    def backward(self, icoming_grad: np.ndarray) -> np.ndarray:
        self.d_weights = np.sum(self.input.transpose(-2, -1) @ icoming_grad, axis=0)
        self.d_bias = np.sum(icoming_grad, axis=0)

        outgoing_grad = icoming_grad @ self.weights.T
        return outgoing_grad
