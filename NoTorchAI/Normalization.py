import numpy as np
from numpy.random import default_rng


class Normalization:
    def __init__(self, d_model: int):
        self.gamma = default_rng(43).random(d_model)
        self.beta = default_rng(42).random(d_model)
        self.epsilon = 1e-8

        self.output = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        mean = np.mean(x, axis=-1, keepdims=True)
        varience = x.var(axis=-1, keepdims=True) + self.epsilon
        self.output = self.gamma * (x - mean) / (varience ** 0.5) + self.beta
        return self.output

    def backward(self, incoming_grad: np.ndarray) -> np.ndarray:
        dbeta = np.sum(incoming_grad, axis=0)
        dgamma = np.sum(incoming_grad * self.output, axis=0)

        dscaling = self.gamma * incoming_grad
        # todo: finish backward propagation for normalization