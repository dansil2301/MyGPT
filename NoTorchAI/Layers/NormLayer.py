import numpy as np
from numpy.random import default_rng


class Normalization:
    def __init__(self, d_model: int):
        self.gamma = default_rng(43).random(d_model)
        self.beta = default_rng(42).random(d_model)
        self.epsilon = 1e-8

        self.x_hat = None
        self.mean = None
        self.variance = None

        self.dbeta = None
        self.dgamma = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.mean = np.mean(x, axis=-1, keepdims=True)
        self.variance = x.var(axis=-1, keepdims=True) + self.epsilon
        self.x_hat = (x - self.mean) / (self.variance ** 0.5)
        return self.gamma * self.x_hat + self.beta

    def backward(self, incoming_grad: np.ndarray) -> np.ndarray:
        self.dbeta = np.sum(incoming_grad, axis=(0, 1))
        self.dgamma = np.sum(incoming_grad * self.x_hat, axis=(0, 1))

        dx_hat = incoming_grad * self.gamma
        N = self.x_hat.shape[-1]
        std_inv = 1.0 / np.sqrt(self.variance)

        sum_dxhat = np.sum(dx_hat, axis=-1, keepdims=True)
        sum_dxhat_xhat = np.sum(dx_hat * self.x_hat, axis=-1, keepdims=True)

        dx = (1.0 / N) * std_inv * (
            N * dx_hat
            - sum_dxhat
            - self.x_hat * sum_dxhat_xhat
        )

        return dx
