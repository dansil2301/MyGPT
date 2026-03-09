import numpy as np
from numpy.random import default_rng

from NoTorchAI.Layers.ABSLayer import ABSLayer


class NormLayer(ABSLayer):
    def __init__(self, d_model: int):
        self.weights = np.ones(d_model, dtype=np.float32)
        self.bias = np.zeros(d_model, dtype=np.float32)
        self.epsilon = 1e-5

        self.x_hat = None
        self.mean = None
        self.variance = None

        self.d_bias = None
        self.d_weights = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.mean = np.mean(x, axis=-1, keepdims=True)
        self.variance = x.var(axis=-1, keepdims=True) + self.epsilon
        self.x_hat = (x - self.mean) / (self.variance ** 0.5)
        return self.weights * self.x_hat + self.bias

    def backward(self, incoming_grad: np.ndarray) -> np.ndarray:
        self.d_bias = np.sum(incoming_grad, axis=(0, 1))
        self.d_weights = np.sum(incoming_grad * self.x_hat, axis=(0, 1))

        dx_hat = incoming_grad * self.weights
        N = self.x_hat.shape[-1]
        std_inv = 1.0 / np.sqrt(self.variance)

        sum_dxhat = np.sum(dx_hat, axis=-1, keepdims=True)
        sum_dxhat_xhat = np.sum(dx_hat * self.x_hat, axis=-1, keepdims=True)

        dx = (1.0 / N) * std_inv * (
            N * dx_hat
            - sum_dxhat
            - self.x_hat * sum_dxhat_xhat
        )

        # Memoty optimization
        self.input = None
        self.x_hat = None
        self.mean = None
        self.variance = None

        return dx
