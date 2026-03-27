import numpy as np

from NoTorchAI.Layers.ABSLayer import ABSLayer
from NoTorchAI.Neuron import Neuron
from NoTorchAI.Utils.Matrix import Matrix


class NormLayer(ABSLayer, Neuron):
    def __init__(self, d_model: int):
        self.weights = Matrix.ones(d_model)
        self.bias = Matrix.zeros(d_model)
        self.epsilon = 1e-5

        self.x_hat = None
        self.mean = None
        self.variance = None

        self.d_bias = None
        self.d_weights = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.mean = Matrix.mean(x, axis=-1, keepdims=True)
        self.variance = Matrix.var(x, axis=-1, keepdims=True) + self.epsilon
        self.x_hat = (x - self.mean) / (self.variance ** 0.5)
        return self.weights * self.x_hat + self.bias

    def backward(self, incoming_grad: np.ndarray) -> np.ndarray:
        self.d_bias = Matrix.sum(incoming_grad, axis=(0, 1))
        self.d_weights = Matrix.sum(incoming_grad * self.x_hat, axis=(0, 1))

        dx_hat = incoming_grad * self.weights
        N = self.x_hat.shape[-1]
        std_inv = 1.0 / Matrix.sqrt(self.variance)
        
        sum_dxhat = Matrix.sum(dx_hat, axis=-1, keepdims=True)
        sum_dxhat_xhat = Matrix.sum(dx_hat * self.x_hat, axis=-1, keepdims=True)

        dx = (1.0 / N) * std_inv * (
            N * dx_hat
            - sum_dxhat
            - self.x_hat * sum_dxhat_xhat
        )

        # Memory optimization
        self.input = None
        self.x_hat = None
        self.mean = None
        self.variance = None

        return dx
