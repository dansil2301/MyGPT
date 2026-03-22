import numpy as np

from NoTorchAI.Layers.ABSLayer import ABSLayer


class NormLayer(ABSLayer):
    def __init__(self, d_model: int, device: str = "cpu", quant: int = 16):
        super().__init__(device, quant)
        self.weights = self.xp.ones(d_model, dtype=self.quant)
        self.bias = self.xp.zeros(d_model, dtype=self.quant)
        self.epsilon = 1e-5

        self.x_hat = None
        self.mean = None
        self.variance = None

        self.d_bias = None
        self.d_weights = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.mean = self.xp.mean(x, axis=-1, keepdims=True)
        self.variance = x.var(axis=-1, keepdims=True) + self.epsilon
        self.x_hat = (x - self.mean) / (self.variance ** 0.5)
        return self.weights * self.x_hat + self.bias

    def backward(self, incoming_grad: np.ndarray) -> np.ndarray:
        self.d_bias = self.xp.sum(incoming_grad, axis=(0, 1))
        self.d_weights = self.xp.sum(incoming_grad * self.x_hat, axis=(0, 1))

        dx_hat = incoming_grad * self.weights
        N = self.x_hat.shape[-1]
        std_inv = 1.0 / self.xp.sqrt(self.variance)

        sum_dxhat = self.xp.sum(dx_hat, axis=-1, keepdims=True)
        sum_dxhat_xhat = self.xp.sum(dx_hat * self.x_hat, axis=-1, keepdims=True)

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
