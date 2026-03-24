import numpy as np

from NoTorchAI.Activation.ActivationFunc import ActivationFunc
from NoTorchAI.Neuron import Neuron


class Softmax(ActivationFunc, Neuron):
    def __init__(self, device: str = "cpu"):
        super().__init__(device)
        self.output = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        x_stable = x - self.xp.max(x, axis=-1, keepdims=True)

        exp_x = self.xp.exp(x_stable)
        self.output = exp_x / self.xp.sum(exp_x, axis=-1, keepdims=True)

        return self.output

    def backward(self, incoming_grad: np.ndarray) -> np.ndarray:
        dot = self.xp.sum(self.output * incoming_grad, axis=-1, keepdims=True)
        outgoing_grad = self.output * (incoming_grad - dot)

        self.output = None
        return outgoing_grad
