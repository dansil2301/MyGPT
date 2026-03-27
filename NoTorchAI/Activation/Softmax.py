import numpy as np

from NoTorchAI.Activation.ActivationFunc import ActivationFunc
from NoTorchAI.Neuron import Neuron
from NoTorchAI.Utils.Matrix import Matrix


class Softmax(ActivationFunc, Neuron):
    def __init__(self):
        self.output = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        x_stable = x - Matrix.max(x, axis=-1, keepdims=True)

        exp_x = Matrix.exp(x_stable)
        self.output = exp_x / Matrix.sum(exp_x, axis=-1, keepdims=True)

        return self.output

    def backward(self, incoming_grad: np.ndarray) -> np.ndarray:
        dot = Matrix.sum(self.output * incoming_grad, axis=-1, keepdims=True)
        outgoing_grad = self.output * (incoming_grad - dot)

        self.output = None
        return outgoing_grad
