import numpy as np

from NoTorchAI.Activation.ActivationFunc import ActivationFunc
from NoTorchAI.Neuron import Neuron
from NoTorchAI.Utils.MatrixOperations import MatrixOperations as mo


class Softmax(ActivationFunc, Neuron):
    def __init__(self):
        self.output = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        x_stable = x - mo.max(x, axis=-1, keepdims=True)

        exp_x = mo.exp(x_stable)
        self.output = exp_x / mo.sum(exp_x, axis=-1, keepdims=True)

        return self.output

    def backward(self, incoming_grad: np.ndarray) -> np.ndarray:
        dot = mo.sum(self.output * incoming_grad, axis=-1, keepdims=True)
        outgoing_grad = self.output * (incoming_grad - dot)

        self.output = None
        return outgoing_grad
