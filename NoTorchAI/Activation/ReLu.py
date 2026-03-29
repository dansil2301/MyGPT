import numpy as np

from NoTorchAI.Activation.ActivationFunc import ActivationFunc
from NoTorchAI.Neuron import Neuron
from NoTorchAI.Utils.MatrixOperations import MatrixOperations as mo


class ReLu(ActivationFunc, Neuron):
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x > 0)  # store mask directly
        return mo.where(self.mask, x, 0)

    def backward(self, incoming_grad):
        return self.mask * incoming_grad
