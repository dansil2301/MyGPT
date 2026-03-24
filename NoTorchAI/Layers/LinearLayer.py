import numpy as np
from numpy.random import default_rng

from NoTorchAI.Layers.ABSLayer import ABSLayer
from NoTorchAI.Neuron import Neuron


class Linear(ABSLayer, Neuron):
    def __init__(self, in_feature: int, out_feature: int, device: str = "cpu", quant: int = 16):
        super().__init__(device, quant)
        
        std = (2.0 / in_feature) ** 0.5
        self.weights = self.xp.random.normal(0, std, (in_feature, out_feature)).astype(self.quant)
        self.bias =  self.xp.zeros(out_feature, dtype=self.quant)

        self.input = None
        self.d_weights = None
        self.d_bias = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        return x @ self.weights + self.bias

    def backward(self, incoming_grad: np.ndarray) -> np.ndarray:
        self.d_weights = self.xp.sum(self.input.transpose(0, 2, 1) @ incoming_grad, axis=0)
        self.d_bias = self.xp.sum(incoming_grad, axis=(0, 1))

        outgoing_grad = incoming_grad @ self.weights.T

        # Memoty optimization
        self.input = None
        
        return outgoing_grad
