import numpy as np
from numpy.random import default_rng

from NoTorchAI.Layers.ABSLayer import ABSLayer
from NoTorchAI.Neuron import Neuron
from NoTorchAI.Utils.Matrix import Matrix


class Linear(ABSLayer, Neuron):
    def __init__(self, in_feature: int, out_feature: int):
        self.weights = Matrix.he_normal_init(in_feature, out_feature)
        self.bias = Matrix.zeros(out_feature)

        self.input = None
        self.d_weights = None
        self.d_bias = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        return x @ self.weights + self.bias

    def backward(self, incoming_grad: np.ndarray) -> np.ndarray:
        self.d_weights = Matrix.sum(self.input.transpose(0, 2, 1) @ incoming_grad, axis=0)
        self.d_bias = Matrix.sum(incoming_grad, axis=(0, 1))
        outgoing_grad = incoming_grad @ self.weights.T

        # Memory optimization
        self.input = None
        
        return outgoing_grad
