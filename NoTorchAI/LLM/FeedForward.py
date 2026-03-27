import numpy as np

from NoTorchAI.Activation.ReLu import ReLu
from NoTorchAI.Gradients.ABSGradient import ABSGradient
from NoTorchAI.Layers.LinearLayer import Linear
from NoTorchAI.Neuron import Neuron


class FeedForward(Neuron):
    def __init__(self, d_model: int, gradient: ABSGradient):
        # Reduced from 4x to 3x expansion to save memory
        intermediate_dim = max(3 * d_model, (d_model * 3) // 2)  # At least 3x, rounded nicely
        self.linear1 = Linear(d_model, intermediate_dim)
        self.relu = ReLu()
        self.linear2 = Linear(intermediate_dim, d_model)

        self.gradient = gradient

    def _change_weights(self) -> None:
        self.gradient.step(self.linear1)
        self.gradient.step(self.linear2)

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = self.linear1.forward(x)
        x = self.relu.forward(x)
        x = self.linear2.forward(x)
        return x
    
    def backward(self, incoming_grad: np.ndarray) -> np.ndarray:
        grad = self.linear2.backward(incoming_grad)
        grad = self.relu.backward(grad)
        grad = self.linear1.backward(grad)

        self._change_weights()
        return grad
