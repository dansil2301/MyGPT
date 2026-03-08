import numpy as np

from NoTorchAI.ActivationFunc import ReLu
from NoTorchAI.Gradients.ABSGradient import ABSGradient
from NoTorchAI.Layers.LinearLayer import Linear


class FeedForward:
    def __init__(self, d_model: int, gradient: ABSGradient):
        self.linear1 = Linear(d_model, 4 * d_model)
        self.relu = ReLu()
        self.linear2 = Linear(4 * d_model, d_model)

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
