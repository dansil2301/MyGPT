from abc import ABC, abstractmethod

import numpy as np


class ActivationFunc(ABC):
    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, icoming_grad: np.ndarray) -> np.ndarray:
        pass


class Softmax(ActivationFunc):
    def __init__(self):
        self.output = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        x_stable = x - np.max(x, axis=-1, keepdims=True)

        exp_x = np.exp(x_stable)
        self.output = exp_x / np.sum(exp_x, axis=-1, keepdims=True)

        return self.output

    def backward(self, incoming_grad: np.ndarray) -> np.ndarray:
        dot = np.sum(self.output * incoming_grad, axis=-1, keepdims=True)
        outgoing_grad = self.output * (incoming_grad - dot)

        return outgoing_grad
    

class ReLu(ActivationFunc):
    def __init__(self):
        self.output = None

    def forward(self, x):
        self.output = np.maximum(x, 0)
        return self.output
    
    def backward(self, icoming_grad):
        mask = (self.output > 0)
        return mask * icoming_grad
