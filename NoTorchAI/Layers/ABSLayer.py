from abc import ABC, abstractmethod
import numpy as np


class ABSLayer(ABC):
    def __init__(self):
        self.weights = None
        self.bias = None

        self.d_weights = None
        self.d_bias = None

        self.grad = None

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, icoming_grad: np.ndarray) -> np.ndarray:
        pass
