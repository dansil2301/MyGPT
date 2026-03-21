from abc import ABC, abstractmethod
import numpy as np

from NoTorchAI.Neuron import Neuron


class ABSLayer(ABC, Neuron):
    def __init__(self, device: str):
        super().__init__(device)
        self.weights = None
        self.bias = None

        self.d_weights = None
        self.d_bias = None

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, icoming_grad: np.ndarray) -> np.ndarray:
        pass
