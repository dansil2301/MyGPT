from abc import ABC, abstractmethod

import numpy as np

from NoTorchAI.Neuron import Neuron


class ActivationFunc(ABC, Neuron):
    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, icoming_grad: np.ndarray) -> np.ndarray:
        pass
