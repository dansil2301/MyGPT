from abc import ABC, abstractmethod
import numpy as np

from NoTorchAI.Neuron import Neuron


class ABSLayer(ABC, Neuron):
    def __init__(self, device: str, quant: int):
        super().__init__(device)
        self.quant = self._set_quant(quant)

    def _set_quant(self, quant: int):
        quant_obj = None
        if quant == 64:
            quant_obj = self.xp.float64
        elif quant == 32:
            quant_obj = self.xp.float32
        elif quant == 16:
            quant_obj = self.xp.float16
        else:
            raise ValueError("There is no such quant available")
        return quant_obj

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, icoming_grad: np.ndarray) -> np.ndarray | None:
        pass
