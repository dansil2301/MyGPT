from abc import ABC, abstractmethod

from NoTorchAI.Embedding import Embedding
from NoTorchAI.Layers.ABSLayer import ABSLayer


class ABSGradient(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def step(self, layer: ABSLayer | Embedding) -> None:
        pass