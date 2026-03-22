from abc import ABC, abstractmethod

from NoTorchAI.Layers.ABSLayer import ABSLayer
from NoTorchAI.Layers.Embedding import Embedding
from NoTorchAI.Neuron import Neuron


class ABSGradient(ABC, Neuron):
    def __init__(self, device: str):
        super().__init__(device)

    @abstractmethod
    def step(self, layer: ABSLayer | Embedding) -> None:
        pass