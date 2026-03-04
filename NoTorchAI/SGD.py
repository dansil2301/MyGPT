from abc import ABC, abstractmethod

from NoTorchAI.Embedding import Embedding
from NoTorchAI.Layers.ABSLayer import ABSLayer


class ABSGradient(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def step(self, layer: ABSLayer) -> None:
        pass


class SGD(ABSGradient):
    def __init__(self, lr: float):
        self.lr = lr

    def step(self, layer: ABSLayer | Embedding) -> None:
        if type(layer) == ABSLayer:
            layer.weights -= self.lr * layer.d_weights
            layer.bias -= self.lr * layer.d_bias
        elif type(layer) == Embedding:
            for token_id in layer.d_embeddings:
                layer.embeddings[token_id] - layer.d_embeddings[token_id]
