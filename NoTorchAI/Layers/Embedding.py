import numpy as np

from NoTorchAI.Layers.ABSLayer import ABSLayer
from NoTorchAI.Neuron import Neuron


class Embedding(ABSLayer, Neuron):
    def __init__(self, num_embedding: int, d_model: int, device: str = "cpu", quant: int = 16):
        super().__init__(device, quant)
        
        std = 1.0 / (d_model ** 0.5)
        self.embeddings = self.xp.random.normal(0, std, (num_embedding, d_model)).astype(self.quant)

        self.input_indices = None
        self.d_embeddings = None

    def forward(self, indices: np.ndarray) -> np.ndarray:
        self.input_indices = indices
        return self.embeddings[indices]

    def backward(self, incoming_grad: np.ndarray) -> None:
        self.d_embeddings = self.xp.zeros_like(self.embeddings)

        self.xp.add.at(self.d_embeddings, self.input_indices, incoming_grad)
        return None  # Embeddings don't have upstream inputs
