import numpy as np
from numpy.random import default_rng

from NoTorchAI.Neuron import Neuron


class Embedding(Neuron):
    def __init__(self, num_embedding: int, d_model: int, device: str = "cpu"):
        super().__init__(device)
        self.embeddings = self.xp.random.normal(0, 0.02, (num_embedding, d_model)).astype(self.xp.float32)

        self.input_indices = None
        self.d_embeddings = None

    def forward(self, indices: np.ndarray) -> np.ndarray:
        self.input_indices = indices
        return self.embeddings[indices]

    def backward(self, incoming_grad: np.ndarray) -> None:
        self.d_embeddings = self.xp.zeros_like(self.embeddings)

        self.xp.add.at(self.d_embeddings, self.input_indices, incoming_grad)
