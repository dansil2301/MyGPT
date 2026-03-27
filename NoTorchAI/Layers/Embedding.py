import numpy as np

from NoTorchAI.Layers.ABSLayer import ABSLayer
from NoTorchAI.Neuron import Neuron
from NoTorchAI.Utils.Matrix import Matrix


class Embedding(ABSLayer, Neuron):
    def __init__(self, num_embedding: int, d_model: int):
        self.embeddings = Matrix.xavier_init(num_embedding, d_model)

        self.input_indices = None
        self.d_embeddings = None

    def forward(self, indices: np.ndarray) -> np.ndarray:
        self.input_indices = indices
        return self.embeddings[indices]

    def backward(self, incoming_grad: np.ndarray) -> None:
        self.d_embeddings = Matrix.zeros_like(self.embeddings)

        Matrix.add_at(self.d_embeddings, self.input_indices, incoming_grad)
        return None
