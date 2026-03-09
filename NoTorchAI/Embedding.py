import numpy as np
from numpy.random import default_rng


class Embedding:
    def __init__(self, num_embedding: int, d_model: int):
        rng = default_rng()

        self.embeddings = rng.normal(0, 0.02, (num_embedding, d_model)).astype(np.float32)

        self.input_indices = None
        self.d_embeddings = None

        self.grad = None

    def forward(self, indices: np.ndarray) -> np.ndarray:
        self.input_indices = indices
        return self.embeddings[indices]

    def backward(self, incoming_grad: np.ndarray) -> None:
        self.grad = incoming_grad
        self.d_embeddings = np.zeros_like(self.embeddings)

        np.add.at(self.d_embeddings, self.input_indices, incoming_grad)
