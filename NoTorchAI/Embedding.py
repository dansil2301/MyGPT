import numpy as np
from numpy.random import default_rng


class Embedding:
    def __init__(self, num_embedding: int, d_model: int):
        self.embeddings = default_rng(45).random((num_embedding, d_model))

        self.input_indices = None
        self.d_embeddings = None

    def forward(self, indices: np.ndarray) -> np.ndarray:
        self.input_indices = indices
        return self.embeddings[indices]

    def backward(self, incoming_grad: np.ndarray) -> None:
        B, T, E = incoming_grad.shape

        self.d_embeddings = np.zeros_like(self.embeddings)

        for b in range(B):
            for t in range(T):
                token_id = self.input_indices[b, t]
                self.d_embeddings[token_id] += incoming_grad[b, t]
