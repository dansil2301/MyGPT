import numpy as np

from NoTorchAI.Embedding import Embedding
from NoTorchAI.Gradients.ABSGradient import ABSGradient
from NoTorchAI.Neuron import Neuron


class EmbeddingInitial(Neuron):
    def __init__(self, vocab_size: int, d_model: int, block_size: int, gradient: ABSGradient):
        self.token_embedding = Embedding(vocab_size, d_model)
        self.position_embedding = Embedding(block_size, d_model)
        self.gradient = gradient

    def _change_weights(self) -> None:
        self.gradient.step(self.token_embedding)
        self.gradient.step(self.position_embedding)

    def forward(self, x: np.ndarray) -> np.ndarray:
        B, T = x.shape  

        token_emb = self.token_embedding.forward(x)
        pos_ids = np.arange(T)
        pos_emb = self.position_embedding.forward(pos_ids)

        x = token_emb + pos_emb  

        return x
    
    def backward(self, incoming_grad: np.ndarray) -> None:
        d_token = incoming_grad
        d_pos = np.sum(incoming_grad, axis=0)
        
        self.token_embedding.backward(d_token)
        self.position_embedding.backward(d_pos)

        self._change_weights()
