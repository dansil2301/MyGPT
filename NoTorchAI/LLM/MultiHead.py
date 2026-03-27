from types import NoneType

import numpy as np

from NoTorchAI.Gradients.ABSGradient import ABSGradient
from NoTorchAI.LLM.SelfAttention import SelfAttention
from NoTorchAI.Layers.LinearLayer import Linear
from NoTorchAI.Neuron import Neuron
from NoTorchAI.Utils.Matrix import Matrix


class MultiHead(Neuron):
    def __init__(self, d_model: int, n_heads: int, gradient_technic: ABSGradient):
        self.d_head = d_model // n_heads
        if self.d_head * n_heads != d_model:
            raise ValueError("Can't devide head dimensions equally")

        self.heads = [SelfAttention(self.d_head, gradient_technic) for _ in range(n_heads)]
        self.linear1 = Linear(d_model, d_model)

        self.gradient = gradient_technic

    def _change_weights(self) -> None:
        self.gradient.step(self.linear1)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        # B, T, E
        results_heads = []
        for i, head in enumerate(self.heads):
            x_head = x[:, :, self.d_head * i : self.d_head * (i + 1)]
            results_heads.append(head.forward(x_head))

        concatinated = Matrix.concatenate(results_heads, axis=2)

        output = self.linear1.forward(concatinated)
        return output
        

    def backward(self, grad: np.ndarray) -> np.ndarray:
        grad = self.linear1.backward(grad)
        
        grads_heads = []
        for i, head in enumerate(self.heads):
            grad_head = grad[:, :, self.d_head * i : self.d_head * (i + 1)]
            grads_heads.append(head.backward(grad_head))

        grad = Matrix.concatenate(grads_heads, axis=2)

        self._change_weights()
        return grad
