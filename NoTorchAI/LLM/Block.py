import numpy as np

from NoTorchAI.Gradients.ABSGradient import ABSGradient
from NoTorchAI.LLM.FeedForward import FeedForward
from NoTorchAI.LLM.MultiHead import MultiHead
from NoTorchAI.LLM.SelfAttention import SelfAttention
from NoTorchAI.Layers.NormLayer import NormLayer
from NoTorchAI.Neuron import Neuron


class Block(Neuron):
    def __init__(self, d_model: int, n_heads: int, gradient: ABSGradient, device: str = "cpu", quant: int = 16):
        super().__init__(device)
        self.linear1 = NormLayer(d_model, device, quant)
        self.attention = MultiHead(d_model, n_heads, gradient, device, quant)
        self.linear2 = NormLayer(d_model, device, quant)
        self.ff = FeedForward(d_model, gradient, device, quant)

        self.gradient = gradient

    def _change_weights(self) -> None:
        self.gradient.step(self.linear1)
        self.gradient.step(self.linear2)

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = x + self.attention.forward(self.linear1.forward(x))
        x = x + self.ff.forward(self.linear2.forward(x))
        return x
    
    def backward(self, grad: np.ndarray) -> np.ndarray:
        d_ff_out = self.ff.backward(grad)
        d_norm2_out = self.linear2.backward(d_ff_out)
        grad_after_ff = grad + d_norm2_out

        d_attn_out = self.attention.backward(grad_after_ff)
        d_norm1_out = self.linear1.backward(d_attn_out)
        d_x = grad_after_ff + d_norm1_out

        self._change_weights()
        return d_x