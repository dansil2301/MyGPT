import numpy as np

from NoTorchAI.Gradients.ABSGradient import ABSGradient
from NoTorchAI.LLM.FeedForward import FeedForward
from NoTorchAI.LLM.SelfAttention import SelfAttention
from NoTorchAI.Layers.NormLayer import NormLayer


class Block:
    def __init__(self, d_model: int, gradient: ABSGradient):
        super().__init__()
        self.linear1 = NormLayer(d_model)
        self.attention = SelfAttention(d_model, gradient)
        self.linear2 = NormLayer(d_model)
        self.ff = FeedForward(d_model, gradient)

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