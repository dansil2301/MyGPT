import numpy as np
from NoTorchAI.Neuron import Neuron
from NoTorchAI.Activation.Softmax import Softmax
from NoTorchAI.Layers.LinearLayer import Linear
from NoTorchAI.Gradients.ABSGradient import ABSGradient


class SelfAttention(Neuron):
    def __init__(self, d_model: int, gradient_technic: ABSGradient, device: str = "cpu", quant: int = 16):
        super().__init__(device)
        self.gradient_technic = gradient_technic

        self.query = Linear(d_model, d_model, device, quant)
        self.key = Linear(d_model, d_model, device, quant)
        self.value = Linear(d_model, d_model, device, quant)

        self.q_output = None
        self.k_output = None
        self.v_output = None

        self.scores = None
        self.softmax = Softmax(device)

        self.attention = None
        self.output = None

    def _forward_mask_fill(self, tensor: np.ndarray):
        B, T, _ = tensor.shape

        mask = self.xp.triu(self.xp.ones((T, T), dtype=bool), k=1)
        mask = mask[None, :, :]
        tensor = self.xp.where(mask, -self.xp.inf, tensor)
        return tensor
    
    def _backward_mask_fill(self, grad: np.ndarray):
        B, T, _ = grad.shape

        mask = self.xp.triu(self.xp.ones((T, T), dtype=bool), k=1)[None, :, :]
        return self.xp.where(mask, 0, grad)
    
    def _change_weights(self):
        self.gradient_technic.step(self.query)
        self.gradient_technic.step(self.key)
        self.gradient_technic.step(self.value)

    def forward(self, x: np.ndarray) -> np.ndarray:
        B, T, E = x.shape

        self.q_output = self.query.forward(x)
        self.k_output = self.key.forward(x)
        self.v_output = self.value.forward(x)

        self.scores = (self.q_output @ self.k_output.transpose(0, 2, 1)) / (E ** 0.5)

        masked_scores = self._forward_mask_fill(self.scores)
        self.attention = self.softmax.forward(masked_scores)

        return self.xp.matmul(self.attention, self.v_output)
    
    def backward(self, incoming_grad: np.ndarray) -> np.ndarray:
        B, T, E = incoming_grad.shape

        dvalue = self.attention.transpose(0, 2, 1) @ incoming_grad
        dattention = incoming_grad @ self.v_output.transpose(0, 2, 1)

        dsoftmax = self.softmax.backward(dattention)
        dsoftmax = self._backward_mask_fill(dsoftmax)

        dscaling = dsoftmax / (E ** 0.5)

        dquery = dscaling @ self.k_output
        dkey = dscaling.transpose(0, 2, 1) @ self.q_output

        dx_v = self.value.backward(dvalue)
        dx_q = self.query.backward(dquery)
        dx_k = self.key.backward(dkey)

        self._change_weights()
        return dx_v + dx_q + dx_k
