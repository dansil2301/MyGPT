import numpy as np
from NoTorchAI.ActivationFunc import Softmax
from NoTorchAI.Layers.LinearLayer import Linear


class SelfAttention:
    def __init__(self, d_model: int):
        self.query = Linear(d_model, d_model)
        self.key = Linear(d_model, d_model)
        self.value = Linear(d_model, d_model)

        self.q_output = None
        self.k_output = None
        self.v_output = None

        self.scores = np.array()
        self.softmax = Softmax()

        self.attention = None
        self.output = None

    def _mask_fill(self, tensor: np.ndarray):
        B, T, _ = tensor.shape

        mask = np.triu(np.ones((T, T), dtype=bool), k=1)
        mask = mask[None, :, :]
        tensor = np.where(mask, -np.inf, tensor)
        return tensor

    def forward(self, x: np.array):
        B, T, E = x.shape

        self.q_output = self.query.forward(x)
        self.k_output = self.key.forward(x)
        self.v_output = self.value.forward(x)

        self.scores = (self.q_output @ self.k_output.transpose(-2, -1)) / (E ** 0.5)

        masked_scores = self._mask_fill(self.scores)
        self.attention = self.softmax.forward(masked_scores)

        return np.matmul(self.attention, self.v_output)
    
    def backward(self, incoming_grad: np.array):
        B, T, E = incoming_grad.shape

        dvalue = self.attention.transpose(-2, -1) @ incoming_grad
        dattention = incoming_grad @ self.v_output.transpose(-2, -1)

        dsoftmax = self.softmax.backward(dattention)
        dscaling = dsoftmax / (E ** 0.5)

        dquery = dscaling @ self.k_output
        dkey = dscaling.transpose(-2, -1) @ self.q_output

        dx_v = self.value.backward(dvalue)
        dx_q = self.query.backward(dquery)
        dx_k = self.key.backward(dkey)

        return dx_v + dx_q + dx_k
