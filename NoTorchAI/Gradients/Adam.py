import numpy as np

from NoTorchAI.Embedding import Embedding
from NoTorchAI.Gradients.ABSGradient import ABSGradient
from NoTorchAI.Layers.ABSLayer import ABSLayer


class Adam(ABSGradient):
    def __init__(self, lr=3e-4, beta1=0.9, beta2=0.95, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = {}
        self.v = {}

    def step(self, layer: ABSLayer | Embedding) -> None:
        self.t += 1
        for param_name in ['weights', 'bias', 'embeddings', 'd_embeddings']:
            grad_name = 'd_' + param_name
            if not hasattr(layer, grad_name):
                continue
            grad = getattr(layer, grad_name)
            param = getattr(layer, param_name)
            key = (id(layer), param_name)

            self.m[key] = self.beta1 * self.m.get(key, 0) + (1 - self.beta1) * grad
            self.v[key] = self.beta2 * self.v.get(key, 0) + (1 - self.beta2) * grad**2

            m_hat = self.m[key] / (1 - self.beta1**self.t)
            v_hat = self.v[key] / (1 - self.beta2**self.t)

            setattr(layer, param_name, param - self.lr * m_hat / (np.sqrt(v_hat) + self.eps))
