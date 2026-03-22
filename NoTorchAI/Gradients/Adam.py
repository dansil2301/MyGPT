import numpy as np

from NoTorchAI.Gradients.ABSGradient import ABSGradient
from NoTorchAI.Layers.ABSLayer import ABSLayer
from NoTorchAI.Layers.Embedding import Embedding


class Adam(ABSGradient):
    def __init__(self, lr=3e-4, beta1=0.9, beta2=0.95, eps=1e-8, warmup_steps=0, min_lr=1e-6, device: str = "cpu"):
        super().__init__(device)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr
        self.m = {}
        self.v = {}

    def get_lr(self):
        if self.warmup_steps > 0 and self.t < self.warmup_steps:
            return max(self.min_lr, self.lr * (self.t / max(1, self.warmup_steps)))
        return max(self.min_lr, self.lr)

    def step(self, layer: ABSLayer | Embedding):
        for param_name in ['weights', 'bias', 'embeddings']:

            grad_name = 'd_' + param_name
            if not hasattr(layer, grad_name):
                continue

            grad = getattr(layer, grad_name)
            grad = self.xp.clip(grad, -5.0, 5.0)
            param = getattr(layer, param_name)

            key = (id(layer), param_name)

            if key not in self.m:
                self.m[key] = self.xp.zeros_like(grad)
                self.v[key] = self.xp.zeros_like(grad)

            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grad
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grad ** 2)

            if self.t == 0:
                m_hat = self.m[key]
                v_hat = self.v[key]
            else:
                m_hat = self.m[key] / (1 - self.beta1 ** self.t)
                v_hat = self.v[key] / (1 - self.beta2 ** self.t)

            lr = self.get_lr()
            param -= lr * m_hat / (self.xp.sqrt(v_hat) + self.eps)

            setattr(layer, param_name, param)
