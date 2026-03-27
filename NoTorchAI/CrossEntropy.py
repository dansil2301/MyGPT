import numpy as np

from NoTorchAI.Activation.Softmax import Softmax
from NoTorchAI.Neuron import Neuron
from NoTorchAI.Utils.Matrix import Matrix


class CrossEntropy(Neuron):
    def __init__(self):
        self.softmax = Softmax()
        self.p_logits = None
        self.targets = None

    def forward(self, logits: np.ndarray, targets: np.ndarray):
        B, T, V = logits.shape

        self.targets = targets
        
        x = logits - logits.max(axis=-1, keepdims=True)
        exp = Matrix.exp(x)
        self.p_logits = exp / exp.sum(axis=-1, keepdims=True)

        p_correct = self.p_logits[
            Matrix.arange(B)[:, None],
            Matrix.arange(T)[None, :],
            targets
        ]

        L = -Matrix.mean(Matrix.log(Matrix.clip(p_correct, 1e-10, 1.0)))
        return L

    def backward(self):
        B, T, V = self.p_logits.shape

        grad = Matrix.copy(self.p_logits)

        grad[
            Matrix.arange(B)[:, None],
            Matrix.arange(T)[None, :],
            self.targets
        ] -= 1

        grad /= (B * T)

        # Memory optimization
        self.p_logits = None
        self.targets = None

        return grad
