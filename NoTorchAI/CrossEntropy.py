import numpy as np

from NoTorchAI.Activation.Softmax import Softmax
from NoTorchAI.Neuron import Neuron


class CrossEntropy(Neuron):
    def __init__(self, device: str = "cpu"):
        super().__init__(device)
        self.softmax = Softmax(device)
        self.p_logits = None
        self.targets = None

    def forward(self, logits: np.ndarray, targets: np.ndarray):
        B, T, V = logits.shape

        self.targets = targets
        self.p_logits = self.softmax.forward(logits)

        p_correct = self.p_logits[
            self.xp.arange(B)[:, None],
            self.xp.arange(T)[None, :],
            targets
        ]

        L = -self.xp.mean(self.xp.log(p_correct))
        return L

    def backward(self):
        B, T, V = self.p_logits.shape

        grad = self.xp.copy(self.p_logits)

        grad[
            self.xp.arange(B)[:, None],
            self.xp.arange(T)[None, :],
            self.targets
        ] -= 1

        grad /= (B * T)

        # Memory optimization
        self.p_logits = None
        self.targets = None

        return grad
