import numpy as np

from NoTorchAI.ActivationFunc import Softmax


class CrossEntropy:
    def __init__(self):
        self.softmax = Softmax()
        self.p_logits = None
        self.targets = None

    def forward(self, logits: np.ndarray, targets: np.ndarray):
        B, T, V = logits.shape

        self.targets = targets
        self.p_logits = self.softmax.forward(logits)

        p_correct = self.p_logits[
            np.arange(B)[:, None],
            np.arange(T)[None, :],
            targets
        ]

        L = -np.mean(np.log(p_correct))
        return L

    def backward(self):
        B, T, V = self.p_logits.shape

        grad = np.copy(self.p_logits)

        grad[
            np.arange(B)[:, None],
            np.arange(T)[None, :],
            self.targets
        ] -= 1

        grad /= (B * T)

        # Memory optimization
        self.p_logits = None
        self.targets = None

        return grad
