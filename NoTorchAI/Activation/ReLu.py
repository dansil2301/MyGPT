import numpy as np

from NoTorchAI.Activation.ActivationFunc import ActivationFunc


class ReLu(ActivationFunc):
    def __init__(self, device: str = "cpu"):
        super().__init__(device)
        self.mask = None

    def forward(self, x):
        self.mask = (x > 0)  # store mask directly
        return self.xp.where(self.mask, x, 0)

    def backward(self, incoming_grad):
        return self.mask * incoming_grad
