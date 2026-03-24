from NoTorchAI.Layers.Embedding import Embedding
from NoTorchAI.Gradients import ABSGradient
from NoTorchAI.Layers.ABSLayer import ABSLayer
from NoTorchAI.Neuron import Neuron


class SGD(ABSGradient, Neuron):
    def __init__(self, lr: float, warmup_steps: int = 100, device: str = "cpu"):
        super().__init__(device)
        self.base_lr = lr
        self.warmup_steps = warmup_steps
        self.step_count = 0

    def get_lr(self):
        if self.step_count < self.warmup_steps:
            return self.base_lr * (self.step_count / self.warmup_steps)
        progress = (self.step_count - self.warmup_steps) / max(1, 5000 - self.warmup_steps)
        return self.base_lr * 0.5 * (1 + self.xp.cos(self.xp.pi * progress))

    def step(self, layer: ABSLayer | Embedding) -> None:
        lr = self.get_lr()

        if isinstance(layer, Embedding):
            layer.embeddings -= lr * layer.d_embeddings
        elif isinstance(layer, ABSLayer):
            layer.weights -= lr * layer.d_weights
            layer.bias -= lr * layer.d_bias

        self.step_count += 1
