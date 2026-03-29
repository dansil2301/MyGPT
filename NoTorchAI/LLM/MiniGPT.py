import numpy as np

from NoTorchAI.Activation.Softmax import Softmax
from NoTorchAI.GlobalState.Device import Device
from NoTorchAI.Neuron import Neuron
from NoTorchAI.CrossEntropy import CrossEntropy
from NoTorchAI.Gradients.ABSGradient import ABSGradient
from NoTorchAI.LLM.Block import Block
from NoTorchAI.LLM.EmbeddingInitial import EmbeddingInitial
from NoTorchAI.Layers.LinearLayer import Linear
from NoTorchAI.Layers.NormLayer import NormLayer
from NoTorchAI.Utils.MatrixOperations import MatrixOperations as mo


class MiniGPT(Neuron):
    def __init__(self, vocab_size: int, d_model: int, block_size: int, n_layers: int, n_heads: int, gradient: ABSGradient):
        self.initial_embedding = EmbeddingInitial(vocab_size, d_model, block_size, gradient)

        self.blocks = [Block(d_model, n_heads, gradient) for _ in range(n_layers)]

        self.linear = NormLayer(d_model)
        self.head = Linear(d_model, vocab_size)
        self.cross_entropy = CrossEntropy()

        self.softmax = Softmax()

        self.block_size = block_size

        self.gradient = gradient

    def _change_weights(self) -> None:
        self.gradient.step(self.linear)
        self.gradient.step(self.head) 

    def _execute_blocks_forward(self, x: np.ndarray) -> np.ndarray:
        for block in self.blocks:
            x = block.forward(x)
        return x
    
    def _execute_blocks_backward(self, grad: np.ndarray) -> np.ndarray:
        for block in reversed(self.blocks):
            grad = block.backward(grad)
        return grad

    def forward(self, x: np.ndarray, targets: np.ndarray = None) -> tuple:
        B, T = x.shape

        x = self.initial_embedding.forward(x)

        x = self._execute_blocks_forward(x)

        x = self.linear.forward(x)

        logits = self.head.forward(x)
        
        if targets is not None:
            loss = self.cross_entropy.forward(logits, targets)
        else:
            loss = None

        return logits, loss
    
    def backward(self):
        loss_grad = self.cross_entropy.backward()
        grad = self.head.backward(loss_grad)

        grad = self.linear.backward(grad)

        grad = self._execute_blocks_backward(grad)

        self.initial_embedding.backward(grad)
        self._change_weights()


    def generate(self, idx, max_tokens, temperature: float = 0.9):
        next_tokens = []
        while idx.shape[1] <= max_tokens:
            idx_cond = idx[:, -self.block_size:]

            logits, _ = self.forward(idx_cond)

            if temperature == 0:
                # greedy decode
                next_token = mo.argmax(logits[:, -1, :], axis=-1)
                next_token = mo.expand_dims(next_token, axis=-1)
            else:
                logits = logits[:, -1, :] / temperature
                probs = self.softmax.forward(logits)

                if Device().device == "gpu":
                    probs = probs.get()  # cupy conversion

                next_tokens = []
                for b in range(probs.shape[0]):
                    next_token = np.random.choice(probs.shape[-1], p=probs[b])
                    next_tokens.append(next_token)

                next_token = np.array(next_tokens).reshape(-1, 1)

            idx = np.concatenate([idx, next_token], axis=1)

        return idx
