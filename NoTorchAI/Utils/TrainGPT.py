from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel

from NoTorchAI.GlobalState.Device import Device
from NoTorchAI.GlobalState.Quant import Quant
from NoTorchAI.Gradients.ABSGradient import ABSGradient
from NoTorchAI.Neuron import Neuron
from NoTorchAI.Utils.Batch import Batch
from NoTorchAI.Utils.MatrixOperations import MatrixOperations as mo


class TrainGPT:
    def __init__(self, model: Neuron, gradient: ABSGradient, batch: Batch,
                 block_size: int, batch_size: int, tokenizer_path: str, save_path: str):
        self.model = model
        self.gradient = gradient
        self.batch = batch
        self.block_size = block_size
        self.batch_size = batch_size
        self.save_path = save_path

        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.tokenizer.decoder = ByteLevel()
        self.xp = Device().set_module()
        self.stop_token = self.tokenizer.token_to_id("<|endofstory|>")
        
        # log for this run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logs_path = Path(__file__).parent.parent.parent.resolve() / "logs" / f"{timestamp}.csv"
        self.logs = pd.DataFrame(columns=["step", "lr", "loss", "ema_loss"])

    def _generate_idx(self, idx: np.ndarray, max_tokens: int, temperature: float = 0.9) -> np.ndarray:
        while idx.shape[1] <= max_tokens:
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self.model.forward(idx_cond)

            if temperature == 0:
                next_token = mo.argmax(logits[:, -1, :], axis=-1)
                next_token = mo.expand_dims(next_token, axis=-1)
            else:
                logits = logits[:, -1, :] / temperature
                probs = self.model.softmax.forward(logits)

                next_tokens = []
                for b in range(probs.shape[0]):
                    next_tokens.append(mo.choice(probs.shape[-1], p=probs[b], size=1)[0])

                next_token = self.xp.array(next_tokens).reshape(-1, 1)

            idx = mo.concatenate([idx, next_token], axis=1)

            if mo.all(next_token == self.stop_token):
                break

        return idx

    def generate_str(self, prompt: str, max_tokens: int, temperature: float = 0.9) -> str:
        encoded = self.tokenizer.encode(prompt).ids
        context = self.xp.array(encoded, dtype=self.xp.int32).reshape(1, -1)
        generated = self._generate_idx(context, max_tokens, temperature)
        return self.tokenizer.decode(generated[0].tolist())

    def train(self, prompt: str, max_tokens: int, temperature: float = 0.9, 
              output_sample_step: int = 500, gradient_sample_step: int = 100, 
              start_step: int = 0, end_step: int = 20_000):
        ema_loss = None

        for step in range(start_step, end_step):
            xb, yb = self.batch.get_batch(self.block_size, self.batch_size)

            _, loss = self.model.forward(xb, yb)

            self.gradient.t += 1
            self.model.backward()

            if ema_loss is None:
                ema_loss = loss
            else:
                ema_loss = 0.99 * ema_loss + 0.01 * loss

            if step == 1 or step % gradient_sample_step == 0:
                self.logs = pd.concat([
                    self.logs,
                    pd.DataFrame([{"step": step, "lr": self.gradient.get_lr(), "loss": float(loss), "ema_loss": float(ema_loss)}])
                ], ignore_index=True)
                print(f"step {step}, lr {self.gradient.get_lr():.6f}, loss {loss:.4f}, ema_loss {ema_loss:.4f}")

            if step % output_sample_step == 0:
                print(self.generate_str(prompt, max_tokens, temperature))
                self.logs.to_csv(self.logs_path)
                self.model.save(str(Path.cwd() / self.save_path))

        self.model.save(self.save_path)
