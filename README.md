# MyGPT

**A minimal Transformer‑based language model built from scratch using only NumPy.**

This repository is an educational playground for anyone who wants to learn how modern large language
models work under the hood. Instead of relying on PyTorch, TensorFlow, or any deep‑learning
framework, every component—from embeddings and self‑attention to layer normalization and the
training loop—is implemented manually using basic linear algebra operations.

> ⚠️ This code is designed for clarity and learning. It’s **not** optimized for performance or
> production use.

---

## 🔧 Project Structure

```
NoTorchAI/                 # core library modules
  ActivationFunc.py
  CrossEntropy.py
  Embedding.py
  Gradients/               # optimizers
    ABSGradient.py
    SGD.py
    Adam.py
  Layers/                  # building blocks
    ABSLayer.py
    LinearLayer.py
    NormLayer.py
  LLM/                     # transformer components
    EmbeddingInitial.py
    SelfAttention.py
    FeedForward.py
    Block.py
    MiniGPT.py             # small GPT‑style model

*.ipynb                   # notebooks for training and experiments
requirements.txt          # Python dependencies (mostly numpy)
README.md                 # you are reading it right now
```

The `NoTorchAI` package contains everything needed to instantiate, train, and sample from a
MiniGPT‑style model; the notebooks demonstrate how to wire the pieces together.

---

## 🚀 Getting Started

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your‑username>/MyGPT.git
   cd MyGPT
   ```

2. **Create a virtual environment and install dependencies**
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Linux/macOS
   # or .\.venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```

3. **Run one of the provided notebooks**

   - `No_Torch_Training.ipynb` – train the NumPy‑only model from scratch
   - `Torch_Training.ipynb` – reference implementation using PyTorch for comparison
   - `Pipeline_text_transformation.ipynb` – simple text preprocessing/augmentation examples

   Open them with Jupyter:
   ```bash
   jupyter notebook
   ```

4. **Use the library programmatically**

   Import classes from `NoTorchAI` in your own scripts:

   ```python
   from NoTorchAI.LLM.MiniGPT import MiniGPT
   from NoTorchAI.Gradients.SGD import SGD
   from NoTorchAI.CrossEntropy import CrossEntropy
   ```

   Assemble a model, loss, and optimizer, then loop over batches of data just as you would
   in a deep‑learning tutorial.

5. **Generate text**

   After training, call the model’s `generate` method (see notebooks) to sample continuations from
a prompt.

---

## ✏️ Features

- Pure NumPy implementation, no external ML frameworks
- Core Transformer components: embedding, self‑attention, feed‑forward, layer norm
- Custom gradient and optimizer classes (SGD, Adam)
- Simple cross‑entropy loss and text generation routines
- Educational Jupyter notebooks demonstrating training and usage
