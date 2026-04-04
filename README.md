# MyGPT

This is a project where I'm building and experimenting with language models from the ground up. Started as a way to understand how transformers work without relying on big libraries like PyTorch, but it's grown into a full setup with custom training code, some trained models, and data processing pipelines.

The core is the NoTorchAI library, which implements all the neural network bits using just NumPy/Cupy. I've used it to train models on stories and instructions, and there are notebooks for training, text processing, and testing things out.

---

## Project Structure

```
MyGPT/
├── NoTorchAI/                 # Custom AI framework built with NumPy
│   ├── Activation/
│   │   ├── ActivationFunc.py
│   │   ├── ReLu.py
│   │   └── Softmax.py
│   ├── GlobalState/
│   │   ├── Device.py
│   │   └── Quant.py
│   ├── Gradients/
│   │   ├── ABSGradient.py
│   │   ├── Adam.py
│   │   └── SGD.py
│   ├── Layers/
│   │   ├── ABSLayer.py
│   │   ├── Embedding.py
│   │   ├── LinearLayer.py
│   │   └── NormLayer.py
│   ├── LLM/
│   │   ├── Block.py
│   │   ├── EmbeddingInitial.py
│   │   ├── FeedForward.py
│   │   ├── MiniGPT.py
│   │   ├── MultiHead.py
│   │   └── SelfAttention.py
│   ├── NLP/
│   │   ├── childern_stories_embeddings.npy
│   │   ├── KeyBERT.py
│   │   ├── Rake.py
│   │   └── StopWords.txt
│   ├── Stats/
│   │   ├── KNN.py
│   │   └── PCA.py
│   ├── Utils/
│   │   ├── Batch.py
│   │   ├── MatrixOperations.py
│   │   └── TrainGPT.py
│   ├── CrossEntropy.py
│   └── Neuron.py
├── No_Torch_Training.ipynb    # Training notebook
├── Pipeline_text_transformation.ipynb  # Text processing notebook
├── test.py                    # Test script
├── requirements.txt           # Dependencies
├── README.md                  # This file
└── challenges.md              # My journey a bit more detailed
```

The NoTorchAI folder has all the building blocks for the models. The notebooks show how to use everything.

---

## Getting Started

1. **Grab the code**
   ```bash
   git clone https://github.com/dansil2301/MyGPT.git
   cd MyGPT
   ```

2. **Set up the environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # On Linux/Mac
   # Or .\.venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```

3. **Check out the notebooks**

   - `No_Torch_Training.ipynb` - Trains a model from scratch using the custom framework
   - `Pipeline_text_transformation.ipynb` - Handles text preprocessing and transformations

   Fire up Jupyter to run them:
   ```bash
   jupyter notebook
   ```

4. **Use the code directly**

   Pull in the classes from NoTorchAI for your own scripts:

   ```python
   from NoTorchAI.LLM.MiniGPT import MiniGPT
   from NoTorchAI.Gradients.Adam import Adam
   ```

   Build your model, pick an optimizer, and train on your data.

5. **Generate some text**

   Once trained, use the model's generate method to create text from prompts. Check the notebooks for examples.

---

## What's Included

- **Custom AI Framework**: All neural net components in NumPy - embeddings, attention, layers, you name it.
- **Saving**: Custom model saving so that one should lose progress of training.
- **Data Processing**: Tools for handling text, tokenizers, and embeddings.
- **Training Tools**: Optimizers (SGD, Adam), loss functions, and batch processing.
- **Notebooks**: Hands-on examples for training and text work.
- **Logs**: Training history to verify things.

This is all about learning and experimenting, not production-ready code. Performance isn't optimized, but it's great for understanding the internals. 

At least helped me :)
