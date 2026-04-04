# Challenges in Building MyGPT

Building this mini GPT-2–style model from scratch was a huge learning experience, but it came with its fair share of hurdles. I usually don't dive into the details of my projects like this, but since this was a big milestone for me, I figured I'd share some of the real challenges I ran into. Everything was done without relying on big frameworks like PyTorch or TensorFlow — just NumPy/CuPy and custom code. Here's a breakdown of the main issues and how I dealt with them.

## Implementing Everything from Scratch

The core idea was to understand the internals, so I built all the components myself: forward and backward passes, attention mechanisms, and the whole training loop. That meant a ton of trial and error. Debugging custom gradients and making sure the math lined up was tricky, especially when things didn't work as expected. It was rewarding to see it come together, but it took way longer than using an off-the-shelf library.

## Instruction Extraction

One of the messier parts was trying to extract "instructions" from text using only algorithms, no pretrained models. I experimented with using embeddings from an early model version to figure out the main intent, then mixed in a small set of hardcoded basic instructions. It sort of worked sometimes, but it wasn't clean or reliable. It showed me how hard it is to parse meaning without modern tools, and it made me appreciate how much pretrained models handle for you.

## Data Handling

Data choice hit me hard early on. The model just couldn't cope with noisy, complex stuff like web data or Wikipedia articles. I had to pivot to a simpler dataset of children's stories, which was more manageable for a small model. It wasn't the most exciting data, but it kept things realistic and actually trainable. Lesson learned: match your data to your model's capabilities from the start.

## Tokenization Woes

Tokenization was a total rabbit hole. I tried rolling my own BPE (Byte Pair Encoding) in Python, and while it technically worked, it was painfully slow — slower than the actual training! That killed my workflow, so I switched to a Hugging Face tokenizer to get things moving. It was a reminder that sometimes you have to compromise on the "from scratch" purity for practicality.

## Slow Iteration and Experimentation

Without optimized frameworks or serious hardware, testing changes was a drag. Even small tweaks to parameters meant long waits, which changed how I approached experiments. I had to be more thoughtful about what to test and how, rather than just throwing things at the wall. It taught me patience and the value of good planning.

Overall, these challenges made the project way more educational. I ended up with a basic "framework" of my own, which opens doors for future experiments, like maybe adding computer vision. It's not perfect, but that's the point — it was fun to build and learn from the struggles.