# PatchPilot ViT

This is a small PyTorch project I put together to work through the main parts of a Vision Transformer classifier: splitting an image into patches, embedding them, adding a class token and positional embeddings, and passing the sequence to a Transformer encoder.

## Model

`PatchEmbedding` turns a 224 × 224 image into 196 patches of size 16 × 16. `ViTEmbedding` adds the learnable class and positional embeddings. The encoder uses PyTorch's `nn.TransformerEncoder`; a LayerNorm and linear layer produce the class scores.

The current configuration uses an embedding dimension of 256, six encoder layers, eight attention heads, an MLP dimension of 512, and ten output classes.

## Dataset

The notebook uses a synthetic dataset in a CIFAR-10-style format for its examples.

## Run the notebook

Open `Vit.ipynb` in Jupyter or Google Colab. The project uses PyTorch.

## Dependency

- PyTorch
