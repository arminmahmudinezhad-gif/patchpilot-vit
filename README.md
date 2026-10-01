# PatchPilot ViT

This is a small PyTorch project I put together to work through the main parts of a Vision Transformer classifier: splitting an image into patches, embedding them, adding a class token and positional embeddings, and passing the sequence to a Transformer encoder.

## Model

`PatchEmbedding` turns a 224 × 224 image into 196 patches of size 16 × 16. `ViTEmbedding` adds the learnable class and positional embeddings. The encoder uses PyTorch's `nn.TransformerEncoder`; a LayerNorm and linear layer produce the class scores.

The current configuration uses an embedding dimension of 256, six encoder layers, eight attention heads, an MLP dimension of 512, and ten output classes.

## Data and current status

The notebook uses `FakeCIFAR10`, which creates random images and random labels. It does **not** load the real CIFAR-10 dataset. I used it to sketch out the data and training pipeline, so its accuracy should not be read as an image-classification result.

The notebook still needs a couple of fixes before it runs end to end: define `device` before creating the model, and correct the `claculate_accuracy` function name to `calculate_accuracy` so it matches the calls below. There are no saved training results in the repository yet.

## Run the notebook

Open `Vit.ipynb` in Jupyter or Colab with PyTorch installed. After making the two fixes above, run the cells from top to bottom. To evaluate the model, replace the synthetic dataset with a real one and report results from an actual run.

## Dependency

- PyTorch
