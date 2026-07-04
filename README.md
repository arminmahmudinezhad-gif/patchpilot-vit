# PatchPilot ViT

**Repository name:** `patchpilot-vit`

**Description:** A lightweight PyTorch Vision Transformer notebook demonstrating patch embeddings, transformer encoding, and an image-classification training loop.

PatchPilot ViT is a compact educational implementation of a Vision Transformer
(ViT) image classifier in PyTorch. The notebook shows the main ViT pipeline:
turning an image into fixed-size patches, projecting patches into embeddings,
adding a class token and positional embeddings, passing the sequence through a
Transformer encoder, and using the class token for classification.

## What Is Inside

- `FakeCIFAR10`: a small synthetic dataset with random RGB images and random labels.
- `PatchEmbedding`: converts each `224 x 224` image into `16 x 16` patches using a convolution.
- `ViTEmbedding`: adds the learnable class token and positional embedding.
- `SimpleViT`: combines the embedding block, Transformer encoder, LayerNorm, and linear classifier.
- Training and validation helpers based on CrossEntropy loss, AdamW, and a cosine learning-rate scheduler.

## Important Note About Results

The current notebook uses random synthetic data, not the real CIFAR-10 dataset.
Because both images and labels are random, accuracy is not expected to become
meaningful. This repository is best understood as a clean ViT architecture and
training-loop practice notebook. To produce real experimental results, replace
`FakeCIFAR10` with a real image dataset such as CIFAR-10, Tiny ImageNet, or a
custom dataset.

## Suggested Fixes Before Running

The notebook is almost ready, but two small fixes are needed:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

Add this before creating the model.

Also rename:

```python
def claculate_accuracy(logits, labels):
```

to:

```python
def calculate_accuracy(logits, labels):
```

The training and validation functions call `calculate_accuracy`, so the function
name must match.

## Quick Start

1. Open `Vit.ipynb` in Jupyter Notebook, JupyterLab, or Google Colab.
2. Install PyTorch if it is not already available.
3. Add the `device` line before model creation.
4. Fix the accuracy function name.
5. Run the cells from top to bottom.

## Minimal Dependencies

```text
torch
```

Optional, if you later add plots or dataset visualization:

```text
matplotlib
torchvision
```

## Model Summary

- Image size: `224 x 224`
- Patch size: `16 x 16`
- Number of patches: `196`
- Token sequence length: `197` including the class token
- Embedding dimension: `256`
- Transformer depth: `6`
- Attention heads: `8`
- MLP dimension: `512`
- Number of classes: `10`

## Recommended Next Steps

- Replace `FakeCIFAR10` with a real dataset.
- Save the best model weights with `torch.save`.
- Plot training and validation loss/accuracy curves.
- Compare different patch sizes, embedding dimensions, and Transformer depths.
- Unfreeze the full model once real data is used.
