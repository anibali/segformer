# SegFormer

This project is an unofficial implementation of SegFormer [^1]. Its purpose is to make using
SegFormer models easier through PyTorch Hub.


## Usage

You can either use SegFormer as a standard Python package or as a PyTorch Hub model. Here are
examples of creating a SegFormer B2 model trained on the CityScapes dataset:

```python
# Standard Python package.
from segformer.model import segformer_b2_city
model = segformer_b2_city(pretrained=True)
```

or

```python
# PyTorch Hub.
import torch
model = torch.hub.load('anibali/segformer:v1.1.0', 'segformer_b2_city', pretrained=True)
```

There are many SegFormer models to choose from, which vary based on model size (B0--B5) and
pretraining data: CityScapes, ADE20K, or ImageNet (backbone-only). Please refer to
[`segformer.model`](segformer/model.py) for a full list of available models. It is also
possible to obtain randomly initialised models by setting `pretrained=False`.

Here's an example of creating a 5-class SegFormer-B1 model. The backbone will use pretrained
ImageNet weights but the head weights will be randomly initialised:

```python
import torch
model = torch.hub.load('anibali/segformer', 'segformer_b1', pretrained=True, num_classes=5)
```

Here's an example of creating a SegFormer-B0 model trained on ADE20K data and subsequently changing
the number of output classes to 5:

```python
import torch

model = torch.hub.load('anibali/segformer', 'segformer_b0_ade', pretrained=True)
model.decode_head.rebuild_output_layer_(num_classes=5)
```


[^1]: Xie, E., Wang, W., Yu, Z., Anandkumar, A., Alvarez, J.M. and Luo, P., 2021. SegFormer: Simple
      and Efficient Design for Semantic Segmentation with Transformers. arXiv preprint
      arXiv:2105.15203.
