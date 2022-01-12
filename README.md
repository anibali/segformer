# SegFormer

This project is an unofficial redistribution of parts of the original code available at
https://github.com/NVlabs/SegFormer. Its purpose is to make using SegFormer models easier through
PyTorch Hub.


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
model = torch.hub.load('anibali/segformer', 'segformer_b2_city', pretrained=True)
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


## Copyright and licenses

This project incorporates code copied from the following third party sources:

* https://github.com/rwightman/pytorch-image-models (Apache License 2.0)
* https://github.com/NVlabs/SegFormer (NVIDIA Source Code License)

Copyright for copied code remains with the original authors. Files containing copied code have been
clearly marked as such with a notice at the beginning of the file.

Excepting code copied from third party sources (as described above), the remaining work is made
available under the terms of the MIT License.
