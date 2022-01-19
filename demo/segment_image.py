"""Demo script for a SegFormer model trained on CityScapes data.
"""

import os.path

import numpy as np
import torch
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms.functional import normalize, to_pil_image

# 1. Prepare the input image.
image_path = os.path.join(os.path.dirname(__file__), 'cityscapes.png')
original_image = read_image(image_path, ImageReadMode.RGB).float()
image = normalize(original_image, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])

# 2. Run the SegFormer model.
model = torch.hub.load('anibali/segformer:v1.1.0', 'segformer_b2_city', pretrained=True)
model.eval()
output = model(image[None, ...])[0]
# Note that we don't bother passing the model output through a softmax since we are only interested
# in finding the maximum here.
preds = output.argmax(dim=0)

# 3. Visualise the results.
palette = np.asarray([
    [128,  64, 128], [244,  35, 232], [ 70,  70,  70], [102, 102, 156], [190, 153, 153],
    [153, 153, 153], [250, 170,  30], [220, 220,   0], [107, 142,  35], [152, 251, 152],
    [ 70, 130, 180], [220,  20,  60], [255,   0,   0], [  0,   0, 142], [  0,   0,  70],
    [  0,  60, 100], [  0,  80, 100], [  0,   0, 230], [119,  11,  32],
])
overlay = np.zeros((preds.shape[0], preds.shape[1], 3), dtype=np.uint8)
for label, color in enumerate(palette):
    overlay[preds == label, :] = color
base_image = np.asarray(original_image.permute(1, 2, 0), np.float32)
overlay = np.asarray(overlay, np.float32)
blended = (base_image * 0.5 + overlay * 0.5).astype(np.uint8)
to_pil_image(blended).show()
