"""A simple demo script for showing a CityScapes SegFormer model in action.
"""

import argparse
import sys

import numpy as np
import torch
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms.functional import normalize, to_pil_image

from segformer.model import segformer_b2_city


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='path to input image file')
    parser.add_argument('--device', default='cuda:0',
                        help='device to use for inference')
    return parser


def cityscapes_palette():
    return [
        [128,  64, 128], [244,  35, 232], [ 70,  70,  70], [102, 102, 156], [190, 153, 153],
        [153, 153, 153], [250, 170,  30], [220, 220,   0], [107, 142,  35], [152, 251, 152],
        [ 70, 130, 180], [220,  20,  60], [255,   0,   0], [  0,   0, 142], [  0,   0,  70],
        [  0,  60, 100], [  0,  80, 100], [  0,   0, 230], [119,  11,  32],
    ]


def create_overlay(seg, palette):
    palette = np.asarray(palette)
    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
    for label, color in enumerate(palette):
        color_seg[seg == label, :] = color
    return color_seg


def blend(image, overlay, alpha):
    image = np.asarray(image, np.float32)
    overlay = np.asarray(overlay, np.float32)
    return (image * (1 - alpha) + overlay * alpha).astype(np.uint8)


def main(args):
    opts = argument_parser().parse_args(args)

    torch.set_grad_enabled(False)
    device = torch.device(opts.device)

    model = segformer_b2_city(pretrained=True, progress=True)
    model.eval()
    model.to(device)

    original_image = read_image(opts.input, ImageReadMode.RGB).float()
    image = normalize(original_image, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])

    seg_logit = model(image[None, ...].to(device))[0]
    # Note that we don't bother passing the logits through a softmax since we are only interested
    # in finding the maximum here.
    seg_pred = seg_logit.argmax(dim=0).cpu().numpy()

    overlay = create_overlay(seg_pred, palette=cityscapes_palette())
    blended = blend(original_image.permute(1, 2, 0), overlay, alpha=0.5)
    to_pil_image(blended).show()


if __name__ == '__main__':
    main(sys.argv[1:])
