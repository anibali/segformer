import torch

from segformer.model import SegFormer, mit_b1, SegFormerHead


def test_SegFormer():
    backbone = mit_b1()
    head = SegFormerHead(in_channels=[64, 128, 320, 512], num_classes=150, embed_dim=256)
    model = SegFormer(backbone, head)
    x = torch.randn(2, 3, 256, 256)
    y = model(x)
    assert y.shape == (2, 150, 256, 256)
