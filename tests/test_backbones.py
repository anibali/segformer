import torch

from segformer.backbones import mit_b1


def test_mit_b1():
    backbone = mit_b1()
    x = torch.randn(2, 3, 256, 256)
    y = backbone(x)
    assert len(y) == 4
    assert y[0].shape == (2, 64, 64, 64)
    assert y[1].shape == (2, 128, 32, 32)
    assert y[2].shape == (2, 320, 16, 16)
    assert y[3].shape == (2, 512, 8, 8)
