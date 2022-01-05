import torch

from segformer.heads import SegFormerHead


def test_SegFormerHead():
    head = SegFormerHead(
        in_channels=[64, 128, 320, 512],
        feature_strides=[4, 8, 16, 32],
        channels=128,
        dropout_ratio=0.1,
        num_classes=150,
        align_corners=False,
        embed_dim=256,
    )
    x = [
        torch.randn(2, 64, 64, 64),
        torch.randn(2, 128, 32, 32),
        torch.randn(2, 320, 16, 16),
        torch.randn(2, 512, 8, 8),
    ]
    y = head(x)
    assert y.shape == (2, 150, 64, 64)
