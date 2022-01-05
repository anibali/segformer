import torch

from segformer.heads import SegFormerHead


def test_SegFormerHead():
    head = SegFormerHead(
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=128,
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        decoder_params=dict(embed_dim=256),
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
    )
    x = [
        torch.randn(2, 64, 64, 64),
        torch.randn(2, 128, 32, 32),
        torch.randn(2, 320, 16, 16),
        torch.randn(2, 512, 8, 8),
    ]
    y = head(x)
    assert y.shape == (2, 150, 64, 64)
