import torch
from mmcv import ConfigDict

from segformer.model import SegFormer, mit_b1, SegFormerHead


def test_SegFormer():
    backbone = mit_b1()
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
    model = SegFormer(backbone, head, test_cfg=ConfigDict(mode='whole'))
    x = torch.randn(2, 3, 256, 256)
    y = model.encode_decode(x, None)
    assert y.shape == (2, 150, 256, 256)
