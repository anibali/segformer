import torch
from mmcv import ConfigDict
from mmseg.models import EncoderDecoder, BaseSegmentor

from segformer.backbones import mit_b0, mit_b1, mit_b2, mit_b3, mit_b4, mit_b5
from segformer.heads import SegFormerHead

model_urls = {
    # Complete SegFormer weights trained on ADE20K.
    'ade': {
        'segformer_b0': 'https://github.com/anibali/segformer/releases/download/v0.0.0/segformer_b0_512x512_ade_160k-46de5006.pth',
        'segformer_b1': 'https://github.com/anibali/segformer/releases/download/v0.0.0/segformer_b1_512x512_ade_160k-0c5d8ae5.pth',
        'segformer_b2': 'https://github.com/anibali/segformer/releases/download/v0.0.0/segformer_b2_512x512_ade_160k-65f853bd.pth',
        'segformer_b3': 'https://github.com/anibali/segformer/releases/download/v0.0.0/segformer_b3_512x512_ade_160k-0fc45502.pth',
        'segformer_b4': 'https://github.com/anibali/segformer/releases/download/v0.0.0/segformer_b4_512x512_ade_160k-5d889df5.pth',
        'segformer_b5': 'https://github.com/anibali/segformer/releases/download/v0.0.0/segformer_b5_640x640_ade_160k-8e73410a.pth',
    },
    # Complete SegFormer weights trained on CityScapes.
    'city': {
        'segformer_b0': 'https://github.com/anibali/segformer/releases/download/v0.0.0/segformer_b0_1024x1024_city_160k-ec9aa2f1.pth',
        'segformer_b1': 'https://github.com/anibali/segformer/releases/download/v0.0.0/segformer_b1_1024x1024_city_160k-11f8e4dd.pth',
        'segformer_b2': 'https://github.com/anibali/segformer/releases/download/v0.0.0/segformer_b2_1024x1024_city_160k-0dcc4ceb.pth',
        'segformer_b3': 'https://github.com/anibali/segformer/releases/download/v0.0.0/segformer_b3_1024x1024_city_160k-b84ccbc9.pth',
        'segformer_b4': 'https://github.com/anibali/segformer/releases/download/v0.0.0/segformer_b4_1024x1024_city_160k-2e933e84.pth',
        'segformer_b5': 'https://github.com/anibali/segformer/releases/download/v0.0.0/segformer_b5_1024x1024_city_160k-d565b9b0.pth',
    },
    # Backbone-only SegFormer weights trained on ImageNet.
    'imagenet': {
        'segformer_b0': 'https://github.com/anibali/segformer/releases/download/v0.0.0/segformer_b0_backbone_imagenet-eb42d485.pth',
        'segformer_b1': 'https://github.com/anibali/segformer/releases/download/v0.0.0/segformer_b1_backbone_imagenet-357971ac.pth',
        'segformer_b2': 'https://github.com/anibali/segformer/releases/download/v0.0.0/segformer_b2_backbone_imagenet-3c162bb8.pth',
        'segformer_b3': 'https://github.com/anibali/segformer/releases/download/v0.0.0/segformer_b3_backbone_imagenet-0d113e32.pth',
        'segformer_b4': 'https://github.com/anibali/segformer/releases/download/v0.0.0/segformer_b4_backbone_imagenet-b757a54d.pth',
        'segformer_b5': 'https://github.com/anibali/segformer/releases/download/v0.0.0/segformer_b5_backbone_imagenet-d552b33d.pth',
    },
}


class SegFormer(EncoderDecoder):
    def __init__(self, backbone, decode_head, train_cfg=None, test_cfg=None):
        BaseSegmentor.__init__(self)

        self.backbone = backbone
        self.decode_head = decode_head
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes
        self._init_auxiliary_head(None)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        assert self.with_decode_head


def create_segformer_b0(num_classes):
    backbone = mit_b0()
    head = SegFormerHead(
        in_channels=[32, 64, 160, 256],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=128,
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        decoder_params=dict(embed_dim=256),
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
    )
    return SegFormer(backbone, head, test_cfg=ConfigDict(mode='whole'))


def create_segformer_b1(num_classes):
    backbone = mit_b1()
    head = SegFormerHead(
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=128,
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        decoder_params=dict(embed_dim=256),
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
    )
    return SegFormer(backbone, head, test_cfg=ConfigDict(mode='whole'))


def create_segformer_b2(num_classes):
    backbone = mit_b2()
    head = SegFormerHead(
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=128,
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        decoder_params=dict(embed_dim=768),
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
    )
    return SegFormer(backbone, head, test_cfg=ConfigDict(mode='whole'))


def create_segformer_b3(num_classes):
    backbone = mit_b3()
    head = SegFormerHead(
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=128,
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        decoder_params=dict(embed_dim=768),
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
    )
    return SegFormer(backbone, head, test_cfg=ConfigDict(mode='whole'))


def create_segformer_b4(num_classes):
    backbone = mit_b4()
    head = SegFormerHead(
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=128,
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        decoder_params=dict(embed_dim=768),
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
    )
    return SegFormer(backbone, head, test_cfg=ConfigDict(mode='whole'))


def create_segformer_b5(num_classes):
    backbone = mit_b5()
    head = SegFormerHead(
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=128,
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        decoder_params=dict(embed_dim=768),
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
    )
    return SegFormer(backbone, head, test_cfg=ConfigDict(mode='whole'))


def _load_pretrained_weights_(model, model_url, progress):
    state_dict = torch.hub.load_state_dict_from_url(model_url, progress=progress)
    model.load_state_dict(state_dict)


def segformer_b0_ade(pretrained=False, progress=True):
    model = create_segformer_b0(num_classes=150)
    if pretrained:
        _load_pretrained_weights_(model, model_urls['ade']['segformer_b0'], progress=progress)
    return model


def segformer_b1_ade(pretrained=False, progress=True):
    model = create_segformer_b1(num_classes=150)
    if pretrained:
        _load_pretrained_weights_(model, model_urls['ade']['segformer_b1'], progress=progress)
    return model


def segformer_b2_ade(pretrained=False, progress=True):
    model = create_segformer_b2(num_classes=150)
    if pretrained:
        _load_pretrained_weights_(model, model_urls['ade']['segformer_b2'], progress=progress)
    return model


def segformer_b3_ade(pretrained=False, progress=True):
    model = create_segformer_b3(num_classes=150)
    if pretrained:
        _load_pretrained_weights_(model, model_urls['ade']['segformer_b3'], progress=progress)
    return model


def segformer_b4_ade(pretrained=False, progress=True):
    model = create_segformer_b4(num_classes=150)
    if pretrained:
        _load_pretrained_weights_(model, model_urls['ade']['segformer_b4'], progress=progress)
    return model


def segformer_b5_ade(pretrained=False, progress=True):
    model = create_segformer_b5(num_classes=150)
    if pretrained:
        _load_pretrained_weights_(model, model_urls['ade']['segformer_b5'], progress=progress)
    return model


def segformer_b0_city(pretrained=False, progress=True):
    model = create_segformer_b0(num_classes=19)
    if pretrained:
        _load_pretrained_weights_(model, model_urls['city']['segformer_b0'], progress=progress)
    return model


def segformer_b1_city(pretrained=False, progress=True):
    model = create_segformer_b1(num_classes=19)
    if pretrained:
        _load_pretrained_weights_(model, model_urls['city']['segformer_b1'], progress=progress)
    return model


def segformer_b2_city(pretrained=False, progress=True):
    model = create_segformer_b2(num_classes=19)
    if pretrained:
        _load_pretrained_weights_(model, model_urls['city']['segformer_b2'], progress=progress)
    return model


def segformer_b3_city(pretrained=False, progress=True):
    model = create_segformer_b3(num_classes=19)
    if pretrained:
        _load_pretrained_weights_(model, model_urls['city']['segformer_b3'], progress=progress)
    return model


def segformer_b4_city(pretrained=False, progress=True):
    model = create_segformer_b4(num_classes=19)
    if pretrained:
        _load_pretrained_weights_(model, model_urls['city']['segformer_b4'], progress=progress)
    return model


def segformer_b5_city(pretrained=False, progress=True):
    model = create_segformer_b5(num_classes=19)
    if pretrained:
        _load_pretrained_weights_(model, model_urls['city']['segformer_b5'], progress=progress)
    return model


def segformer_b0(pretrained=False, progress=True, num_classes=150):
    model = create_segformer_b0(num_classes=num_classes)
    if pretrained:
        _load_pretrained_weights_(model.backbone, model_urls['imagenet']['segformer_b0'],
                                  progress=progress)
    return model


def segformer_b1(pretrained=False, progress=True, num_classes=150):
    model = create_segformer_b1(num_classes=num_classes)
    if pretrained:
        _load_pretrained_weights_(model.backbone, model_urls['imagenet']['segformer_b1'],
                                  progress=progress)
    return model


def segformer_b2(pretrained=False, progress=True, num_classes=150):
    model = create_segformer_b2(num_classes=num_classes)
    if pretrained:
        _load_pretrained_weights_(model.backbone, model_urls['imagenet']['segformer_b2'],
                                  progress=progress)
    return model


def segformer_b3(pretrained=False, progress=True, num_classes=150):
    model = create_segformer_b3(num_classes=num_classes)
    if pretrained:
        _load_pretrained_weights_(model.backbone, model_urls['imagenet']['segformer_b3'],
                                  progress=progress)
    return model


def segformer_b4(pretrained=False, progress=True, num_classes=150):
    model = create_segformer_b4(num_classes=num_classes)
    if pretrained:
        _load_pretrained_weights_(model.backbone, model_urls['imagenet']['segformer_b4'],
                                  progress=progress)
    return model


def segformer_b5(pretrained=False, progress=True, num_classes=150):
    model = create_segformer_b5(num_classes=num_classes)
    if pretrained:
        _load_pretrained_weights_(model.backbone, model_urls['imagenet']['segformer_b5'],
                                  progress=progress)
    return model
