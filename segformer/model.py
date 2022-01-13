import torch
from torch import nn
from torch.nn.functional import interpolate

from segformer.backbones import mit_b0, mit_b1, mit_b2, mit_b3, mit_b4, mit_b5, MixTransformer
from segformer.heads import SegFormerHead

model_urls = {
    # Complete SegFormer weights trained on ADE20K.
    'ade': {
        'segformer_b0': 'https://github.com/anibali/segformer/releases/download/v0.0.0/segformer_b0_512x512_ade_160k-d0c08cfd.pth',
        'segformer_b1': 'https://github.com/anibali/segformer/releases/download/v0.0.0/segformer_b1_512x512_ade_160k-1cd52578.pth',
        'segformer_b2': 'https://github.com/anibali/segformer/releases/download/v0.0.0/segformer_b2_512x512_ade_160k-fa162a4f.pth',
        'segformer_b3': 'https://github.com/anibali/segformer/releases/download/v0.0.0/segformer_b3_512x512_ade_160k-5abb3eb3.pth',
        'segformer_b4': 'https://github.com/anibali/segformer/releases/download/v0.0.0/segformer_b4_512x512_ade_160k-bb0fa50c.pth',
        'segformer_b5': 'https://github.com/anibali/segformer/releases/download/v0.0.0/segformer_b5_640x640_ade_160k-106a5e57.pth',
    },
    # Complete SegFormer weights trained on CityScapes.
    'city': {
        'segformer_b0': 'https://github.com/anibali/segformer/releases/download/v0.0.0/segformer_b0_1024x1024_city_160k-3e581249.pth',
        'segformer_b1': 'https://github.com/anibali/segformer/releases/download/v0.0.0/segformer_b1_1024x1024_city_160k-e415b121.pth',
        'segformer_b2': 'https://github.com/anibali/segformer/releases/download/v0.0.0/segformer_b2_1024x1024_city_160k-9793f658.pth',
        'segformer_b3': 'https://github.com/anibali/segformer/releases/download/v0.0.0/segformer_b3_1024x1024_city_160k-732b9fde.pth',
        'segformer_b4': 'https://github.com/anibali/segformer/releases/download/v0.0.0/segformer_b4_1024x1024_city_160k-1836d907.pth',
        'segformer_b5': 'https://github.com/anibali/segformer/releases/download/v0.0.0/segformer_b5_1024x1024_city_160k-2ca4dff8.pth',
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


class SegFormer(nn.Module):
    def __init__(self, backbone: MixTransformer, decode_head: SegFormerHead):
        super().__init__()
        self.backbone = backbone
        self.decode_head = decode_head

    @property
    def align_corners(self):
        return self.decode_head.align_corners

    @property
    def num_classes(self):
        return self.decode_head.num_classes

    def forward(self, x):
        image_hw = x.shape[2:]
        x = self.backbone(x)
        x = self.decode_head(x)
        x = interpolate(x, size=image_hw, mode='bilinear', align_corners=self.align_corners)
        return x


def create_segformer_b0(num_classes):
    backbone = mit_b0()
    head = SegFormerHead(
        in_channels=(32, 64, 160, 256),
        dropout_p=0.1,
        num_classes=num_classes,
        align_corners=False,
        embed_dim=256,
    )
    return SegFormer(backbone, head)


def create_segformer_b1(num_classes):
    backbone = mit_b1()
    head = SegFormerHead(
        in_channels=(64, 128, 320, 512),
        dropout_p=0.1,
        num_classes=num_classes,
        align_corners=False,
        embed_dim=256,
    )
    return SegFormer(backbone, head)


def create_segformer_b2(num_classes):
    backbone = mit_b2()
    head = SegFormerHead(
        in_channels=(64, 128, 320, 512),
        dropout_p=0.1,
        num_classes=num_classes,
        align_corners=False,
        embed_dim=768,
    )
    return SegFormer(backbone, head)


def create_segformer_b3(num_classes):
    backbone = mit_b3()
    head = SegFormerHead(
        in_channels=(64, 128, 320, 512),
        dropout_p=0.1,
        num_classes=num_classes,
        align_corners=False,
        embed_dim=768,
    )
    return SegFormer(backbone, head)


def create_segformer_b4(num_classes):
    backbone = mit_b4()
    head = SegFormerHead(
        in_channels=(64, 128, 320, 512),
        dropout_p=0.1,
        num_classes=num_classes,
        align_corners=False,
        embed_dim=768,
    )
    return SegFormer(backbone, head)


def create_segformer_b5(num_classes):
    backbone = mit_b5()
    head = SegFormerHead(
        in_channels=(64, 128, 320, 512),
        dropout_p=0.1,
        num_classes=num_classes,
        align_corners=False,
        embed_dim=768,
    )
    return SegFormer(backbone, head)


def _load_pretrained_weights_(model, model_url, progress):
    state_dict = torch.hub.load_state_dict_from_url(model_url, progress=progress)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('decode_head'):
            if k.endswith('.proj.weight'):
                k = k.replace('.proj.weight', '.weight')
                v = v[..., None, None]
            elif k.endswith('.proj.bias'):
                k = k.replace('.proj.bias', '.bias')
            elif '.linear_fuse.conv.' in k:
                k = k.replace('.linear_fuse.conv.', '.linear_fuse.')
            elif '.linear_fuse.bn.' in k:
                k = k.replace('.linear_fuse.bn.', '.bn.')

            if '.linear_c4.' in k:
                k = k.replace('.linear_c4.', '.layers.0.')
            elif '.linear_c3.' in k:
                k = k.replace('.linear_c3.', '.layers.1.')
            elif '.linear_c2.' in k:
                k = k.replace('.linear_c2.', '.layers.2.')
            elif '.linear_c1.' in k:
                k = k.replace('.linear_c1.', '.layers.3.')
        else:
            if 'patch_embed1.' in k:
                k = k.replace('patch_embed1.', 'stages.0.patch_embed.')
            elif 'patch_embed2.' in k:
                k = k.replace('patch_embed2.', 'stages.1.patch_embed.')
            elif 'patch_embed3.' in k:
                k = k.replace('patch_embed3.', 'stages.2.patch_embed.')
            elif 'patch_embed4.' in k:
                k = k.replace('patch_embed4.', 'stages.3.patch_embed.')
            elif 'block1.' in k:
                k = k.replace('block1.', 'stages.0.blocks.')
            elif 'block2.' in k:
                k = k.replace('block2.', 'stages.1.blocks.')
            elif 'block3.' in k:
                k = k.replace('block3.', 'stages.2.blocks.')
            elif 'block4.' in k:
                k = k.replace('block4.', 'stages.3.blocks.')
            elif 'norm1.' in k:
                k = k.replace('norm1.', 'stages.0.norm.')
            elif 'norm2.' in k:
                k = k.replace('norm2.', 'stages.1.norm.')
            elif 'norm3.' in k:
                k = k.replace('norm3.', 'stages.2.norm.')
            elif 'norm4.' in k:
                k = k.replace('norm4.', 'stages.3.norm.')

            if '.mlp.dwconv.dwconv.' in k:
                k = k.replace('.mlp.dwconv.dwconv.', '.mlp.conv.')

            if '.mlp.' in k:
                k = k.replace('.mlp.', '.ffn.')
        new_state_dict[k] = v
    model.load_state_dict(new_state_dict)


def segformer_b0_ade(pretrained=True, progress=True):
    """Create a SegFormer-B0 model for the ADE20K segmentation task.
    """
    model = create_segformer_b0(num_classes=150)
    if pretrained:
        _load_pretrained_weights_(model, model_urls['ade']['segformer_b0'], progress=progress)
    return model


def segformer_b1_ade(pretrained=True, progress=True):
    """Create a SegFormer-B1 model for the ADE20K segmentation task.
    """
    model = create_segformer_b1(num_classes=150)
    if pretrained:
        _load_pretrained_weights_(model, model_urls['ade']['segformer_b1'], progress=progress)
    return model


def segformer_b2_ade(pretrained=True, progress=True):
    """Create a SegFormer-B2 model for the ADE20K segmentation task.
    """
    model = create_segformer_b2(num_classes=150)
    if pretrained:
        _load_pretrained_weights_(model, model_urls['ade']['segformer_b2'], progress=progress)
    return model


def segformer_b3_ade(pretrained=True, progress=True):
    """Create a SegFormer-B3 model for the ADE20K segmentation task.
    """
    model = create_segformer_b3(num_classes=150)
    if pretrained:
        _load_pretrained_weights_(model, model_urls['ade']['segformer_b3'], progress=progress)
    return model


def segformer_b4_ade(pretrained=True, progress=True):
    """Create a SegFormer-B4 model for the ADE20K segmentation task.
    """
    model = create_segformer_b4(num_classes=150)
    if pretrained:
        _load_pretrained_weights_(model, model_urls['ade']['segformer_b4'], progress=progress)
    return model


def segformer_b5_ade(pretrained=True, progress=True):
    """Create a SegFormer-B5 model for the ADE20K segmentation task.
    """
    model = create_segformer_b5(num_classes=150)
    if pretrained:
        _load_pretrained_weights_(model, model_urls['ade']['segformer_b5'], progress=progress)
    return model


def segformer_b0_city(pretrained=True, progress=True):
    """Create a SegFormer-B0 model for the CityScapes segmentation task.
    """
    model = create_segformer_b0(num_classes=19)
    if pretrained:
        _load_pretrained_weights_(model, model_urls['city']['segformer_b0'], progress=progress)
    return model


def segformer_b1_city(pretrained=True, progress=True):
    """Create a SegFormer-B1 model for the CityScapes segmentation task.
    """
    model = create_segformer_b1(num_classes=19)
    if pretrained:
        _load_pretrained_weights_(model, model_urls['city']['segformer_b1'], progress=progress)
    return model


def segformer_b2_city(pretrained=True, progress=True):
    """Create a SegFormer-B2 model for the CityScapes segmentation task.
    """
    model = create_segformer_b2(num_classes=19)
    if pretrained:
        _load_pretrained_weights_(model, model_urls['city']['segformer_b2'], progress=progress)
    return model


def segformer_b3_city(pretrained=True, progress=True):
    """Create a SegFormer-B3 model for the CityScapes segmentation task.
    """
    model = create_segformer_b3(num_classes=19)
    if pretrained:
        _load_pretrained_weights_(model, model_urls['city']['segformer_b3'], progress=progress)
    return model


def segformer_b4_city(pretrained=True, progress=True):
    """Create a SegFormer-B4 model for the CityScapes segmentation task.
    """
    model = create_segformer_b4(num_classes=19)
    if pretrained:
        _load_pretrained_weights_(model, model_urls['city']['segformer_b4'], progress=progress)
    return model


def segformer_b5_city(pretrained=True, progress=True):
    """Create a SegFormer-B5 model for the CityScapes segmentation task.
    """
    model = create_segformer_b5(num_classes=19)
    if pretrained:
        _load_pretrained_weights_(model, model_urls['city']['segformer_b5'], progress=progress)
    return model


def segformer_b0(pretrained=False, progress=True, num_classes=150):
    """Create a SegFormer-B0 model.

    Args:
        pretrained: Download backbone weights pretrained on ImageNet data if true.
        progress: Display the download progress of pretrained weights if true.
        num_classes: Number of output classes;.
    """
    model = create_segformer_b0(num_classes=num_classes)
    if pretrained:
        _load_pretrained_weights_(model.backbone, model_urls['imagenet']['segformer_b0'],
                                  progress=progress)
    return model


def segformer_b1(pretrained=False, progress=True, num_classes=150):
    """Create a SegFormer-B1 model.

    Args:
        pretrained: Download backbone weights pretrained on ImageNet data if true.
        progress: Display the download progress of pretrained weights if true.
        num_classes: Number of output classes;.
    """
    model = create_segformer_b1(num_classes=num_classes)
    if pretrained:
        _load_pretrained_weights_(model.backbone, model_urls['imagenet']['segformer_b1'],
                                  progress=progress)
    return model


def segformer_b2(pretrained=False, progress=True, num_classes=150):
    """Create a SegFormer-B2 model.

    Args:
        pretrained: Download backbone weights pretrained on ImageNet data if true.
        progress: Display the download progress of pretrained weights if true.
        num_classes: Number of output classes;.
    """
    model = create_segformer_b2(num_classes=num_classes)
    if pretrained:
        _load_pretrained_weights_(model.backbone, model_urls['imagenet']['segformer_b2'],
                                  progress=progress)
    return model


def segformer_b3(pretrained=False, progress=True, num_classes=150):
    """Create a SegFormer-B3 model.

    Args:
        pretrained: Download backbone weights pretrained on ImageNet data if true.
        progress: Display the download progress of pretrained weights if true.
        num_classes: Number of output classes;.
    """
    model = create_segformer_b3(num_classes=num_classes)
    if pretrained:
        _load_pretrained_weights_(model.backbone, model_urls['imagenet']['segformer_b3'],
                                  progress=progress)
    return model


def segformer_b4(pretrained=False, progress=True, num_classes=150):
    """Create a SegFormer-B4 model.

    Args:
        pretrained: Download backbone weights pretrained on ImageNet data if true.
        progress: Display the download progress of pretrained weights if true.
        num_classes: Number of output classes;.
    """
    model = create_segformer_b4(num_classes=num_classes)
    if pretrained:
        _load_pretrained_weights_(model.backbone, model_urls['imagenet']['segformer_b4'],
                                  progress=progress)
    return model


def segformer_b5(pretrained=False, progress=True, num_classes=150):
    """Create a SegFormer-B5 model.

    Args:
        pretrained: Download backbone weights pretrained on ImageNet data if true.
        progress: Display the download progress of pretrained weights if true.
        num_classes: Number of output classes;.
    """
    model = create_segformer_b5(num_classes=num_classes)
    if pretrained:
        _load_pretrained_weights_(model.backbone, model_urls['imagenet']['segformer_b5'],
                                  progress=progress)
    return model
