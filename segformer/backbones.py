# This file incorporates work from https://github.com/NVlabs/SegFormer which is covered by the
# following copyright and permission notice:
#
# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------

import math
from typing import Tuple

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn.functional import dropout, gelu

from segformer.timm import DropPath, to_2tuple, trunc_normal_

Tuple4i = Tuple[int, int, int, int]


def _init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
    elif isinstance(m, nn.Conv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        fan_out //= m.groups
        m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if m.bias is not None:
            m.bias.data.zero_()


class MixFeedForward(nn.Module):
    def __init__(self, in_features, out_features, hidden_features, dropout_p=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        # Depth-wise convolution
        self.conv = nn.Conv2d(hidden_features, hidden_features, (3, 3), padding=(1, 1),
                              bias=True, groups=hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout_p = dropout_p

    def forward(self, x, h, w):
        x = self.fc1(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.conv(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = gelu(x)
        x = dropout(x, p=self.dropout_p, training=self.training)
        x = self.fc2(x)
        x = dropout(x, p=self.dropout_p, training=self.training)
        return x


class EfficientAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, dropout_p=0.0, sr_ratio=1):
        super().__init__()

        if dim % num_heads != 0:
            raise ValueError(f'expected dim {dim} to be a multiple of num_heads {num_heads}.')

        self.dim = dim
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.dropout_p = dropout_p

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            sr_ratio_tuple = (sr_ratio, sr_ratio)
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio_tuple, stride=sr_ratio_tuple)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, h, w):
        q = self.q(x)
        q = rearrange(q, ('b hw (m c) -> b m hw c'), m=self.num_heads)

        if self.sr_ratio > 1:
            x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
            x = self.sr(x)
            x = rearrange(x, 'b c h w -> b (h w) c')
            x = self.norm(x)

        x = self.kv(x)
        x = rearrange(x, 'b d (a m c) -> a b m d c', a=2, m=self.num_heads)
        k, v = x.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = attn @ v
        x = rearrange(x, 'b m hw c -> b hw (m c)')
        x = self.proj(x)
        x = dropout(x, p=self.dropout_p, training=self.training)

        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4, qkv_bias=False, dropout_p=0.0,
                 drop_path_p=0.0, sr_ratio=1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = EfficientAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                       dropout_p=dropout_p, sr_ratio=sr_ratio)
        self.drop_path = DropPath(drop_path_p)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = MixFeedForward(dim, dim, hidden_features=dim * mlp_ratio, dropout_p=dropout_p)

    def forward(self, x, h, w):
        x = x + self.drop_path(self.attn(self.norm1(x), h, w))
        x = x + self.drop_path(self.mlp(self.norm2(x), h, w))
        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, patch_size: int, stride: int, in_chans: int, embed_dim: int):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)

        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, h, w = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.norm(x)
        return x, h, w


class MixTransformer(nn.Module):
    def __init__(
        self,
        in_chans: int = 3,
        embed_dims: Tuple4i = (64, 128, 256, 512),
        num_heads: Tuple4i = (1, 2, 4, 8),
        mlp_ratios: Tuple4i = (4, 4, 4, 4),
        qkv_bias: bool = False,
        dropout_p: float = 0.0,
        drop_path_p: float = 0.0,
        depths: Tuple4i = (3, 4, 6, 3),
        sr_ratios: Tuple4i = (8, 4, 2, 1),
    ):
        super().__init__()
        self.depths = depths

        # Patch embedding layers
        self.patch_embed1 = OverlapPatchEmbed(patch_size=7, stride=4, in_chans=in_chans,
                                              embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])

        # Transformer encoder blocks
        dpr = torch.linspace(0, drop_path_p, sum(depths)).tolist()  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList([TransformerBlock(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias,
            dropout_p=dropout_p, drop_path_p=dpr[cur + i],
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        self.norm1 = nn.LayerNorm(embed_dims[0], eps=1e-6)

        cur += depths[0]
        self.block2 = nn.ModuleList([TransformerBlock(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias,
            dropout_p=dropout_p, drop_path_p=dpr[cur + i],
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        self.norm2 = nn.LayerNorm(embed_dims[1], eps=1e-6)

        cur += depths[1]
        self.block3 = nn.ModuleList([TransformerBlock(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias,
            dropout_p=dropout_p, drop_path_p=dpr[cur + i],
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        self.norm3 = nn.LayerNorm(embed_dims[2], eps=1e-6)

        cur += depths[2]
        self.block4 = nn.ModuleList([TransformerBlock(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias,
            dropout_p=dropout_p, drop_path_p=dpr[cur + i],
            sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        self.norm4 = nn.LayerNorm(embed_dims[3], eps=1e-6)

        self.init_weights()

    def init_weights(self):
        self.apply(_init_weights)

    def _forward_stage(self, x, patch_embed, block, norm):
        x, h, w = patch_embed(x)
        for i, blk in enumerate(block):
            x = blk(x, h, w)
        x = norm(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        return x

    def forward(self, x):
        c1 = self._forward_stage(x, self.patch_embed1, self.block1, self.norm1)
        c2 = self._forward_stage(c1, self.patch_embed2, self.block2, self.norm2)
        c3 = self._forward_stage(c2, self.patch_embed3, self.block3, self.norm3)
        c4 = self._forward_stage(c3, self.patch_embed4, self.block4, self.norm4)
        return c1, c2, c3, c4


def _mit_bx(embed_dims: Tuple4i, depths: Tuple4i) -> MixTransformer:
    return MixTransformer(
        embed_dims=embed_dims,
        num_heads=(1, 2, 5, 8),
        mlp_ratios=(4, 4, 4, 4),
        qkv_bias=True,
        depths=depths,
        sr_ratios=(8, 4, 2, 1),
        dropout_p=0.0,
        drop_path_p=0.1,
    )


def mit_b0():
    return _mit_bx(embed_dims=(32, 64, 160, 256), depths=(2, 2, 2, 2))


def mit_b1():
    return _mit_bx(embed_dims=(64, 128, 320, 512), depths=(2, 2, 2, 2))


def mit_b2():
    return _mit_bx(embed_dims=(64, 128, 320, 512), depths=(3, 4, 6, 3))


def mit_b3():
    return _mit_bx(embed_dims=(64, 128, 320, 512), depths=(3, 4, 18, 3))


def mit_b4():
    return _mit_bx(embed_dims=(64, 128, 320, 512), depths=(3, 8, 27, 3))


def mit_b5():
    return _mit_bx(embed_dims=(64, 128, 320, 512), depths=(3, 6, 40, 3))
