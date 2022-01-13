import torch
import torch.nn as nn
from einops import rearrange
from torch.nn.functional import interpolate, relu, dropout


class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-5)
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        if self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = relu(x, inplace=True)
        return x


class PointwiseLinear2d(nn.Module):
    def __init__(self, in_features=2048, out_features=768):
        super().__init__()
        self.proj = nn.Linear(in_features, out_features)

    def forward(self, x):
        w = x.shape[-1]
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.proj(x)
        x = rearrange(x, 'b (h w) c -> b c h w', w=w)
        return x


class SegFormerHead(nn.Module):
    def __init__(self, in_channels, num_classes, embed_dim,
                 dropout_p=0.1, align_corners=False):
        super().__init__()

        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.dropout_p = dropout_p
        self.align_corners = align_corners

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        self.linear_c4 = PointwiseLinear2d(c4_in_channels, embed_dim)
        self.linear_c3 = PointwiseLinear2d(c3_in_channels, embed_dim)
        self.linear_c2 = PointwiseLinear2d(c2_in_channels, embed_dim)
        self.linear_c1 = PointwiseLinear2d(c1_in_channels, embed_dim)

        self.linear_fuse = ConvModule(
            in_channels=embed_dim * 4,
            out_channels=embed_dim,
            kernel_size=1,
        )

        self.rebuild_output_layer_(num_classes)

    def rebuild_output_layer_(self, num_classes):
        self.linear_pred = nn.Conv2d(self.embed_dim, num_classes, kernel_size=(1, 1))
        self.num_classes = num_classes

    def forward(self, x):
        c1, c2, c3, c4 = x  # 1/4, 1/8, 1/16, and 1/32 scale features.
        c1_hw = c1.shape[2:]

        c4 = self.linear_c4(c4)
        c4 = interpolate(c4, size=c1_hw, mode='bilinear', align_corners=self.align_corners)

        c3 = self.linear_c3(c3)
        c3 = interpolate(c3, size=c1_hw, mode='bilinear', align_corners=self.align_corners)

        c2 = self.linear_c2(c2)
        c2 = interpolate(c2, size=c1_hw, mode='bilinear', align_corners=self.align_corners)

        c1 = self.linear_c1(c1)

        x = self.linear_fuse(torch.cat([c4, c3, c2, c1], dim=1))
        x = dropout(x, p=self.dropout_p, training=self.training)
        x = self.linear_pred(x)

        return x
