import torch
import torch.nn as nn
from torch.nn.functional import interpolate, relu, dropout


class SegFormerHead(nn.Module):
    def __init__(self, in_channels, num_classes, embed_dim, dropout_p=0.1, align_corners=False):
        super().__init__()

        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.dropout_p = dropout_p
        self.align_corners = align_corners

        self.layers = nn.ModuleList([nn.Conv2d(chans, embed_dim, (1, 1))
                                     for chans in reversed(in_channels)])
        self.linear_fuse = nn.Conv2d(embed_dim * len(self.layers), embed_dim, (1, 1), bias=False)
        self.bn = nn.BatchNorm2d(embed_dim, eps=1e-5)
        self.rebuild_output_layer_(num_classes)

        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.linear_fuse.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def rebuild_output_layer_(self, num_classes):
        self.linear_pred = nn.Conv2d(self.embed_dim, num_classes, kernel_size=(1, 1))
        self.num_classes = num_classes

    def forward(self, x):
        feats_hw = x[0].shape[2:]
        x = [layer(xi) for layer, xi in zip(self.layers, reversed(x))]
        x = [interpolate(xi, size=feats_hw, mode='bilinear', align_corners=self.align_corners)
             for xi in x[:-1]] + [x[-1]]
        x = self.linear_fuse(torch.cat(x, dim=1))
        x = self.bn(x)
        x = relu(x, inplace=True)
        x = dropout(x, p=self.dropout_p, training=self.training)
        x = self.linear_pred(x)
        return x
