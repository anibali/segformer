import torch

from segformer.heads import SegFormerHead


class TestSegFormerHead():
    def test_smoke(self):
        head = SegFormerHead(in_channels=[64, 128, 320, 512], num_classes=150, embed_dim=256)
        x = [torch.randn(2, 64, 64, 64),  torch.randn(2, 128, 32, 32),
             torch.randn(2, 320, 16, 16), torch.randn(2, 512, 8, 8)]
        y = head(x)
        assert y.shape == (2, 150, 64, 64)

    def test_rebuild_output_layer_(self):
        head = SegFormerHead(in_channels=[64, 128, 320, 512], num_classes=150, embed_dim=256)
        x = [torch.randn(2, 64, 64, 64),  torch.randn(2, 128, 32, 32),
             torch.randn(2, 320, 16, 16), torch.randn(2, 512, 8, 8)]
        head.rebuild_output_layer_(5)
        y = head(x)
        assert y.shape == (2, 5, 64, 64)
