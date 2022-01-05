import os.path

import pytest
import torch


@pytest.fixture
def project_root():
    return os.path.dirname(os.path.dirname(__file__))


@pytest.mark.parametrize('name,pretrained', [
    ('segformer_b0', True),
    ('segformer_b1_ade', True),
    ('segformer_b2_city', True),
    ('segformer_b3', False),
    ('segformer_b4_ade', False),
    ('segformer_b5_city', False),
])
def test_torch_hub_load(project_root, name, pretrained):
    model = torch.hub.load(project_root, name, source='local', pretrained=pretrained,
                           progress=False)
    num_classes = 19 if name.endswith('_city') else 150
    x = torch.randn(1, 3, 64, 64)
    y = model(x)
    assert y.shape == (1, num_classes, 64, 64)


def test_num_classes(project_root):
    model = torch.hub.load(project_root, 'segformer_b0', source='local', pretrained=False,
                           num_classes=5)
    x = torch.randn(1, 3, 64, 64)
    y = model(x)
    assert y.shape == (1, 5, 64, 64)
