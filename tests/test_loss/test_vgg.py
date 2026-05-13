# opensr_srgan/tests/test_truncated_vgg19.py
import torch
import pytest

from opensr_srgan.model.loss.vgg import TruncatedVGG19


def test_truncated_vgg19_valid_layers():
    model = TruncatedVGG19(i=5, j=4, weights=False)
    model.eval()
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        y = model(x)
    assert isinstance(y, torch.Tensor)
    assert y.ndim == 4
    # typical VGG feature map shrinkage
    assert y.shape[2] <= 64 and y.shape[3] <= 64


def test_truncated_vgg19_invalid_layers():
    # impossible layer combination should raise
    with pytest.raises(ValueError):
        TruncatedVGG19(i=10, j=10)


def test_truncated_vgg19_forward_consistency():
    # different i,j give different truncation depth
    model1 = TruncatedVGG19(i=2, j=2)
    model2 = TruncatedVGG19(i=5, j=4)
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        y1, y2 = model1(x), model2(x)
    # deeper truncation should produce smaller spatial size
    assert y2.shape[2] <= y1.shape[2]
    assert y2.shape[3] <= y1.shape[3]
