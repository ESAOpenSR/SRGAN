import matplotlib
import numpy as np
import pytest
import torch
from PIL import Image

from opensr_srgan.utils import logging_helpers

matplotlib.use("Agg", force=True)


def test_to_numpy_img_variants_and_validation():
    gray = torch.tensor([[[0.0, 2.0], [0.5, -1.0]]])
    gray_img = logging_helpers._to_numpy_img(gray)
    assert gray_img.shape == (2, 2)
    assert np.all((gray_img >= 0) & (gray_img <= 1))

    rgb = torch.stack([torch.zeros(2, 2), torch.ones(2, 2), torch.full((2, 2), 0.5)])
    rgb_img = logging_helpers._to_numpy_img(rgb)
    assert rgb_img.shape == (2, 2, 3)
    assert np.allclose(rgb_img[..., 1], 1.0)

    multi = torch.arange(5 * 2 * 2, dtype=torch.float32).view(5, 2, 2)
    multi_img = logging_helpers._to_numpy_img(multi)
    assert multi_img.shape == (2, 2, 3)

    with pytest.raises(ValueError):
        logging_helpers._to_numpy_img(torch.zeros(2, 2))


def test_plot_tensors_returns_pil_image():
    lr = torch.rand(3, 3, 4, 4)
    sr = torch.rand(3, 3, 4, 4)
    hr = torch.rand(3, 3, 4, 4)

    result = logging_helpers.plot_tensors(lr, sr, hr, title="Eval")

    assert isinstance(result, Image.Image)
    assert result.mode == "RGB"
    width, height = result.size
    assert width > 0 and height > 0


def test_plot_tensors_grayscale_batch():
    lr = torch.rand(2, 1, 4, 4)
    sr = torch.rand(2, 1, 4, 4)
    hr = torch.rand(2, 1, 4, 4)

    result = logging_helpers.plot_tensors(lr, sr, hr, title="Gray")

    assert isinstance(result, Image.Image)


def test_plot_tensors_drops_nir_before_stretch(monkeypatch):
    seen_shapes = []

    def fake_stretch(tensor):
        seen_shapes.append(tuple(tensor.shape))
        return tensor

    monkeypatch.setattr(logging_helpers, "minmax_percentile", fake_stretch)

    rgbnir = torch.zeros(1, 4, 4, 4)
    rgbnir[:, 3] = 10000.0

    result = logging_helpers.plot_tensors(rgbnir, rgbnir, rgbnir, title="RGB-NIR")

    assert isinstance(result, Image.Image)
    assert seen_shapes == [(1, 3, 4, 4), (1, 3, 4, 4), (1, 3, 4, 4)]
