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


def test_save_validation_artifacts_writes_fixed_rgb_and_nir_panels(tmp_path):
    lr = torch.linspace(0, 1, 4 * 2 * 2).view(1, 4, 2, 2)
    sr = torch.linspace(0, 1, 4 * 4 * 4).view(1, 4, 4, 4)
    hr = torch.flip(sr, dims=(-1,))

    paths = logging_helpers.save_validation_artifacts(
        lr, sr, hr, tmp_path, epoch=7, num_images=1, sample_offset=12
    )

    assert [path.name for path in paths] == [
        "epoch-0007_sample-012_rgb.png",
        "epoch-0007_sample-012_nir.png",
    ]
    assert all(path.is_file() for path in paths)
    assert all(Image.open(path).mode == "RGB" for path in paths)


def test_save_validation_artifacts_is_disabled_without_local_rank_zero(tmp_path):
    tensors = torch.rand(2, 4, 2, 2)
    assert (
        logging_helpers.save_validation_artifacts(
            tensors,
            tensors,
            tensors,
            tmp_path,
            epoch=1,
            num_images=2,
            local_images=False,
        )
        == []
    )
    assert not (tmp_path / "validation_images").exists()
    assert (
        logging_helpers.save_validation_artifacts(
            tensors,
            tensors,
            tensors,
            tmp_path,
            epoch=1,
            num_images=2,
            is_global_zero=False,
        )
        == []
    )


def test_save_validation_artifacts_accepts_bfloat16(tmp_path):
    lr = torch.rand(1, 4, 2, 2).bfloat16()
    sr = torch.rand(1, 4, 4, 4).bfloat16()
    hr = torch.rand(1, 4, 4, 4).bfloat16()

    paths = logging_helpers.save_validation_artifacts(
        lr, sr, hr, tmp_path, epoch=1, num_images=1
    )

    assert len(paths) == 2
    assert all(path.is_file() for path in paths)


def test_matched_panels_keep_raw_and_corrected_in_same_row(monkeypatch, tmp_path):
    seen = []
    def panel(images, labels, title):
        seen.append((images, labels))
        return Image.new("RGB", (100 * len(images), 100))
    monkeypatch.setattr(logging_helpers, "_panel_image", panel)
    lr = torch.full((1, 4, 2, 3), 0.2)
    sr = torch.full((1, 4, 4, 6), 0.7)
    matched = torch.full_like(sr, 0.2)
    hr = torch.full_like(sr, 0.6)
    logging_helpers.save_validation_artifacts(lr, sr, hr, tmp_path, 1, 1, matched=matched)
    assert len(seen) == 2  # RGB and NIR
    for images, labels in seen:
        assert labels == ["LR", "Bicubic", "SR", "SR matched", "HR"]
        assert torch.all(images[2] == 0.7)
        assert torch.all(images[3] == 0.2)
    seen.clear()
    image = logging_helpers.plot_tensors(lr, sr, hr, matched=matched)
    assert image.size == (400, 100)
    assert seen[0][1] == ["LR", "SR", "SR matched", "HR"]


def test_matched_panel_uses_shared_stretch(monkeypatch):
    seen = []
    original = logging_helpers._to_numpy_img
    def capture(image):
        seen.append(image.clone())
        return original(image)
    monkeypatch.setattr(logging_helpers, "_to_numpy_img", capture)
    lr = torch.full((1, 3, 2, 2), 0.2)
    sr = torch.full((1, 3, 4, 4), 0.8)
    result = logging_helpers.plot_tensors(lr, sr, lr, matched=lr)
    assert isinstance(result, Image.Image)
    assert seen[1].mean() > seen[2].mean() + 0.9


def test_validation_panel_includes_ema_and_live_in_one_shared_display(monkeypatch, tmp_path):
    seen = []
    def panel(images, labels, title):
        seen.append((images, labels))
        return Image.new('RGB', (10, 10))
    monkeypatch.setattr(logging_helpers, '_panel_image', panel)
    lr = torch.zeros(1, 4, 2, 2)
    ema = torch.full((1, 4, 4, 4), 0.2)
    live = torch.full_like(ema, 0.8)
    paths = logging_helpers.save_validation_artifacts(lr, ema, torch.ones_like(ema),
        tmp_path, 0, 1, live_sr=live)
    assert len(paths) == 2 and len(seen) == 2
    for images, labels in seen:
        assert labels == ['LR', 'Bicubic', 'SR EMA', 'SR live (non-EMA)', 'HR']
        torch.testing.assert_close(images[2], torch.full_like(images[2], 0.2))
        torch.testing.assert_close(images[3], torch.full_like(images[3], 0.8))


def test_live_and_matched_columns_are_both_retained(monkeypatch, tmp_path):
    panels = []
    def panel(images, labels, title):
        panels.append(labels)
        return Image.new('RGB', (16, 16))
    monkeypatch.setattr(logging_helpers, '_panel_image', panel)
    lr = torch.zeros(1, 4, 2, 2)
    sr = torch.zeros(1, 4, 8, 8)
    paths = logging_helpers.save_validation_artifacts(
        lr, sr, sr, tmp_path, epoch=0, num_images=1,
        matched=sr + .1, live_sr=sr + .2,
    )
    assert len(paths) == len(panels) == 2
    assert all(labels == ['LR', 'Bicubic', 'SR EMA', 'SR matched', 'SR live (non-EMA)', 'HR'] for labels in panels)
