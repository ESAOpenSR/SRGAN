from types import SimpleNamespace

import torch

from opensr_srgan.data.utils.normalizer import Normalizer
from opensr_srgan.tests.helpers import custom_norms


def _build_cfg(value):
    return SimpleNamespace(Data=SimpleNamespace(normalization=value))


def test_alias_resolves_reflectance_signed():
    cfg = _build_cfg("reflectance_signed")
    normalizer = Normalizer(cfg)

    assert normalizer.method == "normalise_10k_signed"

    tensor = torch.tensor([0.0, 5000.0])
    normalized = normalizer.normalize(tensor)
    restored = normalizer.denormalize(normalized)

    assert torch.all(normalized <= 1) and torch.all(normalized >= -1)
    assert torch.allclose(restored, tensor, atol=1e-5)


def test_custom_strategy_roundtrip_via_import_path():
    cfg = _build_cfg(
        {
            "name": "custom",
            "normalize": "opensr_srgan.tests.helpers.custom_norms:halve",
            "denormalize": "opensr_srgan.tests.helpers.custom_norms:double",
        }
    )

    normalizer = Normalizer(cfg)
    x = torch.ones(4) * 8
    y = normalizer.normalize(x)
    z = normalizer.denormalize(y)

    assert normalizer.method == "custom"
    assert torch.allclose(y, torch.ones_like(x) * 4)
    assert torch.allclose(z, x)


def test_custom_strategy_accepts_callables():
    cfg = _build_cfg(
        {
            "name": "custom",
            "normalize": custom_norms.halve,
            "denormalize": custom_norms.double,
        }
    )

    normalizer = Normalizer(cfg)
    x = torch.tensor([10.0])
    assert torch.allclose(normalizer.denormalize(normalizer.normalize(x)), x)


def test_custom_strategy_supports_kwargs():
    cfg = _build_cfg(
        {
            "name": "custom",
            "normalize": "opensr_srgan.utils.radiometrics:normalise_10k",
            "normalize_kwargs": {"stage": "norm"},
            "denormalize": "opensr_srgan.utils.radiometrics:normalise_10k",
            "denormalize_kwargs": {"stage": "denorm"},
        }
    )

    normalizer = Normalizer(cfg)
    x = torch.tensor([0.0, 10000.0])
    assert torch.allclose(normalizer.denormalize(normalizer.normalize(x)), x)


def test_available_methods_list_standard_entries():
    methods = Normalizer.available_methods()
    expected = {"normalise_10k", "normalise_10k_signed", "zero_one", "zero_one_signed"}
    assert expected.issubset(set(methods))


def test_identity_alias_none_behaves_like_passthrough():
    cfg = _build_cfg("none")
    normalizer = Normalizer(cfg)

    x = torch.randn(3, 3)
    assert normalizer.method == "identity"
    assert torch.allclose(normalizer.normalize(x), x)
    assert torch.allclose(normalizer.denormalize(x), x)
