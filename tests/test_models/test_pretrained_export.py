"""Release-boundary tests for EMA-only inference checkpoint exports."""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from opensr_srgan import pretrained

_DEFAULT = object()
_MISSING = object()


class _TinyGenerator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(2, 3))
        self.register_buffer("counter", torch.tensor(0, dtype=torch.long))


class _TinyInferenceModel:
    def __init__(self, config, mode) -> None:
        assert mode == "eval"
        self.generator = _TinyGenerator()


def _trusted_checkpoint(path, *, ema_state=_DEFAULT, state_dict=_DEFAULT) -> None:
    if ema_state is _DEFAULT:
        ema_state = {
            "decay": 0.999,
            "num_updates": 12,
            "shadow_params": {"weight": torch.full((2, 3), 3.0)},
            "shadow_buffers": {"counter": torch.tensor(4, dtype=torch.long)},
        }
    if state_dict is _DEFAULT:
        state_dict = {"generator.weight": torch.zeros(2, 3)}
    checkpoint = {"state_dict": state_dict}
    if ema_state is not _MISSING:
        checkpoint["ema_state"] = ema_state
    torch.save(checkpoint, path)


def test_export_discovers_release_config_next_to_destination(monkeypatch, tmp_path):
    source = tmp_path / "training" / "full.ckpt"
    source.parent.mkdir()
    _trusted_checkpoint(source)
    target = tmp_path / "release" / "weights" / "inference.ckpt"
    config = target.parent / "config.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("model: tiny\n")
    monkeypatch.setattr(pretrained, "SRGAN_model", _TinyInferenceModel)

    metadata = pretrained.export_inference_checkpoint(source, target)
    artifact = torch.load(target, map_location="cpu", weights_only=True)

    assert metadata["config_sha256"] == pretrained._sha256_file(config)
    assert metadata["generator_state_sha256"] == pretrained._sha256_state_dict(
        artifact["state_dict"]
    )
    assert artifact["metadata"]["format"] == "opensr-srgan-ema-generator-v1"


def test_export_requires_an_existing_config_and_full_checkpoint(monkeypatch, tmp_path):
    with pytest.raises(FileNotFoundError, match="Source checkpoint"):
        pretrained.export_ema_generator_checkpoint(
            tmp_path / "missing.ckpt", tmp_path / "model.ckpt"
        )

    source = tmp_path / "full.ckpt"
    _trusted_checkpoint(source)
    monkeypatch.setattr(pretrained, "SRGAN_model", _TinyInferenceModel)

    with pytest.raises(ValueError, match="generator config is required"):
        pretrained.export_ema_generator_checkpoint(source, tmp_path / "model.ckpt")
    with pytest.raises(FileNotFoundError, match="could not be located"):
        pretrained.export_ema_generator_checkpoint(
            source, tmp_path / "model.ckpt", config_path=tmp_path / "missing.yaml"
        )

    config = tmp_path / "config.yaml"
    config.write_text("model: tiny\n")
    torch.save({"ema_state": {}}, source)
    with pytest.raises(ValueError, match="full Lightning checkpoint"):
        pretrained.export_ema_generator_checkpoint(source, tmp_path / "model.ckpt", config_path=config)


def test_export_never_replaces_the_trusted_checkpoint(monkeypatch, tmp_path):
    source = tmp_path / "full.ckpt"
    _trusted_checkpoint(source)
    config = tmp_path / "config.yaml"
    config.write_text("model: tiny\n")
    monkeypatch.setattr(pretrained, "SRGAN_model", _TinyInferenceModel)

    with pytest.raises(ValueError, match="Destination must differ"):
        pretrained.export_ema_generator_checkpoint(source, source, config_path=config)


@pytest.mark.parametrize(
    ("ema_state", "message"),
    [
        (_MISSING, "has no ema_state"),
        ({"shadow_params": {}, "shadow_buffers": []}, "must contain"),
        (
            {"shadow_params": {"weight": "not-a-tensor"}, "shadow_buffers": {}},
            "non-tensor entry",
        ),
        (
            {
                "shadow_params": {"weight": torch.zeros(2, 3)},
                "shadow_buffers": {"weight": torch.zeros(2, 3)},
            },
            "duplicates",
        ),
    ],
)
def test_export_rejects_malformed_ema_state(monkeypatch, tmp_path, ema_state, message):
    source = tmp_path / "full.ckpt"
    _trusted_checkpoint(source, ema_state=ema_state)
    config = tmp_path / "config.yaml"
    config.write_text("model: tiny\n")
    monkeypatch.setattr(pretrained, "SRGAN_model", _TinyInferenceModel)

    with pytest.raises(ValueError, match=message):
        pretrained.export_ema_generator_checkpoint(
            source, tmp_path / "model.ckpt", config_path=config
        )


@pytest.mark.parametrize(
    ("state_dict", "message"),
    [
        ({"generator.weight": torch.zeros(2, 3)}, "missing: generator.counter"),
        (
            {
                "generator.weight": torch.zeros(2, 3),
                "generator.counter": torch.tensor(0),
                "generator.extra": torch.zeros(1),
            },
            "unexpected: generator.extra",
        ),
        (
            {
                "generator.weight": torch.zeros(2, 2),
                "generator.counter": torch.tensor(0),
            },
            "has shape",
        ),
        (
            {
                "generator.weight": torch.zeros(2, 3, dtype=torch.float64),
                "generator.counter": torch.tensor(0),
            },
            "has dtype",
        ),
        (
            {
                "generator.weight": torch.full((2, 3), math.nan),
                "generator.counter": torch.tensor(0),
            },
            "non-finite",
        ),
    ],
)
def test_validate_ema_generator_state_rejects_unloadable_artifacts(state_dict, message):
    with pytest.raises(ValueError, match=message):
        pretrained.validate_ema_generator_state(state_dict, _TinyInferenceModel(None, "eval"))


def test_validate_ema_generator_state_rejects_non_tensor_values():
    state_dict = {
        "generator.weight": "not-a-tensor",
        "generator.counter": torch.tensor(0),
    }
    with pytest.raises(ValueError, match="not a tensor"):
        pretrained.validate_ema_generator_state(state_dict, _TinyInferenceModel(None, "eval"))


def test_export_cli_parses_paths_and_reports_artifact(monkeypatch, tmp_path, capsys):
    captured = {}

    def fake_export(source, destination, *, config_path):
        captured.update(source=source, destination=destination, config_path=config_path)
        return {"generator_state_sha256": "digest"}

    monkeypatch.setattr(pretrained, "export_ema_generator_checkpoint", fake_export)
    source = tmp_path / "full.ckpt"
    destination = tmp_path / "inference.ckpt"
    config = tmp_path / "config.yaml"

    assert pretrained._main([str(source), str(destination), "--config", str(config)]) == 0
    assert captured == {
        "source": source,
        "destination": destination,
        "config_path": config,
    }
    assert f"Exported {destination} (digest)" in capsys.readouterr().out
