from __future__ import annotations

import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

hydra = pytest.importorskip("hydra")
from hydra.core.global_hydra import GlobalHydra

CONFIG_DIR = Path(__file__).resolve().parents[2] / "opensr_srgan" / "configs" / "hydra"
CONSUMED_SECTIONS = (
    "Data",
    "Model",
    "Training",
    "Generator",
    "Discriminator",
    "Optimizers",
    "Schedulers",
    "Logging",
)


@pytest.fixture(autouse=True)
def clear_hydra_state():
    GlobalHydra.instance().clear()
    yield
    GlobalHydra.instance().clear()


def _compose(overrides: list[str] | None = None, *, include_hydra: bool = False):
    with hydra.initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.3"):
        return hydra.compose(
            config_name="train",
            overrides=overrides or [],
            return_hydra_config=include_hydra,
        )


def _consumed_sections(cfg):
    return {
        section: OmegaConf.to_container(cfg[section], resolve=True)
        for section in CONSUMED_SECTIONS
    }


def _legacy_sections(path: str):
    return _consumed_sections(OmegaConf.load(path))


def test_default_hydra_config_is_cpu_safe_and_uses_example_dataset():
    cfg = _compose()

    assert cfg.Data.dataset_type == "ExampleDataset"
    assert cfg.Training.device == "cpu"
    assert cfg.Training.gpus == []
    assert cfg.Logging.wandb.enabled is False


def test_hydra_config_sets_run_directory_without_changing_cwd():
    cfg = _compose(include_hydra=True)

    assert cfg.hydra.job.chdir is False
    run_cfg = OmegaConf.to_container(cfg.hydra.run, resolve=False)
    assert run_cfg["dir"] == ("logs/${Logging.wandb.project}/${now:%Y-%m-%d_%H-%M-%S}")


@pytest.mark.parametrize(
    ("experiment", "legacy_path"),
    [
        ("example", "opensr_srgan/configs/config_training_example.yaml"),
        ("10m", "opensr_srgan/configs/config_10m.yaml"),
        ("20m", "opensr_srgan/configs/config_20m.yaml"),
    ],
)
def test_hydra_experiments_match_legacy_yaml_sections(experiment, legacy_path):
    cfg = _compose([f"experiment={experiment}"])

    assert _consumed_sections(cfg) == _legacy_sections(legacy_path)


def test_hydra_entrypoint_passes_composed_overrides_to_train(monkeypatch, tmp_path):
    import opensr_srgan.train as train_module
    from opensr_srgan import train_hydra

    captured = {}

    def fake_train(cfg):
        captured["cfg"] = cfg

    monkeypatch.setattr(train_module, "train", fake_train)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "srgan-train",
            "experiment=example",
            "Training.max_epochs=2",
            "Logging.wandb.enabled=false",
            f"hydra.run.dir={tmp_path.as_posix()}",
        ],
    )

    train_hydra.main()

    cfg = captured["cfg"]
    assert cfg.Data.dataset_type == "ExampleDataset"
    assert cfg.Training.max_epochs == 2
    assert cfg.Logging.wandb.enabled is False
    assert Path(cfg.Logging.output_dir) == tmp_path
