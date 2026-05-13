# opensr_srgan/tests/test_model_summary.py
import torch
import pytest
from types import SimpleNamespace
import yaml
from pathlib import Path
from omegaconf import OmegaConf


from opensr_srgan.utils.model_descriptions import print_model_summary


class DummyNet(torch.nn.Module):
    def __init__(self, n_params=10):
        super().__init__()
        self.layer = torch.nn.Linear(n_params, n_params)
        self.scale = 4
        self.n_blocks = 5
        self.n_layers = 5
        self.base_channels = 64
        self.kernel_size = 3
        self.fc_size = 256

    def forward(self, x):
        return x


# ---- Dummy model container ----
class DummyModel:
    def __init__(self, cfg):
        self.config = cfg
        self.generator = DummyNet()
        self.discriminator = DummyNet()
        self.device = "cpu"
        self.pretrain_g_only = False
        self.g_pretrain_steps = 1000
        self.adv_loss_ramp_steps = 2000
        self.adv_target = 0.9
        self.content_loss_criterion = torch.nn.L1Loss()
        self.adversarial_loss_criterion = torch.nn.BCELoss()


def test_print_model_summary(tmp_path):
    # ---- Load example config ----
    config_path = Path("opensr_srgan/configs/config_10m.yaml")
    conf = OmegaConf.load(config_path)
    model = DummyModel(conf)
    # ---- Run the summary ----
    print_model_summary(model)


def test_print_model_summary_custom_generator_without_optional_fields(capsys):
    conf = OmegaConf.create(
        {
            "Model": {"in_bands": 2},
            "Generator": {"model_type": "custom_net"},
            "Discriminator": {"model_type": "standard"},
            "Training": {
                "Losses": {"content_loss_weight": 2.0, "adv_loss_beta": 0.3}
            },
        }
    )
    model = DummyModel(conf)
    model.generator.scale = None
    model.generator.n_blocks = None
    model.discriminator.n_blocks = None
    model.discriminator.n_layers = None
    model.discriminator.fc_size = None

    print_model_summary(model)

    out = capsys.readouterr().out
    assert "Custom Generator Type: custom_net" in out
    assert "Super-Resolution Factor: Unknown" in out
    assert "Content: 2.0 | Adversarial: 0.3" in out


@pytest.mark.parametrize(
    ("generator_type", "block_type", "expected"),
    [
        ("rrdb", None, "SRResNet (RRDB Dense Residual Blocks)"),
        ("SRResNet", "made_up", "SRResNet (Custom Block Variant: made_up)"),
        ("cgan", None, "Stochastic SRGAN"),
        ("esrgan", None, "ESRGAN (RRDB Residual-in-Residual Dense Network)"),
    ],
)
def test_print_model_summary_generator_descriptions(
    generator_type, block_type, expected, capsys
):
    generator_cfg = {"model_type": generator_type, "scaling_factor": 4}
    if block_type is not None:
        generator_cfg["block_type"] = block_type
    conf = OmegaConf.create(
        {
            "Model": {"in_bands": 3},
            "Generator": generator_cfg,
            "Discriminator": {"model_type": "patchgan"},
            "Training": {"Losses": {"perceptual_loss_weight": 0.1}, "EMA": None},
        }
    )

    print_model_summary(DummyModel(conf))

    assert expected in capsys.readouterr().out


def test_print_model_summary_enabled_ema_details(capsys):
    conf = OmegaConf.create(
        {
            "Model": {"in_bands": 3},
            "Generator": {"model_type": "SRResNet", "scaling_factor": 4},
            "Discriminator": {"model_type": "esrgan"},
            "Training": {
                "EMA": {
                    "enabled": True,
                    "decay": 0.99,
                    "update_after_step": 12,
                    "use_num_updates": False,
                },
                "Losses": {},
            },
        }
    )

    print_model_summary(DummyModel(conf))

    out = capsys.readouterr().out
    assert "Decay:             0.99" in out
    assert "Update After Step: 12" in out
    assert "Use Num Updates:   False" in out
