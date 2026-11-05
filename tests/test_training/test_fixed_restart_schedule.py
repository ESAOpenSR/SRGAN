from types import SimpleNamespace

import pytest
import torch

from opensr_srgan.model.SRGAN import SRGAN_model
from opensr_srgan.utils.training_callbacks import FixedRestartSchedule


def schedule(peak=5e-5):
    return FixedRestartSchedule(start_optimizer_step=600000, peak_g_lr=peak,
                                floor_g_lr=1e-6)


def test_fixed_schedule_endpoints_hold_cosine_and_resume_position():
    cb = schedule()
    assert cb.generator_lr(600000) == pytest.approx(1e-6)
    assert cb.generator_lr(600500) == pytest.approx(25.5e-6)
    assert cb.generator_lr(601000) == pytest.approx(5e-5)
    assert cb.generator_lr(620000) == pytest.approx(5e-5)
    assert cb.generator_lr(640000) == pytest.approx(25.5e-6)
    assert cb.generator_lr(660000) == pytest.approx(1e-6)
    assert schedule().generator_lr(640000) == cb.generator_lr(640000)
    with pytest.raises(ValueError):
        cb.generator_lr(599998)


def test_fixed_schedule_updates_both_optimizers_after_restore_without_resetting_adam():
    optimizers = [torch.optim.Adam([torch.nn.Parameter(torch.ones(1))], lr=1e-6)
                  for _ in range(2)]
    for opt in optimizers:
        opt.param_groups[0]["params"][0].grad = torch.ones(1)
        opt.step()
    moments = [next(iter(opt.state.values()))["exp_avg"].clone() for opt in optimizers]
    schedulers = [SimpleNamespace(last_epoch=100) for _ in range(3)]
    model = SimpleNamespace(_plateau_scheduler_d=schedulers[0],
                            _plateau_scheduler_g=schedulers[1],
                            _warmup_scheduler_g=schedulers[2], _generator_warmup_steps=100)
    trainer = SimpleNamespace(global_step=620000, optimizers=optimizers)
    cb = schedule()
    cb.on_train_start(trainer, model)
    assert optimizers[0].param_groups[0]["lr"] == pytest.approx(5e-6)
    assert optimizers[1].param_groups[0]["lr"] == pytest.approx(5e-5)
    for index, opt in enumerate(optimizers):
        torch.testing.assert_close(next(iter(opt.state.values()))["exp_avg"], moments[index])


def test_inherited_plateau_metric_cannot_change_fixed_restart_rates():
    def fail(*args):
        pytest.fail("The inherited plateau scheduler must not step")
    model = SimpleNamespace(
        config=SimpleNamespace(Schedulers=SimpleNamespace(
            fixed_restart=SimpleNamespace(enabled=True))),
        trainer=SimpleNamespace(sanity_checking=False),
        _plateau_scheduler_g=SimpleNamespace(step=fail))
    SRGAN_model._step_plateau_schedulers(model)
