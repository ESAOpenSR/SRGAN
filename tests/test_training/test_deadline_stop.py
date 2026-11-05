from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from opensr_srgan.utils.training_callbacks import DeadlineStop


def test_absolute_deadline_stops_all_ranks_and_reserves_checkpoint_time(monkeypatch):
    deadline = datetime(2026, 10, 29, 22, tzinfo=timezone.utc)
    callback = DeadlineStop(deadline.isoformat(), 120)
    calls = []
    trainer = SimpleNamespace(should_stop=False, limit_val_batches=1.0,
        strategy=SimpleNamespace(broadcast=lambda value: calls.append(value) or value))
    monkeypatch.setattr("opensr_srgan.utils.training_callbacks.time.time", lambda: deadline.timestamp() - 121)
    callback.on_train_batch_end(trainer, None, None, None, 0)
    assert not trainer.should_stop
    monkeypatch.setattr("opensr_srgan.utils.training_callbacks.time.time", lambda: deadline.timestamp() - 119)
    callback.on_train_batch_end(trainer, None, None, None, 1)
    assert trainer.should_stop and trainer.limit_val_batches == 0 and callback.triggered
    assert calls == [False, True]


def test_deadline_checks_before_first_update_and_rejects_ambiguous_timezone():
    callback = DeadlineStop("2026-10-29T22:00:00+00:00")
    trainer = SimpleNamespace(should_stop=False, limit_val_batches=1.0,
        strategy=SimpleNamespace(broadcast=lambda value: True))
    callback.on_fit_start(trainer, None)
    assert trainer.should_stop
    with pytest.raises(ValueError):
        DeadlineStop("2026-10-29T22:00:00")
