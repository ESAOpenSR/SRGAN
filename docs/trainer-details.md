# Trainer details

This page walks through the control flow that powers adversarial optimisation in OpenSR-SRGAN. It cross-references the exact helper functions in the codebase so you can trace which checks run on every batch, how pretraining vs. adversarial steps are chosen, and how the PyTorch Lightning integration remains compatible with both 1.x and 2.x releases.

## Version-aware bootstrap

1. **Detect the installed Lightning release.** `SRGAN_model` stores the parsed semantic version via `self.pl_version = tuple(int(x) for x in pl.__version__.split("."))`. 【F:opensr_srgan/model/SRGAN.py†L66-L67】
2. **Bind the appropriate training step.** `setup_lightning()` switches between the automatic-optimisation `training_step_PL1` helper (Lightning 1.x) and the manual-optimisation `training_step_PL2` clone (Lightning 2.x) while asserting the required optimisation mode. 【F:opensr_srgan/model/SRGAN.py†L191-L206】
3. **Assemble Trainer keyword arguments.** `build_lightning_kwargs()` mirrors the version choice when it prepares the `Trainer` arguments: pre-2.0 builds receive `resume_from_checkpoint`, whereas 2.x runs use `ckpt_path`. It also normalises device selection (`Training.device`, `Training.gpus`) and strategy flags so multi-GPU training works consistently. 【F:opensr_srgan/utils/build_trainer_kwargs.py†L10-L122】
4. **Resume or continue training.** When `Model.continue_training` points to a checkpoint path the trainer will resume in-place, preserving optimiser state, EMA buffers, and step counters. A fresh run keeps the value at `False`. 【F:opensr_srgan/train.py†L36-L63】

These checks ensure you can retrain a model on Lightning 1.9 or 2.2 with the same configuration file—no manual flag-flipping required.

## Training step anatomy (Lightning 1.x)

The legacy automatic-optimisation path receives `(batch, batch_idx, optimizer_idx)` and splits discriminator and generator logic by the `optimizer_idx` flag:

```python
# opensr_srgan/model/training_step_PL.py
pretrain_phase = self._pretrain_check()
if optimizer_idx == 1:
    self.log("training/pretrain_phase", float(pretrain_phase), sync_dist=True)

if pretrain_phase:
    if optimizer_idx == 1:
        content_loss, metrics = self.content_loss_criterion.return_loss(sr_imgs, hr_imgs)
        self._log_generator_content_loss(content_loss)
        adv_weight = self._compute_adv_loss_weight()
        self._log_adv_loss_weight(adv_weight)
        return content_loss
    else:
        dummy = torch.zeros((), device=device, dtype=dtype, requires_grad=True)
        self.log("discriminator/adversarial_loss", dummy, sync_dist=True)
        return dummy
```

* `_pretrain_check()` compares `self.global_step` against `Training.g_pretrain_steps` to decide whether the generator-only warm-up is active. 【F:opensr_srgan/model/training_step_PL.py†L10-L46】
* The pretraining branch logs the instantaneous adversarial weight even though it stays unused until GAN training begins. This keeps dashboards continuous when you review historical runs.
* The discriminator receives a zero-valued tensor with `requires_grad=True` so Lightning's closure executes without mutating weights. Dummy logs (`discriminator/D(y)_prob`, `discriminator/D(G(x))_prob`) remain pinned to zero for clarity.

Once `_pretrain_check()` flips to `False`, the function splits into discriminator and generator updates:

* **Discriminator (`optimizer_idx == 0`).** Real and fake logits are compared against smoothed targets, and the resulting BCE components are summed into `discriminator/adversarial_loss`. The helper logs running opinions (`discriminator/D(y)_prob`, `discriminator/D(G(x))_prob`) so you can diagnose mode collapse early. 【F:opensr_srgan/model/training_step_PL.py†L52-L98】
* **Generator (`optimizer_idx == 1`).** The generator measures content metrics once, reuses them for logging, queries the adversarial signal (`adversarial_loss_criterion(sr_discriminated, ones)`), and multiplies it with `_adv_loss_weight()` before combining both parts into `generator/total_loss`. 【F:opensr_srgan/model/training_step_PL.py†L100-L133】

## Training step anatomy (Lightning 2.x)

Lightning 2.x requires manual optimisation to alternate between optimisers. `training_step_PL2` mirrors the structure of the 1.x helper but drives the two optimisers explicitly:

```python
# opensr_srgan/model/training_step_PL.py
opt_d, opt_g = self.optimizers()
pretrain_phase = self._pretrain_check()
self.log("training/pretrain_phase", float(pretrain_phase), sync_dist=True)

if pretrain_phase:
    zero = torch.tensor(0.0, device=hr_imgs.device, dtype=hr_imgs.dtype)
    self.log("discriminator/adversarial_loss", zero, sync_dist=True)
    content_loss, metrics = self.content_loss_criterion.return_loss(sr_imgs, hr_imgs)
    self._log_generator_content_loss(content_loss)
    self._log_adv_loss_weight(_adv_weight())
    opt_g.zero_grad(); self.manual_backward(content_loss); opt_g.step()
    if self.ema and self.global_step >= self._ema_update_after_step:
        self.ema.update(self.generator)
    return content_loss
```

The adversarial branch toggles each optimiser in turn, accumulates identical logs to the PL1.x path, and performs the EMA update after every generator step. 【F:opensr_srgan/model/training_step_PL.py†L136-L234】

## Adversarial weight schedule

Both training-step variants call `_adv_loss_weight()` (or `_compute_adv_loss_weight()` in older modules) to retrieve the ramped coefficient that blends the adversarial and content terms. The helper logs `training/adv_loss_weight` so you can confirm whether the ramp has reached its configured `Training.Losses.adv_loss_beta`. During pretraining this value stays at zero; afterwards it climbs toward the configured maximum.

## Retraining and checkpoint flow

When you relaunch an experiment with `Model.continue_training` set to the saved checkpoint path, Lightning restores optimiser states, EMA buffers, and global step counters before the next batch runs. The same logic works on both Lightning branches because the resume argument is threaded through `build_lightning_kwargs()` according to the detected version. 【F:opensr_srgan/train.py†L36-L90】【F:opensr_srgan/utils/build_trainer_kwargs.py†L16-L122】

## Summary of runtime checks

| Check | Source | Purpose |
| --- | --- | --- |
| PyTorch Lightning version | `SRGAN_model.setup_lightning()` | Select PL1 vs. PL2 training-step implementation and toggle manual optimisation. |
| Continue training? | `train.py` (`Model.continue_training`) | Resume checkpoints with schedulers/EMA intact. |
| Pretraining active? | `_pretrain_check()` | Gate between content-only updates and full GAN updates. |
| Adversarial weight value | `_adv_loss_weight()` / `_compute_adv_loss_weight()` | Log instantaneous GAN weight and blend it into `generator/total_loss`. |
| EMA ready to update? | `self.global_step >= self._ema_update_after_step` | Delay shadow-weight updates until the warm-up step threshold. |

Keeping these checkpoints visible in the logs and documentation makes it easy to understand what happens when the trainer toggles between warm-up, adversarial learning, and resumed runs.

## Branch map (full text)

The textual workflow in `opensr_srgan/model/training_workflow.txt` mirrors the branches and logging described above. It is reproduced verbatim below so you can scan the entire decision tree without leaving the docs:

```text
ENTRY: trainer.fit(model, datamodule)
│
├─ PRELUDE (opensr_srgan/train.py)
│   ├─ Load config (OmegaConf) and resolve device list (`Training.gpus`).
│   ├─ Check `Model.load_checkpoint` / `Model.continue_training` to decide between fresh training vs. retraining from a checkpoint.
│   ├─ Call `build_lightning_kwargs()` → detects PyTorch Lightning version, normalises accelerator/devices, and routes resume arguments (`resume_from_checkpoint` for PL<2, `ckpt_path` for PL≥2).
│   └─ Instantiate `Trainer(**trainer_kwargs)` and invoke `trainer.fit(..., **fit_kwargs)`.
│
├─ SRGAN_model.setup_lightning()
│   ├─ Parse `pl.__version__` into `self.pl_version`.
│   ├─ IF `self.pl_version >= (2,0,0)`
│   │     ├─ Set `automatic_optimization = False` (manual optimisation required by PL2).
│   │     └─ Bind `training_step_PL2` as the active `training_step` implementation.
│   └─ ELSE (PL1.x)
│         ├─ Ensure `automatic_optimization is True`.
│         └─ Bind `training_step_PL1` (optimizer_idx-based training).
│
└─ ACTIVE TRAINING STEP (batch, batch_idx[, optimizer_idx])
    │
    ├─ 1) Forward + metrics (no grad for logging reuse)
    │   ├─ (lr, hr) = batch
    │   ├─ sr = G(lr)
    │   └─ metrics = content_loss.return_metrics(sr, hr)
    │       └─ LOG: `train_metrics/*` (L1, SAM, perceptual, TV, PSNR, SSIM)
    │
    ├─ 2) Phase checks
    │   ├─ `pretrain = _pretrain_check()`  # compare global_step vs. `Training.g_pretrain_steps`
    │   ├─ LOG: `training/pretrain_phase` (on G step for PL1, per-batch for PL2)
    │   └─ `adv_weight = _adv_loss_weight()` or `_compute_adv_loss_weight()`  # ramp toward `Training.Losses.adv_loss_beta`
    │       └─ LOG: `training/adv_loss_weight`
    │
    ├─ 3) IF `pretrain` True  (Generator warm-up)
    │   ├─ Generator path
    │   │   ├─ Compute `(content_loss, metrics) = content_loss.return_loss(sr, hr)`
    │   │   ├─ LOG: `generator/content_loss`
    │   │   ├─ Reuse metrics for logging (`train_metrics/*`)
    │   │   ├─ LOG: `training/adv_loss_weight` (even though weight is 0 during warm-up)
    │   │   └─ RETURN/STEP on `content_loss` only (PL1 returns scalar; PL2 manual_backward + `opt_g.step()`)
    │   └─ Discriminator path
    │       ├─ LOG zeros for `discriminator/D(y)_prob`, `discriminator/D(G(x))_prob`, `discriminator/adversarial_loss`
    │       └─ Return dummy zero tensor with `requires_grad=True` (PL1) or skip optimisation but keep logs (PL2)
    │
    └─ 4) ELSE `pretrain` False  (Full GAN training)
        │
        ├─ 4A) Discriminator update
        │   ├─ hr_logits = D(hr)
        │   ├─ sr_logits = D(sr.detach())
        │   ├─ real_target = adv_target (0.9 with label smoothing else 1.0)
        │   ├─ fake_target = 0.0
        │   ├─ loss_real = BCEWithLogits(hr_logits, real_target)
        │   ├─ loss_fake = BCEWithLogits(sr_logits, fake_target)
        │   ├─ d_loss = loss_real + loss_fake
        │   ├─ LOG: `discriminator/adversarial_loss`
        │   ├─ LOG: `discriminator/D(y)_prob` = sigmoid(hr_logits).mean()
        │   ├─ LOG: `discriminator/D(G(x))_prob` = sigmoid(sr_logits).mean()
        │   └─ Optimise D (return `d_loss` in PL1; manual backward + `opt_d.step()` in PL2)
        │
        └─ 4B) Generator update
            ├─ (content_loss, metrics) = content_loss.return_loss(sr, hr)
            ├─ LOG: `generator/content_loss`
            ├─ sr_logits = D(sr)
            ├─ g_adv = BCEWithLogits(sr_logits, target=1.0)
            ├─ LOG: `generator/adversarial_loss` = g_adv
            ├─ total_loss = content_loss + adv_weight * g_adv
            ├─ LOG: `generator/total_loss`
            ├─ Optimise G (return `total_loss` in PL1; manual backward + `opt_g.step()` in PL2)
            └─ IF EMA enabled AND `global_step >= _ema_update_after_step`: update shadow weights (`EMA/update_after_step`, `EMA/is_active` logs)
```
