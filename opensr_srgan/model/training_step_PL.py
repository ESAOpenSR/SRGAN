import torch


def training_step_PL2(self, batch, batch_idx):
    """Manual-optimization training step for PyTorch Lightning >= 2.

    Performs two explicit optimizer updates per batch:
    - **Pretraining phase**: Discriminator logs dummies; Generator is optimized with
      hardwired L1 loss only (no adversarial term), and EMA optionally updates.
    - **Adversarial phase**: Performs a Discriminator step (real vs. fake BCE),
      followed by a Generator step (content + λ_adv · BCE against ones).

    Assumptions:
        - `self.automatic_optimization` is `False` (manual opt).
        - `configure_optimizers()` returns optimizers in order `[opt_d, opt_g]`.
        - EMA updates occur after `self._ema_update_after_step`.

    Args:
        batch (Tuple[torch.Tensor, torch.Tensor]): `(lr_imgs, hr_imgs)` tensors with shape `(B, C, H, W)`.
        batch_idx (int): Index of the current batch.

    Returns:
        torch.Tensor:
            - **Pretraining**: content loss (Generator-only step).
            - **Adversarial**: total generator loss = content + λ_adv · BCE(G).

    Logged metrics (selection):
        - `"training/pretrain_phase"` (0/1)
        - `"train_metrics/*"` (from content criterion)
        - `"generator/content_loss"`, `"generator/adversarial_loss"`, `"generator/total_loss"`
        - `"discriminator/adversarial_loss"`, `"discriminator/D(y)_prob"`, `"discriminator/D(G(x))_prob"`
        - `"training/adv_loss_weight"` (λ_adv from ramp schedule)
    """
    if self.automatic_optimization is not False:
        raise RuntimeError("training_step_PL2 requires manual optimization.")

    # -------- CREATE SR DATA --------
    lr_imgs, hr_imgs = batch
    sr_imgs = self.forward(lr_imgs)
    use_wasserstein = getattr(self, "adv_loss_type", "gan") == "wasserstein"
    use_relativistic = bool(getattr(self, "relativistic_average_d", False))

    # --- helper to resolve adv-weight function name mismatches ---
    def _adv_weight():
        if hasattr(self, "_adv_loss_weight"):
            return self._adv_loss_weight()
        return self._compute_adv_loss_weight()

    # fetch optimizers (expects two)
    opt_d, opt_g = self.optimizers()

    # optional gradient clipping support (norm-based)
    # Keep this aligned with build_lightning_kwargs, which reads Optimizers.gradient_clip_val.
    optim_cfg = getattr(self.config, "Optimizers", None)
    sched_cfg = getattr(self.config, "Schedulers", None)
    gradient_clip_val = float(
        getattr(
            optim_cfg,
            "gradient_clip_val",
            getattr(sched_cfg, "gradient_clip_val", 0.0),
        )
    )

    def _maybe_clip_gradients(module, optimizer=None):
        if gradient_clip_val > 0.0 and module is not None:
            precision_plugin = getattr(self.trainer, "precision_plugin", None)
            if (
                optimizer is not None
                and precision_plugin is not None
                and hasattr(precision_plugin, "unscale_optimizer")
            ):
                precision_plugin.unscale_optimizer(optimizer)
            torch.nn.utils.clip_grad_norm_(module.parameters(), gradient_clip_val)

    # ======================================================================
    # SECTION: Pretraining phase gate
    # ======================================================================
    pretrain_phase = self._pretrain_check()
    self.log(
        "training/pretrain_phase", float(pretrain_phase), prog_bar=False, sync_dist=True
    )

    # ======================================================================
    # SECTION: Pretraining branch (L1-only on G; D logs dummies)
    # ======================================================================
    if pretrain_phase:
        # --- D dummy logs (no step during pretraining) ---
        with torch.no_grad():
            zero = torch.tensor(0.0, device=hr_imgs.device, dtype=hr_imgs.dtype)
            self.log("discriminator/D(y)_prob", zero, prog_bar=True, sync_dist=True)
            self.log("discriminator/D(G(x))_prob", zero, prog_bar=True, sync_dist=True)
            self.log(
                "discriminator/adversarial_loss",
                zero,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
            )

        # --- G step: hardwired L1-only pretraining loss ---
        content_loss = torch.nn.functional.l1_loss(sr_imgs, hr_imgs)
        metrics = {"l1": content_loss.detach()}
        self._log_generator_content_loss(content_loss)
        for key, value in metrics.items():
            self.log(f"train_metrics/{key}", value, sync_dist=True)

        # ensure adv-weight is still logged like in pretrain
        self._log_adv_loss_weight(_adv_weight())

        # manual optimize G
        if hasattr(self, "toggle_optimizer"):
            self.toggle_optimizer(opt_g)
        opt_g.zero_grad()
        self.manual_backward(content_loss)
        _maybe_clip_gradients(self.generator, opt_g)
        opt_g.step()
        if hasattr(self, "untoggle_optimizer"):
            self.untoggle_optimizer(opt_g)

        # EMA in PL2 manual mode
        if self.ema is not None and self.global_step >= self._ema_update_after_step:
            self.ema.update(self.generator)

        return content_loss

    # ======================================================================
    # SECTION: Adversarial training — Discriminator step
    # ======================================================================
    if hasattr(self, "toggle_optimizer"):
        self.toggle_optimizer(opt_d)
    opt_d.zero_grad()

    # Get R1 Gamma
    r1_gamma = getattr(self, "r1_gamma", 0.0)
    hr_imgs.requires_grad_(r1_gamma > 0)  # enable grad for R1 penalty if needed

    hr_discriminated = self.discriminator(hr_imgs)  # D(y)
    sr_discriminated = self.discriminator(sr_imgs.detach())  # D(G(x)) w/o grad to G

    if use_wasserstein:  # Wasserstein GAN loss
        loss_real = -hr_discriminated.mean()
        loss_fake = sr_discriminated.mean()
    else:  # Standard GAN loss (BCE)
        real_target = torch.full_like(hr_discriminated, self.adv_target)
        fake_target = torch.zeros_like(sr_discriminated)

        if use_relativistic:
            # Relativistic Average GAN loss
            real_mean = hr_discriminated.mean()
            fake_mean = sr_discriminated.mean()

            sr_discriminated_rel = sr_discriminated - real_mean
            hr_discriminated_rel = hr_discriminated - fake_mean

            loss_real = self.adversarial_loss_criterion(
                hr_discriminated_rel, real_target
            )
            loss_fake = self.adversarial_loss_criterion(
                sr_discriminated_rel, fake_target
            )
        else:
            # Keep loss scales consistent for non-relativistic BCE.
            loss_real = (
                self.adversarial_loss_criterion(hr_discriminated, real_target) * 0.5
            )
            loss_fake = (
                self.adversarial_loss_criterion(sr_discriminated, fake_target) * 0.5
            )

    # R1 Gradient Penalty
    r1_penalty = torch.zeros((), device=hr_imgs.device, dtype=hr_imgs.dtype)
    if r1_gamma > 0:
        grad_real = torch.autograd.grad(
            outputs=hr_discriminated.sum(),
            inputs=hr_imgs,
            create_graph=True,
            retain_graph=True,
        )[0]
        grad_penalty = grad_real.pow(2).reshape(grad_real.size(0), -1).sum(dim=1)
        r1_penalty = 0.5 * r1_gamma * grad_penalty.mean()

    adversarial_loss = (
        loss_real + loss_fake + r1_penalty
    )  # sum up loss with R1 (0 when turned off)
    self.log(
        "discriminator/adversarial_loss",
        adversarial_loss,
        on_step=True,
        on_epoch=True,
        sync_dist=True,
    )
    self.log(
        "discriminator/r1_penalty", r1_penalty.detach(), sync_dist=True
    )  # log R1 penalty regardless, is 0 when turned off

    with torch.no_grad():
        d_real_prob = torch.sigmoid(hr_discriminated).mean()
        d_fake_prob = torch.sigmoid(sr_discriminated).mean()
    self.log("discriminator/D(y)_prob", d_real_prob, prog_bar=True, sync_dist=True)
    self.log("discriminator/D(G(x))_prob", d_fake_prob, prog_bar=True, sync_dist=True)

    if use_relativistic and not use_wasserstein:
        # Standard probabilities are less informative in RaGAN mode; log relativistic ones too.
        with torch.no_grad():
            real_mean = hr_discriminated.mean()
            fake_mean = sr_discriminated.mean()

            sr_discriminated_rel = sr_discriminated - real_mean
            hr_discriminated_rel = hr_discriminated - fake_mean

            d_real_prob_rel = torch.sigmoid(hr_discriminated_rel).mean()
            d_fake_prob_rel = torch.sigmoid(sr_discriminated_rel).mean()

        self.log(
            "train_metrics/discriminator/D(y)_prob_relativistic",
            d_real_prob_rel,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "train_metrics/discriminator/D(G(x))_prob_relativistic",
            d_fake_prob_rel,
            prog_bar=True,
            sync_dist=True,
        )

    self.manual_backward(adversarial_loss)
    _maybe_clip_gradients(self.discriminator, opt_d)
    opt_d.step()
    if hasattr(self, "untoggle_optimizer"):
        self.untoggle_optimizer(opt_d)

    # ======================================================================
    # SECTION: Adversarial training — Generator step
    # ======================================================================
    if hasattr(self, "toggle_optimizer"):
        self.toggle_optimizer(opt_g)
    opt_g.zero_grad()

    # 1) content loss (identical to original)
    content_loss, metrics = self.content_loss_criterion.return_loss(sr_imgs, hr_imgs)
    self._log_generator_content_loss(content_loss)
    for key, value in metrics.items():
        self.log(f"train_metrics/{key}", value, sync_dist=True)

    # 2) adversarial loss against ones
    sr_discriminated_for_g = self.discriminator(sr_imgs)
    if use_wasserstein:  # Wasserstein GAN loss
        g_adv = -sr_discriminated_for_g.mean()
    else:  # Standard GAN loss (BCE)
        if use_relativistic:
            # Relativistic Average GAN loss for G
            with torch.no_grad():
                hr_discriminated = self.discriminator(hr_imgs)
                real_mean = hr_discriminated.mean()

            fake_mean = sr_discriminated_for_g.mean()
            sr_discriminated_rel = sr_discriminated_for_g - real_mean
            hr_discriminated_rel = hr_discriminated - fake_mean

            loss_fake = self.adversarial_loss_criterion(
                sr_discriminated_rel, torch.ones_like(sr_discriminated_for_g)
            )
            loss_real = self.adversarial_loss_criterion(
                hr_discriminated_rel, torch.zeros_like(hr_discriminated)
            )
            g_adv = (loss_fake + loss_real) / 2.0
        else:
            g_adv = self.adversarial_loss_criterion(
                sr_discriminated_for_g, torch.ones_like(sr_discriminated_for_g)
            )
    self.log("generator/adversarial_loss", g_adv, sync_dist=True)

    # 3) weighted total
    adv_weight = _adv_weight()
    total_loss = content_loss + (g_adv * adv_weight)
    self.log("generator/total_loss", total_loss, sync_dist=True)

    self.manual_backward(total_loss)
    _maybe_clip_gradients(self.generator, opt_g)
    opt_g.step()
    if hasattr(self, "untoggle_optimizer"):
        self.untoggle_optimizer(opt_g)

    # EMA in PL2 manual mode
    if self.ema is not None and self.global_step >= self._ema_update_after_step:
        self.ema.update(self.generator)

    return total_loss
