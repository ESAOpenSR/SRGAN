import torch

def training_step_PL1(self, batch, batch_idx, optimizer_idx):
    """One training step for PL < 2.0 using automatic optimization and multi-optimizers.

    Implements GAN training with two optimizers (D first, then G) and a
    pretraining gate. During the **pretraining phase**, only the generator
    (optimizer_idx == 1) is optimized with content loss; the discriminator
    branch returns a dummy loss and logs zeros. During **adversarial training**,
    the discriminator minimizes BCE on real HR vs. fake SR logits, and the
    generator minimizes content loss plus a ramped adversarial loss.

    Args:
        batch (Tuple[torch.Tensor, torch.Tensor]): `(lr_imgs, hr_imgs)` with shape `(B, C, H, W)`.
        batch_idx (int): Global batch index for the current epoch.
        optimizer_idx (int): Active optimizer index provided by Lightning:
            - `0`: Discriminator step.
            - `1`: Generator step.

    Returns:
        torch.Tensor:
            - **Pretraining**:
              - `optimizer_idx == 1`: content loss tensor for the generator.
              - `optimizer_idx == 0`: dummy scalar tensor with `requires_grad=True`.
            - **Adversarial training**:
              - `optimizer_idx == 0`: discriminator BCE loss (real + fake).
              - `optimizer_idx == 1`: generator total loss = content + λ_adv · BCE(G).

    Logged Metrics (selection):
        - `"training/pretrain_phase"`: 1.0 during pretraining (logged on G step).
        - `"train_metrics/*"`: content metrics from the content loss criterion.
        - `"generator/content_loss"`, `"generator/adversarial_loss"`, `"generator/total_loss"`.
        - `"discriminator/adversarial_loss"`, `"discriminator/D(y)_prob"`,
          `"discriminator/D(G(x))_prob"`.
        - `"training/adv_loss_weight"`: current λ_adv from the ramp scheduler.

    Notes:
        - Discriminator step uses `sr_imgs.detach()` to prevent G gradients.
        - Adversarial loss weight λ_adv ramps from 0 → `adv_loss_beta` per configured schedule.
        - Assumes optimizers are ordered as `[D, G]` in `configure_optimizers()`.
    """
    
    # -------- CREATE SR DATA --------
    lr_imgs, hr_imgs = batch                                  # unpack LR/HR tensors from dataloader batch
    sr_imgs = self.forward(lr_imgs)                          # forward pass of the generator to produce SR from LR

    # ======================================================================
    # SECTION: Pretraining phase gate
    # Purpose: decide if we are in the content-only pretrain stage.
    # ======================================================================

    # -------- DETERMINE PRETRAINING --------
    pretrain_phase = self._pretrain_check()    # check schedule: True => content-only pretraining
    if optimizer_idx == 1:  # log whether pretraining is active or not
        self.log("training/pretrain_phase", float(pretrain_phase), prog_bar=False,sync_dist=True)  # log once per G step to track phase state

    # ======================================================================
    # SECTION: Pretraining branch (delegated)
    # Purpose: during pretrain, only content loss for G and dummy logging for D.
    # ======================================================================

    # -------- IF PRETRAIN: delegate --------
    if pretrain_phase:
        # run pretrain step separately and return loss here
        if optimizer_idx == 1:
            content_loss, metrics = self.content_loss_criterion.return_loss(sr_imgs, hr_imgs)  # compute perceptual/content loss (e.g., VGG or L1)
            self._log_generator_content_loss(content_loss)                             # log content loss for G (consistent args)
            for key, value in metrics.items():
                self.log(f"train_metrics/{key}", value,sync_dist=True)                               # reuse computed metrics for logging

            # Ensure adversarial weight is logged even when not used during pretraining
            adv_weight = self._compute_adv_loss_weight()
            self._log_adv_loss_weight(adv_weight)
            return content_loss                                                        # return loss for optimizer step (G only)

        # ======================================================================
        # SECTION: Discriminator (D) pretraining step
        # Purpose: no real training — just log zeros and return dummy loss to satisfy closure.
        # ======================================================================
        elif optimizer_idx == 0:
            device, dtype = hr_imgs.device, hr_imgs.dtype                              # get tensor device and dtype for consistency
            zero = torch.tensor(0.0, device=device, dtype=dtype)                       # define reusable zero tensor

            # --- Log dummy discriminator "opinions" (always zero during pretrain) ---
            self.log("discriminator/D(y)_prob",    zero, prog_bar=True,  sync_dist=True)  # fake real-prob (always 0)
            self.log("discriminator/D(G(x))_prob", zero, prog_bar=True,  sync_dist=True)  # fake fake-prob (always 0)

            # --- Create dummy scalar loss (ensures PL closure runs) ---
            dummy = torch.zeros((), device=device, dtype=dtype, requires_grad=True)    # dummy value with grad for optimizer compatibility
            self.log("discriminator/adversarial_loss", dummy, sync_dist=True)          # log dummy adversarial loss (always 0)
            return dummy           
    # -------- END PRETRAIN --------

    # ======================================================================
    # SECTION: Adversarial training — Discriminator step
    # Purpose: update D to distinguish HR (real) vs SR (fake).
    # ======================================================================

    # -------- Normal Train: Discriminator Step  --------
    if optimizer_idx==0:
        # run discriminator and get loss between pred labels and true labels
        hr_discriminated = self.discriminator(hr_imgs)       # D(real): logits for HR images
        sr_discriminated = self.discriminator(sr_imgs.detach()) # detach so G doesn’t get gradients from D’s step

        # targets
        real_target = torch.full_like(hr_discriminated, self.adv_target) # get labels/fuzzy labels
        fake_target = torch.zeros_like(sr_discriminated) # zeros, since generative prediction

        # Binary Cross-Entropy loss
        loss_real = self.adversarial_loss_criterion(hr_discriminated, real_target)   # BCEWithLogitsLoss for D(G(x))
        loss_fake = self.adversarial_loss_criterion(sr_discriminated, fake_target)  # BCEWithLogitsLoss for D(y)
        adversarial_loss = loss_real + loss_fake # Sum up losses
        self.log("discriminator/adversarial_loss",adversarial_loss,sync_dist=True) # log weighted loss

        # [LOG-B] Always log D opinions: real probs in normal training
        with torch.no_grad():
            d_real_prob = torch.sigmoid(hr_discriminated).mean()   # estimate mean real probability
            d_fake_prob = torch.sigmoid(sr_discriminated).mean()   # estimate mean fake probability
        self.log("discriminator/D(y)_prob", d_real_prob, prog_bar=True,sync_dist=True)      # log D(real) confidence
        self.log("discriminator/D(G(x))_prob", d_fake_prob, prog_bar=True,sync_dist=True)   # log D(fake) confidence

        # return weighted discriminator loss
        return adversarial_loss                                # PL will use this to step the D optimizer

    # ======================================================================
    # SECTION: Adversarial training — Generator step
    # Purpose: update G to minimize content loss + (weighted) adversarial loss.
    # ======================================================================

    # -------- Normal Train: Generator Step  --------
    if optimizer_idx==1:

        """ 1. Get VGG space loss """
        # encode images
        content_loss, metrics = self.content_loss_criterion.return_loss(sr_imgs, hr_imgs)   # perceptual/content criterion (e.g., VGG)
        self._log_generator_content_loss(content_loss)                             # log content loss for G (consistent args)
        for key, value in metrics.items():
            self.log(f"train_metrics/{key}", value,sync_dist=True)                             # log detailed metrics without extra forward passes


        """ 2. Get Discriminator Opinion and loss """
        # run discriminator and get loss between pred labels and true labels
        sr_discriminated = self.discriminator(sr_imgs)                             # D(SR): logits for generator outputs
        adversarial_loss = self.adversarial_loss_criterion(sr_discriminated, torch.ones_like(sr_discriminated)) # keep taargets 1.0 for G loss
        self.log("generator/adversarial_loss",adversarial_loss,sync_dist=True)     # log unweighted adversarial loss

        """ 3. Weight the losses"""
        adv_weight = self._adv_loss_weight() # get adversarial weight based on current step
        adversarial_loss_weighted = (adversarial_loss * adv_weight) # weight adversarial loss
        total_loss = content_loss + adversarial_loss_weighted # total content loss
        self.log("generator/total_loss",total_loss,sync_dist=True)  # log combined objective (content + λ_adv * adv)

        # return Generator loss
        return total_loss         
    
    
def training_step_PL2(self, batch, batch_idx):
    """Manual-optimization training step for PyTorch Lightning ≥ 2.0.

    Mirrors the PL1.x logic with explicit optimizer control:
    - **Pretraining phase**: Discriminator logs dummies; Generator is optimized with
      content loss only (no adversarial term), and EMA optionally updates.
    - **Adversarial phase**: Performs a Discriminator step (real vs. fake BCE),
      followed by a Generator step (content + λ_adv · BCE against ones). Uses the
      same log keys and ordering as the PL1.x path.

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

    Raises:
        AssertionError: If PL version < 2.0 or `automatic_optimization` is True.
    """
    assert self.pl_version >= (2,0,0), "training_step_PL2 requires PyTorch Lightning >= 2.x."
    assert self.automatic_optimization is False, "training_step_PL2 requires manual_optimization."
    
    # -------- CREATE SR DATA --------
    lr_imgs, hr_imgs = batch
    sr_imgs = self.forward(lr_imgs)

    # --- helper to resolve adv-weight function name mismatches ---
    def _adv_weight():
        if hasattr(self, "_adv_loss_weight"):
            return self._adv_loss_weight()
        return self._compute_adv_loss_weight()

    # fetch optimizers (expects two)
    opt_d, opt_g = self.optimizers()

    # optional gradient clipping support (norm-based)
    sched_cfg = getattr(self.config, "Schedulers", None)
    gradient_clip_val = 0.0
    if sched_cfg is not None:
        gradient_clip_val = float(getattr(sched_cfg, "gradient_clip_val", 0.0) or 0.0)

    def _maybe_clip_gradients(module):
        if gradient_clip_val > 0.0 and module is not None:
            torch.nn.utils.clip_grad_norm_(module.parameters(), gradient_clip_val)

    # ======================================================================
    # SECTION: Pretraining phase gate
    # ======================================================================
    pretrain_phase = self._pretrain_check()
    # in PL1.x you logged this only on G-step; here we log once per batch
    self.log("training/pretrain_phase", float(pretrain_phase), prog_bar=False, sync_dist=True)

    # ======================================================================
    # SECTION: Pretraining branch (content-only on G; D logs dummies)
    # ======================================================================
    if pretrain_phase:
        # --- D dummy logs (no step) to mimic your optimizer_idx==0 branch ---
        with torch.no_grad():
            zero = torch.tensor(0.0, device=hr_imgs.device, dtype=hr_imgs.dtype)
            self.log("discriminator/D(y)_prob",    zero, prog_bar=True,  sync_dist=True)
            self.log("discriminator/D(G(x))_prob", zero, prog_bar=True,  sync_dist=True)
            self.log("discriminator/adversarial_loss", zero, sync_dist=True)

        # --- G step: content loss only (identical to your optimizer_idx==1 pretrain) ---
        content_loss, metrics = self.content_loss_criterion.return_loss(sr_imgs, hr_imgs)
        self._log_generator_content_loss(content_loss)
        for key, value in metrics.items():
            self.log(f"train_metrics/{key}", value, sync_dist=True)

        # ensure adv-weight is still logged like in pretrain
        self._log_adv_loss_weight(_adv_weight())

        # manual optimize G
        if hasattr(self, "toggle_optimizer"): self.toggle_optimizer(opt_g)
        opt_g.zero_grad()
        self.manual_backward(content_loss)
        _maybe_clip_gradients(self.generator)
        opt_g.step()
        if hasattr(self, "untoggle_optimizer"): self.untoggle_optimizer(opt_g)
        
        # EMA in PL2 manual mode
        if self.ema is not None and self.global_step >= self._ema_update_after_step:
            self.ema.update(self.generator)

        # return same scalar you’d have returned in PL1.x (content loss)
        return content_loss

    # ======================================================================
    # SECTION: Adversarial training — Discriminator step
    # ======================================================================
    if hasattr(self, "toggle_optimizer"): self.toggle_optimizer(opt_d)
    opt_d.zero_grad()

    hr_discriminated = self.discriminator(hr_imgs)              # D(y)
    sr_discriminated = self.discriminator(sr_imgs.detach())     # D(G(x)) w/o grad to G

    real_target = torch.full_like(hr_discriminated, self.adv_target)
    fake_target = torch.zeros_like(sr_discriminated)

    loss_real = self.adversarial_loss_criterion(hr_discriminated, real_target)
    loss_fake = self.adversarial_loss_criterion(sr_discriminated, fake_target)
    adversarial_loss = loss_real + loss_fake
    self.log("discriminator/adversarial_loss", adversarial_loss, sync_dist=True)

    with torch.no_grad():
        d_real_prob = torch.sigmoid(hr_discriminated).mean()
        d_fake_prob = torch.sigmoid(sr_discriminated).mean()
    self.log("discriminator/D(y)_prob", d_real_prob, prog_bar=True,  sync_dist=True)
    self.log("discriminator/D(G(x))_prob", d_fake_prob, prog_bar=True, sync_dist=True)

    self.manual_backward(adversarial_loss)
    _maybe_clip_gradients(self.discriminator)
    opt_d.step()
    if hasattr(self, "untoggle_optimizer"): self.untoggle_optimizer(opt_d)

    # ======================================================================
    # SECTION: Adversarial training — Generator step
    # ======================================================================
    if hasattr(self, "toggle_optimizer"): self.toggle_optimizer(opt_g)
    opt_g.zero_grad()

    # 1) content loss (identical to original)
    content_loss, metrics = self.content_loss_criterion.return_loss(sr_imgs, hr_imgs)
    self._log_generator_content_loss(content_loss)
    for key, value in metrics.items():
        self.log(f"train_metrics/{key}", value, sync_dist=True)

    # 2) adversarial loss against ones
    sr_discriminated_for_g = self.discriminator(sr_imgs)
    g_adv = self.adversarial_loss_criterion(sr_discriminated_for_g, torch.ones_like(sr_discriminated_for_g))
    self.log("generator/adversarial_loss", g_adv, sync_dist=True)

    # 3) weighted total
    adv_weight = _adv_weight()
    total_loss = content_loss + (g_adv * adv_weight)
    self.log("generator/total_loss", total_loss, sync_dist=True)

    self.manual_backward(total_loss)
    _maybe_clip_gradients(self.generator)
    opt_g.step()
    if hasattr(self, "untoggle_optimizer"): self.untoggle_optimizer(opt_g)
    
    # EMA in PL2 manual mode
    if self.ema is not None and self.global_step >= self._ema_update_after_step:
        self.ema.update(self.generator)

    # return same scalar you return in PL1.x G path
    return total_loss
