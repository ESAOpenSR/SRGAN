def print_model_summary(self):
    """
    Prints a detailed and visually structured summary of the SRGAN configuration.
    Includes architecture info, resolution scale, training parameters, loss weights, and model sizes.
    """

    # --- helpers (millions) ---
    def count_trainable_params(m):
        return sum(p.numel() for p in m.parameters() if p.requires_grad) / 1e6

    def count_total_params(m):
        return sum(p.numel() for p in m.parameters()) / 1e6

    # --- counts ---
    g_trainable = count_trainable_params(self.generator)
    g_total = count_total_params(self.generator)
    d_trainable = count_trainable_params(self.discriminator)
    d_total = count_total_params(self.discriminator)

    # expose the exact names you want to use elsewhere
    g_params = g_trainable  # alias for “trainable G params (M)”
    d_params = d_trainable  # alias for “trainable D params (M)”

    total_trainable = g_trainable + d_trainable
    total_all = g_total + d_total

    # ------------------------------------------------------------------
    # Derive human-readable generator description
    # ------------------------------------------------------------------
    g_type = getattr(self.config.Generator, "model_type", "SRResNet")
    g_type_norm = str(g_type).lower().replace("-", "_")
    block_type = getattr(self.config.Generator, "block_type", None)

    if g_type_norm in {"res", "rcab", "rrdb", "lka"} and block_type is None:
        block_type = g_type_norm
        g_type_norm = "srresnet"

    block_desc_map = {
        "standard": "Residual Blocks with BatchNorm",
        "res": "Residual Blocks without BatchNorm",
        "rcab": "RCAB Blocks with Channel Attention",
        "rrdb": "RRDB Dense Residual Blocks",
        "lka": "LKA Large-Kernel Attention Blocks",
    }

    if g_type_norm in {"srresnet", "sr_resnet"}:
        block_key = "standard" if block_type is None else str(block_type).lower()
        desc = block_desc_map.get(block_key, f"Custom Block Variant: {block_key}")
        g_desc = f"SRResNet ({desc})"
    elif g_type_norm in {"stochastic_gan", "cgan", "conditional_cgan"}:
        g_desc = "Stochastic SRGAN (Latent-modulated Residual Blocks)"
    elif g_type_norm == "esrgan":
        g_desc = "ESRGAN (RRDB Residual-in-Residual Dense Network)"
    else:
        g_desc = f"Custom Generator Type: {g_type}"

    # ------------------------------------------------------------------
    # Resolution info (input → output)
    # ------------------------------------------------------------------
    scale_factor = getattr(
        self.config.Generator, "scaling_factor", getattr(self.generator, "scale", None)
    )
    if scale_factor is not None:
        res_str = f"Super-Resolution Factor: ×{scale_factor}"
    else:
        res_str = "Super-Resolution Factor: Unknown"

    # ------------------------------------------------------------------
    # Retrieve loss weights if available
    # ------------------------------------------------------------------
    loss_cfg = getattr(self.config.Training, "Losses", {})
    content_w = getattr(loss_cfg, "content_loss_weight", 1.0)
    adv_w = getattr(loss_cfg, "adv_loss_beta", 1.0)
    perceptual_w = getattr(loss_cfg, "perceptual_loss_weight", None)
    total_w_str = f"   • Content: {content_w} | Adversarial: {adv_w}" + (
        f" | Perceptual: {perceptual_w}" if perceptual_w is not None else ""
    )

    print("\n" + "=" * 90)
    print("🚀  SRGAN Model Summary")
    print("=" * 90)

    # ------------------------------------------------------------------
    # Generator Info
    # ------------------------------------------------------------------
    print(f"🧩 Generator")
    print(f"   • Architecture:      {g_desc}")
    print(f"   • Resolution:        {res_str}")
    print(f"   • Input Channels:    {self.config.Model.in_bands}")
    feature_channels = getattr(self.config.Generator, "n_channels", None)
    if feature_channels is not None:
        print(f"   • Feature Channels:  {feature_channels}")

    block_count = getattr(
        self.config.Generator,
        "n_blocks",
        getattr(self.generator, "n_blocks", None),
    )
    if block_count is not None:
        print(f"   • Residual Blocks:   {block_count}")

    small_kernel = getattr(self.config.Generator, "small_kernel_size", None)
    large_kernel = getattr(self.config.Generator, "large_kernel_size", None)
    if small_kernel is not None or large_kernel is not None:
        print(
            "   • Kernel Sizes:      small={small}, large={large}".format(
                small=small_kernel if small_kernel is not None else "N/A",
                large=large_kernel if large_kernel is not None else "N/A",
            )
        )
    print(f"   • Params:            {g_params:.2f} M\n")

    # ------------------------------------------------------------------
    # Discriminator Info
    # ------------------------------------------------------------------
    d_type = getattr(self.config.Discriminator, "model_type", "standard")
    d_blocks = getattr(self.config.Discriminator, "n_blocks", None)
    effective_blocks = getattr(
        self.discriminator,
        "n_blocks",
        getattr(self.discriminator, "n_layers", d_blocks),
    )
    base_channels = getattr(self.discriminator, "base_channels", "N/A")
    kernel_size = getattr(self.discriminator, "kernel_size", "N/A")
    fc_size = getattr(self.discriminator, "fc_size", None)

    if d_type == "patchgan":
        d_desc = "PatchGAN"
    elif d_type == "esrgan":
        d_desc = "ESRGAN"
    else:
        d_desc = "SRGAN"

    print(f"🧠 Discriminator")
    print(f"   • Architecture:     {d_desc}")
    if effective_blocks is not None:
        print(f"   • Blocks/Layers:    {effective_blocks}")
    print(f"   • Base Channels:    {base_channels}")
    print(f"   • Kernel Size:      {kernel_size}")
    if fc_size is not None:
        print(f"   • FC Layer Size:    {fc_size}")
    print(f"   • Params:            {d_params:.2f} M\n")

    # ------------------------------------------------------------------
    # Training Setup
    # ------------------------------------------------------------------
    print(f"⚙️  Training Configuration")
    print(f"   • Pretrain Generator: {self.pretrain_g_only}")
    print(f"   • Pretrain Steps:     {self.g_pretrain_steps}")
    print(f"   • Adv. Ramp Steps:    {self.adv_loss_ramp_steps}")
    print(f"   • Label Smoothing:    {self.adv_target < 1.0}")
    print(f"   • Adv. Target Label:  {self.adv_target}\n")

    # ------------------------------------------------------------------
    # EMA Configuration (if present)
    # ------------------------------------------------------------------
    ema_cfg = getattr(getattr(self.config.Training, "EMA", None), "__dict__", None)
    if ema_cfg or hasattr(self.config.Training, "EMA"):
        ema = self.config.Training.EMA
        enabled = getattr(ema, "enabled", False)
        decay = getattr(ema, "decay", None)
        update_after = getattr(ema, "update_after_step", None)
        use_num_updates = getattr(ema, "use_num_updates", None)

        print(f"🔁 Exponential Moving Average (EMA)")
        print(f"   • Enabled:           {enabled}")
        if enabled:
            print(f"   • Decay:             {decay}")
            print(f"   • Update After Step: {update_after}")
            print(f"   • Use Num Updates:   {use_num_updates}")
        print()  # newline for spacing

    # ------------------------------------------------------------------
    # Loss Functions
    # ------------------------------------------------------------------
    print(f"📉 Loss Functions")
    print(f"   • Content Loss:       {type(self.content_loss_criterion).__name__}")
    print(f"   • Adversarial Loss:   {type(self.adversarial_loss_criterion).__name__}")
    print(total_w_str + "\n")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"📊 Model Summary")
    print(f"   • Total Params:           {total_all:.2f} M")
    print(f"   • Total Trainable Params: {total_trainable:.2f} M")
    print(
        f"   • Device:                 {self.device if hasattr(self, 'device') else 'Not set'}"
    )
    print("=" * 90 + "\n")
