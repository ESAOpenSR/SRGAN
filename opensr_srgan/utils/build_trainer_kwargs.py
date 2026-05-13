import inspect
from collections.abc import Sequence

import torch
import pytorch_lightning as pl


def build_lightning_kwargs(
    config,
    logger,
    checkpoint_callback,
    early_stop_callback,
    resume_ckpt: str | None = None,
):
    """Return Trainer/fit keyword arguments for Lightning 2+.

    Builds two dictionaries:
    1) ``trainer_kwargs`` — arguments for ``pytorch_lightning.Trainer``.
    2) ``fit_kwargs`` — arguments for ``Trainer.fit`` (e.g., ``ckpt_path``).

    The helper normalizes device configuration (CPU/GPU, DDP when multiple GPUs),
    removes ``None`` entries, and filters kwargs against the active Lightning
    signatures to avoid passing unsupported arguments.

    Args:
        config: OmegaConf-like config with ``Training`` fields:
            - ``Training.device`` (str): "auto"|"cpu"|"cuda"/"gpu".
            - ``Training.gpus`` (int|Sequence[int]|None): device count/IDs.
            - ``Training.val_check_interval`` (int|float).
            - ``Training.limit_val_batches`` (int|float).
            - ``Training.max_epochs`` (int).
        logger: A Lightning-compatible logger instance.
        checkpoint_callback: Model checkpoint callback instance.
        early_stop_callback: Early stopping callback instance.
        resume_ckpt (str | None): Path to checkpoint to resume from.

    Returns:
        Tuple[Dict[str, Any], Dict[str, Any]]:
            - trainer_kwargs: Dict for ``pl.Trainer(**trainer_kwargs)``.
            - fit_kwargs: Dict for ``trainer.fit(..., **fit_kwargs)``.

    Raises:
        ValueError: If ``Training.device`` is not one of {"auto","cpu","cuda","gpu"}.

    Notes:
        - CPU runs force ``devices=1`` and no strategy.
        - GPU runs honor ``Training.gpus``; DDP is enabled when requesting >1 device.
        - Resume checkpoints are forwarded through ``Trainer.fit(ckpt_path=...)``.
    """

    # ---------------------------------------------------------------------
    # 1) Parse device configuration from the OmegaConf config
    # ---------------------------------------------------------------------
    # ``Training.gpus`` may be specified either as an integer (e.g. ``2``) or a
    # sequence (e.g. ``[0, 1]``).  We keep the raw object so it can be passed to
    # the Trainer later if required, but we also count how many devices are
    # requested to decide on the parallelisation strategy.
    devices_cfg = getattr(config.Training, "gpus", None)

    # ``Training.device`` is the user-facing string that selects the backend.
    # Valid values are ``"cuda"`` / ``"gpu"`` (equivalent), ``"cpu"`` or
    # ``"auto"`` to defer to ``torch.cuda.is_available``.
    device_cfg = str(getattr(config.Training, "device", "auto")).lower()

    def _count_devices(devices):
        """Return how many explicit device identifiers were supplied."""

        # ``Trainer(devices=N)`` accepts both integers and sequences.  When the
        # user specifies an integer we can return it directly.  For sequences we
        # only count non-string iterables because strings are technically
        # sequences too but do not represent a collection of device identifiers.
        if isinstance(devices, int):
            return devices
        if isinstance(devices, Sequence) and not isinstance(devices, (str, bytes)):
            return len(devices)
        return 0

    ndev = _count_devices(devices_cfg)

    # Map the high-level ``device`` selector to the Lightning ``accelerator``
    # option.  ``auto`` chooses GPU when available and CPU otherwise so CLI
    # overrides are not required when moving between machines.
    if device_cfg in {"cuda", "gpu"}:
        accelerator = "gpu"
    elif device_cfg == "cpu":
        accelerator = "cpu"
    elif device_cfg in {"auto", ""}:
        accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    else:
        raise ValueError(f"Unsupported Training.device '{device_cfg}'")

    # When operating on CPU we force Lightning to a single device.  Allowing the
    # caller to pass the GPU list would be misleading because PyTorch does not
    # support multiple CPUs in the same way as GPUs.  On GPU we honour the user
    # supplied configuration and enable DistributedDataParallel only when more
    # than one device is requested.
    if accelerator == "cpu":
        devices = 1
        strategy = None
    else:
        devices = devices_cfg if ndev else 1
        if ndev > 1:
            # GAN manual optimization updates only one optimizer branch at a time,
            # so DDP must track unused params on each step.
            find_unused = bool(getattr(config.Training, "find_unused_parameters", True))
            strategy = "ddp_find_unused_parameters_true" if find_unused else "ddp"
        else:
            strategy = None

    # ---------------------------------------------------------------------
    # 2) Assemble the base Trainer kwargs
    # ---------------------------------------------------------------------
    trainer_kwargs = dict(
        accelerator=accelerator,
        strategy=strategy,  # removed in the next step when ``None``
        devices=devices,
        val_check_interval=config.Training.val_check_interval,
        limit_val_batches=config.Training.limit_val_batches,
        max_epochs=config.Training.max_epochs,
        log_every_n_steps=50,
        logger=[logger],
        callbacks=[checkpoint_callback, early_stop_callback],
    )

    # ``strategy`` defaults to ``None`` on CPU runs.  Lightning does not accept
    # explicit ``None`` values in its constructor, therefore we prune every
    # key/value pair whose value evaluates to ``None`` before forwarding the
    # kwargs.
    trainer_kwargs = {k: v for k, v in trainer_kwargs.items() if v is not None}

    # Some Lightning releases occasionally deprecate constructor arguments.  To
    # ensure we do not pass stale options we filter the dictionary so it only
    # contains parameters that are still accepted by ``Trainer.__init__``.
    init_sig = inspect.signature(pl.Trainer.__init__).parameters
    trainer_kwargs = {k: v for k, v in trainer_kwargs.items() if k in init_sig}

    # ---------------------------------------------------------------------
    # 3) ``Trainer.fit`` keyword arguments
    # ---------------------------------------------------------------------
    fit_kwargs = {}
    if resume_ckpt:
        fit_sig = inspect.signature(pl.Trainer.fit).parameters
        if "ckpt_path" in fit_sig:
            fit_kwargs["ckpt_path"] = resume_ckpt

    return trainer_kwargs, fit_kwargs
