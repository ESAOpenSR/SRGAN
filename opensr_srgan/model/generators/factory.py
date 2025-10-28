"""Factory helpers for constructing generator backbones from a config object."""

from __future__ import annotations

from typing import Any

from torch import nn

from .cgan_generator import StochasticGenerator
from .esrgan import ESRGANGenerator
from .flexible_generator import FlexibleGenerator
from .srresnet import Generator as SRResNetGenerator

__all__ = ["build_generator"]


_SRRESNET_BLOCK_ALIASES = {
    "standard": {"standard", "srresnet", "vanilla", "classic", "default", "basic"},
    "res": {"res", "residual", "resblocks"},
    "rcab": {"rcab", "attention", "channel_attention"},
    "rrdb": {"rrdb", "dense", "residual_in_residual"},
    "lka": {"lka", "large_kernel", "large-kernel"},
}

_MODEL_TYPE_ALIASES = {
    "srresnet": {"srresnet", "sr_resnet", "sr-resnet"},
    "esrgan": {"esrgan", "rrdbnet", "rrdb_net"},
    "stochastic_gan": {
        "stochastic_gan",
        "stochastic",
        "stochastic-gan",
        "stochasticgan",
        "cgan",
        "conditional_cgan",
        "conditional-gan",
    },
}


def _normalise(value: str) -> str:
    return value.strip().lower().replace(" ", "_").replace("-", "_")


def _match_alias(value: str, mapping: dict[str, set[str]]) -> str | None:
    norm = _normalise(value)
    for canonical, aliases in mapping.items():
        if norm == canonical or norm in aliases:
            return canonical
    return None


def _resolve_srresnet_block(generator_cfg: Any, model_type_hint: str) -> str:
    """Determine which SRResNet variant should be instantiated."""

    block_type = getattr(generator_cfg, "block_type", None)
    if block_type is None:
        block_type = model_type_hint

    if block_type is None:
        return "standard"

    block_key = _normalise(str(block_type))

    for canonical, aliases in _SRRESNET_BLOCK_ALIASES.items():
        if block_key == canonical or block_key in aliases:
            return canonical

    raise ValueError(
        "Unknown SRResNet block type '{block}'. Expected one of: {options}.".format(
            block=block_type,
            options=", ".join(sorted(_SRRESNET_BLOCK_ALIASES)),
        )
    )


def _warn_overridden_options(component: str, model: str, options: list[str]) -> None:
    """Emit a user-facing notice when configuration options are ignored."""

    if not options:
        return

    joined = ", ".join(sorted(options))
    print(f"[{component}:{model}] Ignoring unsupported configuration options: {joined}.")


def _collect_overridden(generator_cfg: Any, *keys: str) -> list[str]:
    overridden: list[str] = []
    for key in keys:
        value = getattr(generator_cfg, key, None)
        if value is not None:
            overridden.append(key)
    return overridden


def build_generator(config: Any) -> nn.Module:
    """Instantiate a generator module from the user configuration.

    Parameters
    ----------
    config:
        Resolved configuration object. The function expects at least the
        ``Model`` and ``Generator`` sections used throughout the training
        scripts.

    Returns
    -------
    torch.nn.Module
        Fully constructed generator ready for training or inference.
    """

    generator_cfg = config.Generator
    model_cfg = config.Model

    raw_model_type = str(getattr(generator_cfg, "model_type", "srresnet"))
    model_type = _match_alias(raw_model_type, _MODEL_TYPE_ALIASES)

    # Legacy support: configs that directly specified the block variant inside
    # ``model_type`` (e.g., "rcab") should still build the flexible generator.
    if model_type is None:
        srresnet_block = _match_alias(raw_model_type, _SRRESNET_BLOCK_ALIASES)
        if srresnet_block is not None:
            model_type = "srresnet"
        else:
            raise ValueError(
                "Unknown generator model type '{model_type}'. Expected one of: {options}.".format(
                    model_type=raw_model_type,
                    options=", ".join(sorted(_MODEL_TYPE_ALIASES)),
                )
            )
    else:
        srresnet_block = None

    in_channels = int(getattr(model_cfg, "in_bands"))
    scale = int(getattr(generator_cfg, "scaling_factor", 4))

    if model_type == "srresnet":
        large_kernel = int(getattr(generator_cfg, "large_kernel_size", 9))
        small_kernel = int(getattr(generator_cfg, "small_kernel_size", 3))
        n_channels = int(getattr(generator_cfg, "n_channels", 64))
        n_blocks = int(getattr(generator_cfg, "n_blocks", 16))

        block_variant = _resolve_srresnet_block(generator_cfg, srresnet_block)

        if block_variant == "standard":
            return SRResNetGenerator(
                in_channels=in_channels,
                large_kernel_size=large_kernel,
                small_kernel_size=small_kernel,
                n_channels=n_channels,
                n_blocks=n_blocks,
                scaling_factor=scale,
            )

        return FlexibleGenerator(
            in_channels=in_channels,
            n_channels=n_channels,
            n_blocks=n_blocks,
            small_kernel=small_kernel,
            large_kernel=large_kernel,
            scale=scale,
            block_type=block_variant,
        )

    if model_type == "stochastic_gan":
        large_kernel = int(getattr(generator_cfg, "large_kernel_size", 9))
        small_kernel = int(getattr(generator_cfg, "small_kernel_size", 3))
        n_channels = int(getattr(generator_cfg, "n_channels", 64))
        n_blocks = int(getattr(generator_cfg, "n_blocks", 16))
        noise_dim = int(getattr(generator_cfg, "noise_dim", 128))
        res_scale = float(getattr(generator_cfg, "res_scale", 0.2))

        _warn_overridden_options(
            "Generator",
            "stochastic_gan",
            _collect_overridden(generator_cfg, "block_type"),
        )

        return StochasticGenerator(
            in_channels=in_channels,
            n_channels=n_channels,
            n_blocks=n_blocks,
            small_kernel=small_kernel,
            large_kernel=large_kernel,
            scale=scale,
            noise_dim=noise_dim,
            res_scale=res_scale,
        )

    if model_type == "esrgan":
        n_channels = int(getattr(generator_cfg, "n_channels", 64))
        n_rrdb = int(getattr(generator_cfg, "n_blocks", 23))
        growth_channels = int(getattr(generator_cfg, "growth_channels", 32))
        res_scale = float(getattr(generator_cfg, "res_scale", 0.2))
        out_channels = int(getattr(generator_cfg, "out_channels", in_channels))

        _warn_overridden_options(
            "Generator",
            "esrgan",
            _collect_overridden(
                generator_cfg,
                "block_type",
                "large_kernel_size",
                "small_kernel_size",
                "noise_dim",
            ),
        )

        return ESRGANGenerator(
            in_channels=in_channels,
            out_channels=out_channels,
            n_features=n_channels,
            n_blocks=n_rrdb,
            growth_channels=growth_channels,
            res_scale=res_scale,
            scale=scale,
        )

    raise ValueError(
        "Unhandled generator model type '{model_type}'. Expected one of: {options}.".format(
            model_type=raw_model_type,
            options=", ".join(sorted(_MODEL_TYPE_ALIASES)),
        )
    )
