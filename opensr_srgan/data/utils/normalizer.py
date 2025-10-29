"""Utility helpers for configuring tensor normalization strategies."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from importlib import import_module
from typing import Any, Callable, Dict, Mapping, MutableMapping, Optional, Tuple

import torch

from ...utils.radiometrics import (
    normalise_10k,
    normalise_10k_signed,
    normalise_s2,
    sen2_stretch,
    zero_one_signed,
)


NormalizationFn = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class NormalizationStrategy:
    """Container storing forward/backward callable pairs."""

    normalize: NormalizationFn
    denormalize: NormalizationFn


@dataclass
class _NormalizerConfig:
    """Lightweight wrapper around config lookups.

    Parameters
    ----------
    method: str
        Name of the normalization strategy requested via the user config.
    """

    method: str


class Normalizer:
    """Factory for applying configurable normalization/denormalization.

    The normalizer inspects the provided configuration, determines the
    requested normalization scheme, and exposes ``normalize`` / ``denormalize``
    helpers that downstream components can reuse without importing
    :mod:`utils.spectral_helpers` directly.

    Supported methods include remote-sensing-focused helpers such as
    ``"normalise_10k"`` (0–10000 reflectance → ``[0, 1]``),
    ``"normalise_10k_signed"`` (0–10000 reflectance → ``[-1, 1]``),
    ``"normalise_s2"`` (Sentinel-2 symmetric stretch), ``"zero_one"`` (clamp to
    ``[0, 1]``) and ``"zero_one_signed"`` (``[0, 1]`` ↔ ``[-1, 1]``). Custom
    strategies can be registered by providing a mapping with
    ``{"name": "custom", "normalize": "module:callable", ...}`` in the
    configuration.
    """

    _STANDARD_METHODS: Dict[str, NormalizationStrategy] = {
        "sen2_stretch": NormalizationStrategy(
            normalize=sen2_stretch,
            denormalize=lambda tensor: torch.clamp(tensor * (3.0 / 10.0), 0.0, 1.0),
        ),
        "normalise_10k": NormalizationStrategy(
            normalize=partial(normalise_10k, stage="norm"),
            denormalize=partial(normalise_10k, stage="denorm"),
        ),
        "normalise_10k_signed": NormalizationStrategy(
            normalize=partial(normalise_10k_signed, stage="norm"),
            denormalize=partial(normalise_10k_signed, stage="denorm"),
        ),
        "normalise_s2": NormalizationStrategy(
            normalize=partial(normalise_s2, stage="norm"),
            denormalize=partial(normalise_s2, stage="denorm"),
        ),
        "zero_one": NormalizationStrategy(
            normalize=lambda tensor: torch.clamp(tensor, 0.0, 1.0),
            denormalize=lambda tensor: torch.clamp(tensor, 0.0, 1.0),
        ),
        "zero_one_signed": NormalizationStrategy(
            normalize=partial(zero_one_signed, stage="norm"),
            denormalize=partial(zero_one_signed, stage="denorm"),
        ),
        "identity": NormalizationStrategy(
            normalize=lambda tensor: tensor,
            denormalize=lambda tensor: tensor,
        ),
    }

    _ALIASES: Dict[str, str] = {
        "normalize_10k": "normalise_10k",
        "reflectance": "normalise_10k",
        "reflectance_0_1": "normalise_10k",
        "reflectance_signed": "normalise_10k_signed",
        "normalize_10k_signed": "normalise_10k_signed",
        "sentinel2": "normalise_10k",
        "sentinel2_signed": "normalise_10k_signed",
        "s2": "normalise_10k",
        "s2_signed": "normalise_10k_signed",
        "normalize_s2": "normalise_s2",
        "zero_to_one": "zero_one",
        "zero_one_range": "zero_one",
        "signed_zero_one": "zero_one_signed",
        "minusone_one": "zero_one_signed",
        "none": "identity",
    }

    def __init__(self, config: Any):
        data_cfg = getattr(config, "Data", None)
        raw_cfg: Any = None
        if data_cfg is not None:
            raw_cfg = getattr(data_cfg, "normalization", None)
            if raw_cfg is None and isinstance(data_cfg, dict):
                raw_cfg = data_cfg.get("normalization")
        if raw_cfg is None:
            raw_cfg = "sen2_stretch"

        method, strategy = self._resolve_strategy(raw_cfg)
        self._cfg = _NormalizerConfig(method=method)
        self._strategy = strategy

    @property
    def method(self) -> str:
        """Return the normalization method configured for this instance."""

        return self._cfg.method

    def normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Normalize ``tensor`` according to the configured method."""

        return self._strategy.normalize(tensor)

    def denormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Invert the normalization previously applied by :meth:`normalize`."""

        return self._strategy.denormalize(tensor)

    @classmethod
    def available_methods(cls) -> Tuple[str, ...]:
        """Return the canonical names of built-in normalization strategies."""

        return tuple(sorted(cls._STANDARD_METHODS.keys()))

    def _resolve_strategy(
        self, raw_cfg: Any
    ) -> Tuple[str, NormalizationStrategy]:
        """Resolve ``raw_cfg`` into a normalisation strategy.

        Parameters
        ----------
        raw_cfg : Any
            Configuration value extracted from ``Data.normalization``. Can be a
            string alias, a mapping with ``name``/``method`` keys, or a mapping
            describing custom callables.
        """

        if isinstance(raw_cfg, Mapping):
            name = raw_cfg.get("name") or raw_cfg.get("method") or "custom"
            name = str(name).strip().lower()
            name = name.replace("normalize", "normalise")
            if name == "custom":
                strategy = self._build_custom_strategy(raw_cfg)
                return "custom", strategy
            raw_cfg = name

        if not isinstance(raw_cfg, str):
            raise TypeError(
                "Normalization config must be a string or mapping, "
                f"received: {type(raw_cfg)!r}"
            )

        method = raw_cfg.strip().lower()
        method = method.replace("normalize", "normalise")
        method = self._ALIASES.get(method, method)

        if method == "custom":
            raise ValueError(
                "Use a mapping with 'name: custom' and callable paths to configure custom normalization."
            )

        try:
            strategy = self._STANDARD_METHODS[method]
        except KeyError as exc:
            supported = ", ".join(sorted(self._STANDARD_METHODS))
            raise ValueError(
                f"Unsupported normalization '{raw_cfg}'. Supported methods: {supported}."
            ) from exc

        return method, strategy

    def _build_custom_strategy(
        self, cfg: Mapping[str, Any]
    ) -> NormalizationStrategy:
        """Instantiate a strategy from user-supplied callables."""

        if "normalize" not in cfg:
            raise ValueError(
                "Custom normalization requires a 'normalize' callable path."
            )

        normalize_path = cfg["normalize"]
        denormalize_path = cfg.get("denormalize")

        shared_kwargs = dict(cfg.get("kwargs", {}))
        norm_kwargs = {**shared_kwargs, **cfg.get("normalize_kwargs", {})}
        denorm_kwargs = {**shared_kwargs, **cfg.get("denormalize_kwargs", {})}

        normalize_fn = _load_callable(normalize_path, norm_kwargs)
        if denormalize_path is None:
            if denorm_kwargs:
                raise ValueError(
                    "'denormalize_kwargs' provided without a 'denormalize' callable."
                )
            denormalize_fn = lambda tensor: tensor
        else:
            denormalize_fn = _load_callable(denormalize_path, denorm_kwargs)

        return NormalizationStrategy(
            normalize=normalize_fn,
            denormalize=denormalize_fn,
        )


def _load_callable(
    path: Any, kwargs: Optional[MutableMapping[str, Any]]
) -> NormalizationFn:
    """Import ``path`` and partially apply ``kwargs`` if provided."""

    if callable(path):
        func = path  # already a callable (useful for tests)
    else:
        if not isinstance(path, str):
            raise TypeError(
                "Normalization callable paths must be strings or callables."
            )
        module_name, attr = _split_import_path(path)
        module = import_module(module_name)
        func = getattr(module, attr)
        if not callable(func):
            raise TypeError(
                f"Imported object '{path}' is not callable (type={type(func)!r})."
            )

    call_kwargs: Dict[str, Any] = dict(kwargs or {})

    if call_kwargs:
        bound = partial(func, **call_kwargs)
    else:
        bound = func

    return lambda tensor: bound(tensor)


def _split_import_path(path: str) -> Tuple[str, str]:
    """Split ``module.attr`` or ``module:attr`` import paths."""

    if ":" in path:
        module_name, attr = path.split(":", 1)
    else:
        module_name, _, attr = path.rpartition(".")
    if not module_name or not attr:
        raise ValueError(
            "Normalization callable paths must include a module and attribute, "
            f"received: '{path}'."
        )
    return module_name, attr
