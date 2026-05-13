"""Exponential Moving Average (EMA) utilities for model parameter smoothing."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Iterator

import torch
from torch import nn


class ExponentialMovingAverage:
    """Maintain an exponential moving average (EMA) of a model’s parameters and buffers.

    This class provides a self-contained implementation of parameter smoothing
    via EMA, commonly used to stabilize training and improve generalization in
    deep generative models. It tracks both model parameters and registered
    buffers (e.g., batch norm statistics), maintains a decayed running average,
    and allows temporary swapping of model weights for evaluation or checkpointing.

    EMA is updated with each training step:
    ```
    shadow = decay * shadow + (1 - decay) * parameter
    ```
    where ``decay`` is typically close to 1.0 (e.g., 0.999–0.9999).

    The class includes:
        - On-the-fly registration of parameters/buffers from an existing model.
        - Safe apply/restore methods to temporarily replace model weights.
        - Device management for multi-GPU and CPU environments.
        - Full checkpoint serialization support.

    Args:
        model (nn.Module): The model whose parameters are to be tracked.
        decay (float, optional): Smoothing coefficient (0 ≤ decay ≤ 1).
            Higher values make EMA updates slower. Default is 0.999.
        use_num_updates (bool, optional): Whether to adapt decay during early
            updates (useful for warm-up). Default is True.
        device (str | torch.device | None, optional): Optional target device for
            storing EMA parameters (e.g., "cpu" for offloading). Default is None.

    Attributes:
        decay (float): EMA smoothing coefficient.
        num_updates (int | None): Counter of EMA updates, used to adapt decay.
        device (torch.device | None): Device where EMA tensors are stored.
        shadow_params (Dict[str, torch.Tensor]): Smoothed parameter tensors.
        shadow_buffers (Dict[str, torch.Tensor]): Smoothed buffer tensors.
        collected_params (Dict[str, torch.Tensor]): Temporary cache for original
            parameters during apply/restore operations.
        collected_buffers (Dict[str, torch.Tensor]): Temporary cache for original
            buffers during apply/restore operations.
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.999,
        *,
        use_num_updates: bool = True,
        device: str | torch.device | None = None,
    ) -> None:
        if not 0.0 <= decay <= 1.0:
            raise ValueError("decay must be between 0 and 1 (inclusive)")

        self.decay = float(decay)
        self.num_updates = 0 if use_num_updates else None
        self.device = torch.device(device) if device is not None else None

        self.shadow_params: Dict[str, torch.Tensor] = {}
        self.shadow_buffers: Dict[str, torch.Tensor] = {}
        self.collected_params: Dict[str, torch.Tensor] = {}
        self.collected_buffers: Dict[str, torch.Tensor] = {}

        self._register(model)

    def _register(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            shadow = param.detach().clone()
            if self.device is not None:
                shadow = shadow.to(self.device)
            self.shadow_params[name] = shadow

        for name, buffer in model.named_buffers():
            shadow = buffer.detach().clone()
            if self.device is not None:
                shadow = shadow.to(self.device)
            self.shadow_buffers[name] = shadow

    def update(self, model: nn.Module) -> None:
        """Update the EMA weights using the latest parameters from ``model``.

        Performs an in-place exponential moving average update on all
        trainable parameters and buffers tracked in ``shadow_params`` and
        ``shadow_buffers``. If ``use_num_updates=True``, adapts the decay
        coefficient during early steps for smoother warm-up.

        Args:
            model (nn.Module): Model whose parameters and buffers are used to
                update the EMA state.

        Notes:
            - Dynamically adds new parameters or buffers if they were not
              present during initialization.
            - Operates in ``torch.no_grad()`` context to avoid gradient tracking.
        """
        if self.num_updates is not None:
            self.num_updates += 1
            decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        else:
            decay = self.decay

        one_minus_decay = 1.0 - decay

        with torch.no_grad():
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue

                if name not in self.shadow_params:
                    # Parameter may have been added dynamically (rare, but safeguard)
                    shadow = param.detach().clone()
                    if self.device is not None:
                        shadow = shadow.to(self.device)
                    self.shadow_params[name] = shadow

                shadow_param = self.shadow_params[name]
                param_data = param.detach()
                if param_data.device != shadow_param.device:
                    param_data = param_data.to(shadow_param.device)

                shadow_param.lerp_(param_data, one_minus_decay)

            for name, buffer in model.named_buffers():
                if name not in self.shadow_buffers:
                    shadow = buffer.detach().clone()
                    if self.device is not None:
                        shadow = shadow.to(self.device)
                    self.shadow_buffers[name] = shadow

                shadow_buffer = self.shadow_buffers[name]
                buffer_data = buffer.detach()
                if buffer_data.device != shadow_buffer.device:
                    buffer_data = buffer_data.to(shadow_buffer.device)
                shadow_buffer.copy_(buffer_data)

    def apply_to(self, model: nn.Module) -> None:
        """Replace model parameters with EMA-smoothed versions (in-place).

        Temporarily swaps the current model parameters and buffers with their
        EMA counterparts for evaluation or checkpoint export. The original
        tensors are cached internally and can be restored later with
        :meth:`restore`.

        Args:
            model (nn.Module): Model whose parameters will be replaced.

        Raises:
            RuntimeError: If EMA weights are already applied and not yet restored.
        """
        if self.collected_params or self.collected_buffers:
            raise RuntimeError(
                "EMA weights already applied; call restore() before reapplying."
            )

        for name, param in model.named_parameters():
            if not param.requires_grad or name not in self.shadow_params:
                continue
            self.collected_params[name] = param.detach().clone()
            param.data.copy_(self.shadow_params[name].to(param.device))

        for name, buffer in model.named_buffers():
            if name not in self.shadow_buffers:
                continue
            self.collected_buffers[name] = buffer.detach().clone()
            buffer.data.copy_(self.shadow_buffers[name].to(buffer.device))

    def restore(self, model: nn.Module) -> None:
        """Restore the model’s original parameters after an EMA swap.

        Reverts the parameter and buffer changes made by :meth:`apply_to`
        by restoring the cached tensors. This is a no-op if EMA weights
        were never applied.

        Args:
            model (nn.Module): Model whose parameters will be restored.
        """
        for name, param in model.named_parameters():
            cached = self.collected_params.pop(name, None)
            if cached is None:
                continue
            param.data.copy_(cached.to(param.device))

        for name, buffer in model.named_buffers():
            cached = self.collected_buffers.pop(name, None)
            if cached is None:
                continue
            buffer.data.copy_(cached.to(buffer.device))

    @contextmanager
    def average_parameters(self, model: nn.Module) -> Iterator[None]:
        """Context manager to temporarily apply EMA weights to ``model``.

        This convenience wrapper allows for automatic restoration after use.
        Example:
        ```python
        with ema.average_parameters(model):
            validate(model)
        ```

        Args:
            model (nn.Module): The model to temporarily replace parameters for.

        Yields:
            None: Executes the body of the context with EMA weights applied.
        """
        self.apply_to(model)
        try:
            yield
        finally:
            self.restore(model)

    def to(self, device: str | torch.device) -> None:
        """Move EMA-tracked tensors to a target device.

        Transfers all shadow parameters and buffers to the specified device,
        updating the internal ``self.device`` reference.

        Args:
            device (str | torch.device): Target device (e.g., "cuda", "cpu").
        """
        target_device = torch.device(device)
        for name, tensor in list(self.shadow_params.items()):
            self.shadow_params[name] = tensor.to(target_device)
        for name, tensor in list(self.shadow_buffers.items()):
            self.shadow_buffers[name] = tensor.to(target_device)
        self.device = target_device

    def state_dict(self) -> Dict[str, object]:
        """Return a serializable state dictionary for checkpointing.

        Packages all relevant EMA state into a plain dictionary, compatible
        with PyTorch’s standard checkpoint format. Converts all tensors to CPU
        for safe serialization.

        Returns:
            Dict[str, object]: Dictionary containing EMA decay, update count,
            device info, and copies of shadow parameters/buffers.
        """
        return {
            "decay": self.decay,
            "num_updates": self.num_updates,
            "device": str(self.device) if self.device is not None else None,
            "shadow_params": {
                k: v.detach().cpu() for k, v in self.shadow_params.items()
            },
            "shadow_buffers": {
                k: v.detach().cpu() for k, v in self.shadow_buffers.items()
            },
        }

    def load_state_dict(self, state_dict: Dict[str, object]) -> None:
        """Load EMA state from a previously saved checkpoint.

        Reconstructs the EMA tracking state from a saved dictionary, restoring
        all tracked parameters, buffers, and metadata such as decay, device,
        and update count.

        Args:
            state_dict (Dict[str, object]): Dictionary as produced by
                :meth:`state_dict`.

        Notes:
            - Tensors are moved to the current or saved device automatically.
            - Clears existing collected (applied) caches to avoid stale state.
        """
        self.decay = float(state_dict["decay"])
        self.num_updates = state_dict["num_updates"]
        device_str = state_dict.get("device", None)
        self.device = torch.device(device_str) if device_str is not None else None

        self.shadow_params = {
            name: (
                tensor.clone().to(self.device)
                if self.device is not None
                else tensor.clone()
            )
            for name, tensor in state_dict.get("shadow_params", {}).items()
        }
        self.shadow_buffers = {
            name: (
                tensor.clone().to(self.device)
                if self.device is not None
                else tensor.clone()
            )
            for name, tensor in state_dict.get("shadow_buffers", {}).items()
        }

        self.collected_params = {}
        self.collected_buffers = {}
