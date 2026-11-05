# SPDX-License-Identifier: MIT
"""Run an SR-GAN preset on one small BCHW tensor saved with ``torch.save``."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from opensr_srgan import load_inference_model


def _tensor_from_payload(payload: Any, *, source: str) -> torch.Tensor:
    """Extract an LR tensor from common dataset serialization layouts."""

    if isinstance(payload, torch.Tensor):
        return payload
    if isinstance(payload, Mapping):
        for key in ("lr", "lr_imgs", "input", "image", "x"):
            if key in payload:
                return _tensor_from_payload(payload[key], source=f"{source}[{key!r}]")
        tensors = [value for value in payload.values() if isinstance(value, torch.Tensor)]
        if len(tensors) == 1:
            return tensors[0]
        raise ValueError(
            f"{source} is a dictionary without a recognized LR key. "
            "Use one of: lr, lr_imgs, input, image, x."
        )
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        for index, value in enumerate(payload):
            if isinstance(value, torch.Tensor):
                return value
        raise ValueError(f"{source} contains no tensor; expected an LR tensor or (lr, hr).")
    raise TypeError(f"{source} must contain a torch.Tensor, got {type(payload).__name__}.")


def _load_input(path: Path) -> torch.Tensor:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch versions before ``weights_only`` accepted this argument.
        payload = torch.load(path, map_location="cpu")
    return _tensor_from_payload(payload, source=str(path))


def _as_bchw(tensor: torch.Tensor, *, channels: int) -> torch.Tensor:
    if tensor.dtype == torch.bool or tensor.is_complex():
        raise TypeError(f"Expected real-valued input, got dtype {tensor.dtype}.")
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 4:
        raise ValueError(f"Expected CHW or BCHW input, got shape {tuple(tensor.shape)}.")
    if tensor.shape[1] != channels:
        raise ValueError(f"Preset expects {channels} channels, got shape {tuple(tensor.shape)}.")
    tensor = tensor.float()
    if not torch.isfinite(tensor).all():
        raise ValueError("Input contains NaN or infinite values.")
    return tensor.contiguous()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=("SEN2-SR", "RGB-NIR", "SWIR"), default="SEN2-SR")
    parser.add_argument("--device", default="cpu", help="Torch device, for example cpu or cuda.")
    parser.add_argument("--input", type=Path, help="A torch.save tensor, (lr, hr), or supported mapping.")
    parser.add_argument("--output", type=Path, default=Path("sr.pt"), help="Destination BCHW tensor.")
    parser.add_argument("--revision", help="Optional Hugging Face commit, tag, or branch to resolve.")
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Use forward on an already normalized 0..1 tensor; skip predict_step handling.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use a generated zero tensor only to verify installation and output shape.",
    )
    args = parser.parse_args()
    if args.smoke == (args.input is not None):
        parser.error("provide exactly one of --input or --smoke")
    return args


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device)
    model = load_inference_model(
        args.preset, map_location=device, revision=args.revision
    ).to(device)
    model.eval()
    channels = int(model.config.Model.in_bands)
    scale = int(model.config.Generator.scaling_factor)

    input_tensor = (
        torch.zeros((1, channels, 16, 16), dtype=torch.float32)
        if args.smoke
        else _load_input(args.input)
    )
    input_tensor = _as_bchw(input_tensor, channels=channels)
    with torch.inference_mode():
        if args.raw:
            output = model(input_tensor.to(device)).detach().cpu()
            convention = "normalized network output (forward; no post-processing)"
        else:
            output = model.predict_step(input_tensor)
            convention = "raw DN/reflectance output (predict_step)"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    source = "generated smoke tensor" if args.smoke else str(args.input)
    print(
        f"Saved {tuple(output.shape)} to {args.output} from {source}; "
        f"preset={args.preset}, scale={scale}, convention={convention}."
    )


if __name__ == "__main__":
    main()
