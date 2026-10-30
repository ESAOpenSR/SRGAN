from pathlib import Path
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia.metrics as km

from ...data.utils import Normalizer

LOSS_EPS = 1e-12


def _ensure_finite(tensor: torch.Tensor, *, eps: float = LOSS_EPS) -> torch.Tensor:
    """Clamp NaN/Inf values that can emerge from loss computations.

    Args:
        tensor: Input tensor potentially containing NaNs/Infs.
        eps: Small positive constant used when replacing NaNs.

    Returns:
        torch.Tensor: Tensor with finite values only.
    """

    if not torch.is_floating_point(tensor):
        return tensor
    return torch.nan_to_num(tensor, nan=eps, posinf=1.0 / eps, neginf=-1.0 / eps)


def _cfg_get(cfg, keys, default=None):
    """Safely retrieve a nested configuration value.

    Traverses a mixture of dict-like and attribute-based configs using a list
    of keys, returning ``default`` if any link in the chain is missing.

    Args:
        cfg: Root configuration object (supports ``.__getattr__`` and/or mapping).
        keys (Iterable[str]): Sequence of keys/attributes to traverse.
        default (Any, optional): Fallback value if the path is absent. Defaults to ``None``.

    Returns:
        Any: The found value or ``default`` if not present.
    """
    cur = cfg
    for k in keys:
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(k, None)
        else:
            cur = getattr(cur, k, None)
    return default if cur is None else cur


class GeneratorContentLoss(nn.Module):
    """Composite generator content loss with perceptual metric selection.

    Combines multiple terms to form the generator's content objective:
    ``total = l1_w * L1 + sam_w * SAM + perc_w * Perceptual + tv_w * TV``.
    Also computes auxiliary quality metrics (PSNR/SSIM) for logging/evaluation.

    Loss weights, max value, window sizes, band selection settings, and the
    perceptual backend (VGG or LPIPS) are read from the provided config.

    Args:
        cfg: Configuration object (OmegaConf/dict-like) with fields under:
            - ``Training.Losses.{l1_weight,sam_weight,perceptual_weight,tv_weight}``
            - ``Training.Losses.{max_val,ssim_win,fixed_idx}``
            - ``Training.Losses.perceptual_metric`` in {``"vgg"``, ``"lpips"``}
            - ``TruncatedVGG.{i,j}`` (if VGG is used)
        testing (bool, optional): If True, do not load pretrained VGG weights
            (avoids downloads in CI/tests). Defaults to False.

    Attributes:
        l1_w, sam_w, perc_w, tv_w (float): Loss term weights.
        max_val (float): Dynamic range for PSNR/SSIM.
        ssim_win (int): Window size for SSIM.
        fixed_idx (torch.LongTensor|None): Explicit fixed triplet override. With
            ``None`` (default), inputs with more than three bands use shuffled
            triplets, shared between SR and HR on each call.
        perc_metric (str): Selected perceptual backend (``"vgg"`` or ``"lpips"``).
        perceptual_model (nn.Module): Back-end feature network/metric.
        normalizer (Normalizer): Shared normalizer for evaluation metrics.
    """

    def __init__(self, cfg, testing=False):
        super().__init__()
        self.cfg = cfg

        # --- weights & settings from config ---
        # (fallback to deprecated Training.perceptual_loss_weight if needed)
        self.l1_w = float(_cfg_get(cfg, ["Training", "Losses", "l1_weight"], 1.0))
        self.sam_w = float(_cfg_get(cfg, ["Training", "Losses", "sam_weight"], 0.05))
        perc_w_cfg = _cfg_get(
            cfg,
            ["Training", "Losses", "perceptual_weight"],
            _cfg_get(cfg, ["Training", "perceptual_loss_weight"], 0.1),
        )
        self.perc_w = float(perc_w_cfg)
        self.tv_w = float(_cfg_get(cfg, ["Training", "Losses", "tv_weight"], 0.0))
        self.vgg_input_normalization = bool(
            _cfg_get(cfg, ["Training", "Losses", "vgg_input_normalization"], False)
        )
        self.texture_w = float(_cfg_get(cfg, ["Training", "Losses", "texture_weight"], 0.0))
        self.texture_grid = int(_cfg_get(cfg, ["Training", "Losses", "texture_grid"], 4))
        if not math.isfinite(self.texture_w) or self.texture_w < 0 or self.texture_grid < 1:
            raise ValueError("Texture weight must be finite/nonnegative and grid positive")

        self.max_val = float(_cfg_get(cfg, ["Training", "Losses", "max_val"], 1.0))
        self.ssim_win = int(_cfg_get(cfg, ["Training", "Losses", "ssim_win"], 11))

        fixed_idx = _cfg_get(cfg, ["Training", "Losses", "fixed_idx"], None)
        if fixed_idx is not None:
            fixed_idx = torch.as_tensor(fixed_idx, dtype=torch.long)
            if fixed_idx.numel() != 3:
                raise ValueError("fixed_idx must have length 3")
        self.register_buffer(
            "fixed_idx", fixed_idx if fixed_idx is not None else None, persistent=False
        )

        # Only init model if perceptual weight > 0
        if self.perc_w != 0.0 or self.texture_w != 0.0:
            # --- configure perceptual metric ---
            self.perc_metric = str(
                _cfg_get(cfg, ["Training", "Losses", "perceptual_metric"], "vgg")
            ).lower()

            if self.perc_metric == "vgg":
                from .vgg import TruncatedVGG19

                i = int(_cfg_get(cfg, ["TruncatedVGG", "i"], 5))
                j = int(_cfg_get(cfg, ["TruncatedVGG", "j"], 4))
                before_activation = bool(
                    _cfg_get(
                        cfg,
                        ["Training", "Losses", "vgg_before_activation"],
                        False,
                    )
                )
                self.perceptual_model = TruncatedVGG19(
                    i=i,
                    j=j,
                    weights=not testing,
                    before_activation=before_activation,
                )
                self.vgg_before_activation = before_activation
            elif self.perc_metric == "lpips":
                import lpips

                self.perceptual_model = lpips.LPIPS(net="alex")
            else:
                raise ValueError(f"Unsupported perceptual metric: {self.perc_metric}")

            # Set to eval and freeze params
            for p in self.perceptual_model.parameters():
                p.requires_grad = False
            self.perceptual_model.eval()
            if self.texture_w and self.perc_metric != "vgg":
                raise ValueError("Local Gram texture loss requires the VGG backbone")
            if self.texture_w and len(self.perceptual_model.truncated_vgg19) < 17:
                raise ValueError("Local Gram loss requires VGG through conv3_4 (or deeper)")
        else:  # Set to None when no perc. wanted
            self.perceptual_model = None
            self.perc_metric = None

        # ImageNet VGG expects channel-wise normalized RGB inputs. Keep these
        # as buffers so they follow the loss module across devices/dtypes.
        if self.perc_metric == "vgg":
            self.register_buffer(
                "_vgg_mean",
                torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
                persistent=False,
            )
            self.register_buffer(
                "_vgg_std",
                torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
                persistent=False,
            )

        # Shared normalizer for computing evaluation metrics
        self.normalizer = Normalizer(cfg)

    def train(self, mode: bool = True):
        """Keep the frozen perceptual backbone in inference mode.

        ``LightningModule.train()`` recursively toggles child modules every
        epoch, which otherwise silently re-enables dropout/batch-stat updates
        in a perceptual network.
        """
        super().train(mode)
        if self.perceptual_model is not None:
            self.perceptual_model.eval()
            for parameter in self.perceptual_model.parameters():
                parameter.requires_grad_(False)
        return self

    # ---------- public API ----------
    def return_loss(
        self, sr: torch.Tensor, hr: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the weighted content loss and return raw component metrics.

        Builds the autograd graph for terms with non-zero weights and returns the
        scalar total along with a dict of unweighted component values.

        Args:
            sr (torch.Tensor): Super-resolved prediction, shape ``(B, C, H, W)``.
            hr (torch.Tensor): High-resolution target, shape ``(B, C, H, W)``.

        Returns:
            Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
                - Total loss tensor (requires grad).
                - Dict with component tensors: ``{"l1","sam","perceptual","tv", "psnr","ssim"}``
                (component values detached except the ones used in the graph).
        """
        comps = self._compute_components(sr, hr, build_graph=True)
        loss = (
            self.l1_w * comps["l1"]
            + self.sam_w * comps["sam"]
            + self.perc_w * comps["perceptual"]
            + self.tv_w * comps["tv"]
            + self.texture_w * comps.get("texture", 0.0)
        )
        loss = _ensure_finite(loss)
        metrics = {k: v.detach() for k, v in comps.items()}
        return loss, metrics

    @torch.no_grad()
    def return_metrics(
        self, sr: torch.Tensor, hr: torch.Tensor, prefix: str = ""
    ) -> dict[str, torch.Tensor]:
        """
        Compute all unweighted metric components and (optionally) prefix their keys.

        Args:
            sr, hr: tensors in the same range as the generator output/HR targets.
            prefix: key prefix like 'train/' or 'val'. If non-empty and doesn't end
                    with '/', a '/' is added automatically.

        Returns:
            dict mapping metric names -> tensors (detached), e.g. {'train/l1': ...}.
            Includes raw PSNR/SSIM metrics computed on stretched/clipped inputs.
        """
        comps = self._compute_components(sr, hr, build_graph=False)
        p = (prefix + "/") if prefix and not prefix.endswith("/") else prefix
        return {f"{p}{k}": v.detach() for k, v in comps.items()}

    # ---------- internals ----------
    @staticmethod
    def _tv_loss(x: torch.Tensor) -> torch.Tensor:
        """Total variation (TV) regularizer.

        Computes the L1 norm of first-order finite differences along height and width.

        Args:
            x (torch.Tensor): Input tensor, shape ``(B, C, H, W)``.

        Returns:
            torch.Tensor: Scalar TV loss (mean over batch/channels/pixels).
        """
        dh = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
        dw = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
        return _ensure_finite(dh + dw)

    @staticmethod
    def _sam_loss(
        sr: torch.Tensor, hr: torch.Tensor, eps: float = 1e-8
    ) -> torch.Tensor:
        """Spectral Angle Mapper (SAM) in radians.

        Flattens spatial dims and computes the mean angle between spectral vectors
        of SR and HR across all pixels.

        Args:
            sr (torch.Tensor): Super-resolved tensor, shape ``(B, C, H, W)``.
            hr (torch.Tensor): Target tensor, shape ``(B, C, H, W)``.
            eps (float, optional): Numerical stability epsilon. Defaults to ``1e-8``.

        Returns:
            torch.Tensor: Scalar mean SAM (radians).
        """
        B, C, H, W = sr.shape
        # bfloat16/float16 do not have enough mantissa precision for a safe
        # cosine clamp near one; keep the angle computation in float32.
        work_dtype = torch.float32 if sr.dtype in (torch.float16, torch.bfloat16) else sr.dtype
        sr_f = sr.to(work_dtype).reshape(B, C, -1)
        hr_f = hr.to(work_dtype).reshape(B, C, -1)
        dot = (sr_f * hr_f).sum(dim=1)
        dtype_eps = torch.finfo(work_dtype).eps
        eps = max(eps, dtype_eps)
        sr_n = sr_f.norm(dim=1).clamp_min(eps)
        hr_n = hr_f.norm(dim=1).clamp_min(eps)
        denom = torch.clamp(sr_n * hr_n, min=eps)
        cos = torch.clamp(dot / denom, -1.0 + 1e-6, 1.0 - 1e-6)
        ang = torch.acos(cos)
        return _ensure_finite(ang.mean())

    def _prepare_perceptual_input(
        self, sr: torch.Tensor, hr: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select three channels for perceptual computation.

        Inputs with more than three bands use three unique indices from a fresh
        random permutation by default, in both training and metric evaluation.
        SR and HR share the selected indices and order. An explicit
        ``fixed_idx`` overrides this policy. Three-band inputs stay unchanged;
        one- and two-band inputs repeat channels to form a triplet.

        Args:
            x (torch.Tensor): Input tensor, shape ``(B, C, H, W)``.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: SR and HR tensors with three channels.
        """
        _, C, _, _ = sr.shape
        if C == 1:
            # repeat single channel 3 times
            sr = sr.repeat(1, 3, 1, 1)
            hr = hr.repeat(1, 3, 1, 1)
            return sr, hr
        elif C == 2:
            # repeat first channel to make 3
            sr = torch.cat([sr, sr[:, :1, :, :]], dim=1)
            hr = torch.cat([hr, hr[:, :1, :, :]], dim=1)
            return sr, hr
        elif C == 3:
            # already 3 channels, return as is
            return sr, hr
        else:
            if self.fixed_idx is not None:
                idx = self.fixed_idx.to(sr.device)
                if int(idx.min()) < 0 or int(idx.max()) >= C:
                    raise ValueError(
                        f"fixed_idx values must be valid channel indices for C={C}."
                    )
            else:
                # when over 3 channels, randomly select 3 unique indices
                idx = torch.randperm(C, device=sr.device)[:3]
            sr = sr[:, idx, :, :]
            hr = hr[:, idx, :, :]
            return sr, hr

    def _perceptual_distance(
        self, sr_3: torch.Tensor, hr_3: torch.Tensor, *, build_graph: bool
    ) -> torch.Tensor:
        """Compute perceptual distance between SR and HR (3-channel inputs).

        Uses the configured backend:
            - ``"vgg"``: MSE between intermediate VGG features.
            - ``"lpips"``: Learned LPIPS distance (expects inputs in [-1, 1]).

        The computation detaches HR features and optionally detaches SR path if
        ``build_graph`` is False or the perceptual weight is zero.

        Args:
            sr_3 (torch.Tensor): SR slice with 3 channels, shape ``(B, 3, H, W)``, values in [0, 1].
            hr_3 (torch.Tensor): HR slice with 3 channels, shape ``(B, 3, H, W)``, values in [0, 1].
            build_graph (bool): Whether to keep gradients for SR.

        Returns:
            torch.Tensor: Scalar perceptual distance (mean over batch/spatial).
        """
        requires_grad = build_graph and self.perc_w != 0.0

        if self.perc_metric == "vgg":
            if self.vgg_input_normalization:
                sr_3 = (sr_3 - self._vgg_mean.to(dtype=sr_3.dtype)) / self._vgg_std.to(dtype=sr_3.dtype)
                hr_3 = (hr_3 - self._vgg_mean.to(dtype=hr_3.dtype)) / self._vgg_std.to(dtype=hr_3.dtype)
            if requires_grad:
                sr_features = self.perceptual_model(sr_3)
            else:
                with torch.no_grad():
                    sr_features = self.perceptual_model(sr_3)
            with torch.no_grad():
                hr_features = self.perceptual_model(hr_3)
            distance = F.mse_loss(sr_features, hr_features)
        elif self.perc_metric == "lpips":
            sr_norm = sr_3.mul(2.0).sub(1.0)
            hr_norm = hr_3.mul(2.0).sub(1.0).detach()
            if requires_grad:
                distance = self.perceptual_model(sr_norm, hr_norm)
            else:
                with torch.no_grad():
                    distance = self.perceptual_model(sr_norm, hr_norm)
            distance = distance.mean()
        else:
            raise RuntimeError(f"Unhandled perceptual metric: {self.perc_metric}")

        if not requires_grad:
            distance = distance.detach()
        return _ensure_finite(distance)

    def _texture_distance(self, sr_3, hr_3):
        """Average local Gram distances at preactivation VGG conv2_2/conv3_4."""
        from .local_gram import local_gram_distance
        mean = self._vgg_mean.to(dtype=sr_3.dtype)
        std = self._vgg_std.to(dtype=sr_3.dtype)
        generated, reference = sr_3, hr_3.detach()
        if self.vgg_input_normalization:
            generated, reference = (generated - mean) / std, (reference - mean) / std
        distances = []
        # VGG19 indices 7/16 are conv2_2/conv3_4, before their ReLUs.
        for index, layer in enumerate(self.perceptual_model.truncated_vgg19):
            generated = layer(generated)
            with torch.no_grad():
                reference = layer(reference)
            if index in (7, 16):
                distances.append(local_gram_distance(generated, reference, self.texture_grid))
            if index == 16:
                break
        return torch.stack(distances).mean()

    def _compute_components(
        self, sr: torch.Tensor, hr: torch.Tensor, *, build_graph: bool
    ) -> dict[str, torch.Tensor]:
        """Compute individual content components and auxiliary quality metrics.

        Produces a dictionary with: L1, SAM, Perceptual, TV (optionally with grads),
        and PSNR/SSIM (always without grads). Per-component autograd is enabled only
        if ``build_graph`` is True and the corresponding weight is non-zero.

        Args:
            sr (torch.Tensor): Super-resolved prediction, shape ``(B, C, H, W)``.
            hr (torch.Tensor): High-resolution target, shape ``(B, C, H, W)``.
            build_graph (bool): Whether to allow gradients for weighted components.

        Returns:
            Dict[str, torch.Tensor]: Keys ``{"l1","sam","perceptual","tv","psnr","ssim"}``.
            Component tensors are scalar means; PSNR/SSIM are detached.
        """
        comps: dict[str, torch.Tensor] = {}

        def _compute(weight: float, fn) -> torch.Tensor:
            requires_grad = build_graph and weight != 0.0
            if requires_grad:
                return fn()
            with torch.no_grad():
                return fn().detach()

        # Core reconstruction metrics (always unweighted)
        comps["l1"] = _compute(self.l1_w, lambda: _ensure_finite(F.l1_loss(sr, hr)))
        comps["sam"] = _compute(self.sam_w, lambda: self._sam_loss(sr, hr))

        # Perceptual distance on 3 selected bands
        if self.perceptual_model != None and self.perc_w != 0.0:
            sr_3, hr_3 = self._prepare_perceptual_input(sr=sr, hr=hr)
            comps["perceptual"] = self._perceptual_distance(
                sr_3, hr_3, build_graph=build_graph
            )
        else:
            comps["perceptual"] = torch.tensor(0.0, device=sr.device)

        if self.texture_w:
            if self.perc_w == 0:
                sr_3, hr_3 = self._prepare_perceptual_input(sr=sr, hr=hr)
            comps["texture"] = _compute(
                self.texture_w, lambda: self._texture_distance(sr_3, hr_3)
            )

        # Total variation
        comps["tv"] = _compute(self.tv_w, lambda: self._tv_loss(sr))

        # --- Quality metrics ---
        with torch.no_grad():
            safe_max_val = max(self.max_val, LOSS_EPS)
            sr_metric = torch.clamp(sr, 0.0, safe_max_val)
            hr_metric = torch.clamp(hr, 0.0, safe_max_val)
            psnr = km.psnr(sr_metric, hr_metric, max_val=safe_max_val)
            ssim = km.ssim(
                sr_metric,
                hr_metric,
                window_size=self.ssim_win,
                max_val=safe_max_val,
            )

            if psnr.dim() > 0:
                psnr = psnr.mean()
            if ssim.dim() > 0:
                ssim = ssim.mean()

            comps["psnr"] = _ensure_finite(psnr).detach()
            comps["ssim"] = _ensure_finite(ssim).detach()

        return comps

