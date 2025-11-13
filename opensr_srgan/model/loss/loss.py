from pathlib import Path

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
            - ``Training.Losses.{max_val,ssim_win,randomize_bands,fixed_idx}``
            - ``Training.Losses.perceptual_metric`` in {``"vgg"``, ``"lpips"``}
            - ``TruncatedVGG.{i,j}`` (if VGG is used)
        testing (bool, optional): If True, do not load pretrained VGG weights
            (avoids downloads in CI/tests). Defaults to False.

    Attributes:
        l1_w, sam_w, perc_w, tv_w (float): Loss term weights.
        max_val (float): Dynamic range for PSNR/SSIM.
        ssim_win (int): Window size for SSIM.
        randomize_bands (bool): Whether to sample random bands for perceptual loss.
        fixed_idx (torch.LongTensor|None): Fixed 3-channel indices if not randomized.
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

        self.max_val = float(_cfg_get(cfg, ["Training", "Losses", "max_val"], 1.0))
        self.ssim_win = int(_cfg_get(cfg, ["Training", "Losses", "ssim_win"], 11))

        fixed_idx = _cfg_get(cfg, ["Training", "Losses", "fixed_idx"], None)
        if fixed_idx is not None:
            fixed_idx = torch.as_tensor(fixed_idx, dtype=torch.long)
            assert fixed_idx.numel() == 3, "fixed_idx must have length 3"
        self.register_buffer(
            "fixed_idx", fixed_idx if fixed_idx is not None else None, persistent=False
        )

        # Only init model if perceptual weight > 0
        if self.perc_w != 0.0:
            # --- configure perceptual metric ---
            self.perc_metric = str(
                _cfg_get(cfg, ["Training", "Losses", "perceptual_metric"], "vgg")
            ).lower()

            if self.perc_metric == "vgg":
                from .vgg import TruncatedVGG19

                i = int(_cfg_get(cfg, ["TruncatedVGG", "i"], 5))
                j = int(_cfg_get(cfg, ["TruncatedVGG", "j"], 4))
                self.perceptual_model = TruncatedVGG19(i=i, j=j, weights=not testing)
            elif self.perc_metric == "lpips":
                import lpips

                self.perceptual_model = lpips.LPIPS(net="alex")
            else:
                raise ValueError(f"Unsupported perceptual metric: {self.perc_metric}")

            # Set to eval and freeze params
            for p in self.perceptual_model.parameters():
                p.requires_grad = False
            self.perceptual_model.eval()
        else:  # Set to None when no perc. wanted
            self.perceptual_model = None
            self.perc_metric = None

        # Shared normalizer for computing evaluation metrics
        self.normalizer = Normalizer(cfg)

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
        sr_f = sr.view(B, C, -1)
        hr_f = hr.view(B, C, -1)
        dot = (sr_f * hr_f).sum(dim=1)
        if torch.is_floating_point(sr):
            dtype_eps = torch.finfo(sr.dtype).eps
        else:
            dtype_eps = torch.finfo(torch.float32).eps
        eps = max(eps, dtype_eps)
        sr_n = sr_f.norm(dim=1).clamp_min(eps)
        hr_n = hr_f.norm(dim=1).clamp_min(eps)
        denom = torch.clamp(sr_n * hr_n, min=eps)
        cos = torch.clamp(dot / denom, -1 + 1e-7, 1 - 1e-7)
        ang = torch.acos(cos)
        return _ensure_finite(ang.mean())

    def _prepare_perceptual_input(
        self, sr: torch.Tensor, hr: torch.Tensor
    ) -> torch.Tensor:
        """Select three channels for perceptual computation.

        If the input has exactly 3 channels, returns them unchanged. Otherwise,
        selects either random unique indices  or
        the fixed indices stored in ``self.fixed_idx``.

        Args:
            x (torch.Tensor): Input tensor, shape ``(B, C, H, W)``.

        Returns:
            torch.Tensor: Tensor with three channels, shape ``(B, 3, H, W)``.
        """
        B, C, H, W = sr.shape
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


if __name__ == "__main__":
    # simple test
    from omegaconf import OmegaConf

    config_path = Path(__file__).resolve().parents[2] / "configs" / "config_20m.yaml"
    with open(config_path, "r") as f:
        cfg = OmegaConf.load(f)

    loss_fn = GeneratorContentLoss(cfg)

    B, C, H, W = 2, 13, 64, 64
    sr = torch.rand(B, C, H, W)
    hr = torch.rand(B, C, H, W)

    loss, metrics = loss_fn.return_loss(sr, hr)
    print("Loss:", loss.item())
    print("Metrics:", {k: v.item() for k, v in metrics.items()})

    eval_metrics = loss_fn.return_metrics(sr, hr, prefix="val")
    print("Eval metrics:", {k: v.item() for k, v in eval_metrics.items()})
