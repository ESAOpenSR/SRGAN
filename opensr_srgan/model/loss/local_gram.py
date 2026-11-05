"""Local feature statistics for texture synthesis, without cross-region mixing."""

import torch
import torch.nn.functional as F


def local_gram_distance(prediction, target, grid=4):
    """Relative L1 distance between Gram matrices in matching nonoverlapping tiles.

    Target features are detached. Float32 statistics avoid mixed-precision
    overflow. Normalizing by target Gram magnitude makes layer scales comparable.
    """
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("Gram features must have matching BCHW shapes")
    b, c, h, w = prediction.shape
    if grid < 1 or h % grid or w % grid:
        raise ValueError("Texture grid must divide both feature dimensions")
    with torch.autocast(device_type=prediction.device.type, enabled=False):
        def gram(features):
            tiles = F.unfold(features.float(), (h // grid, w // grid),
                             stride=(h // grid, w // grid))
            tiles = tiles.transpose(1, 2).reshape(b * grid * grid, c, -1)
            return tiles @ tiles.transpose(1, 2) / (c * tiles.shape[-1])
        reference = gram(target.detach())
        generated = gram(prediction)
        scale = reference.abs().mean(dim=(1, 2)).clamp_min(1e-6)
        return ((generated - reference).abs().mean(dim=(1, 2)) / scale).mean()
