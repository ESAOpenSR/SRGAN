"""VGG based perceptual feature extractor utilities."""

from torch import nn
import torchvision


class TruncatedVGG19(nn.Module):
    """A truncated VGG-19 feature extractor for perceptual loss computation.

    This class wraps a pretrained VGG-19 network from ``torchvision.models``
    and truncates it at a specific convolutional layer ``(i, j)`` within
    the feature hierarchy, following the convention in perceptual and
    style-transfer literature (e.g., layer *relu{i}_{j}*).

    The truncated model outputs intermediate feature maps that capture
    perceptual similarity more effectively than raw pixel losses. These
    feature activations are typically used in content or perceptual loss
    terms such as:
    ```
    L_perceptual = || Φ_j(x_sr) − Φ_j(x_hr) ||_1
    ```
    where ``Φ_j`` denotes the truncated VGG feature extractor.

    Args:
        i (int, optional): The convolutional block index (1-5) at which to
            truncate. Each block corresponds to a region between max-pooling
            layers. Defaults to ``5``.
        j (int, optional): The convolution layer index within the chosen block.
            Defaults to ``4``.
        weights (bool, optional): Whether to load pretrained ImageNet weights.
            If ``False``, the model is initialized without downloading weights
            (useful for testing environments). Defaults to ``True``.

    Attributes:
        truncated_vgg19 (nn.Sequential): Sequential container of layers up to
        the specified truncation point.

    Raises:
        AssertionError: If the provided ``(i, j)`` combination does not match a
            valid convolutional layer in VGG-19.

    Example:
        >>> vgg = TruncatedVGG19(i=5, j=4)
        >>> feats = vgg(img_batch)  # [B, C, H, W] feature map
    """

    def __init__(self, i: int = 5, j: int = 4, weights=True) -> None:
        super().__init__()

        if weights:  # omit downloading for tests
            vgg19 = torchvision.models.vgg19(
                weights=torchvision.models.VGG19_Weights.DEFAULT
            )
        else:
            vgg19 = torchvision.models.vgg19(weights=None)

        maxpool_counter = 0
        conv_counter = 0
        truncate_at = 0
        for layer in vgg19.features.children():
            truncate_at += 1

            if isinstance(layer, nn.Conv2d):
                conv_counter += 1
            if isinstance(layer, nn.MaxPool2d):
                maxpool_counter += 1
                conv_counter = 0

            if maxpool_counter == i - 1 and conv_counter == j:
                break

        if not (maxpool_counter == i - 1 and conv_counter == j):
            raise ValueError(
                f"One or both of i={i} and j={j} are not valid choices for the VGG19!"
            )

        self.truncated_vgg19 = nn.Sequential(
            *list(vgg19.features.children())[: truncate_at + 1]
        )

    def forward(self, input):  # type: ignore[override]
        """Compute VGG-19 features up to the configured truncation layer.

        Args:
            input (torch.Tensor): Input tensor of shape ``(B, 3, H, W)``
                with values normalized to ImageNet statistics (mean/std).

        Returns:
            torch.Tensor: The feature map extracted from the specified
            intermediate layer of VGG-19.
        """
        return self.truncated_vgg19(input)


__all__ = ["TruncatedVGG19"]
