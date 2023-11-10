"""Encoder based on Efficientnet."""

from typing import Optional

import torch
import torchvision
from torch import nn

from robotic_transformer_pytorch.film_efficientnet import (
    film_conditioning_layer,
    film_efficientnet_encoder,
)

_MODELS = {
    "b3": film_efficientnet_encoder.EfficientNetB3,
}

_SIZES = {
    "b3": 300,
}


class EfficientNetEncoder(nn.Module):
    """Applies a pretrained Efficientnet based encoder."""

    def __init__(
        self,
        model_variant: str = "b3",
        freeze: bool = False,
        early_film: bool = True,
        weights: Optional[str] = "imagenet",
        include_top: bool = False,
        pooling: bool = False,
        **kwargs,
    ):
        """Initialize the model.

        Args:
          model_variant: One of 'b0-b7' of the efficient encoders. See
            https://arxiv.org/abs/1905.11946 to understand the variants.
          freeze: Whether or not to freeze the pretrained weights (seems to not work
            well).
          early_film: Whether to inject film layers into the efficientnet encoder
            (seems to be essential to getting strong performance).
          weights: Which pretrained weights to use. Either 'imagenet', a path to the
            pretrained weights, or None for from scratch.
          include_top: Whether to add the top fully connected layer. If True, this
            will cause encoding to fail and is used only for unit testing purposes.
          pooling: If false, returns feature map before global average pooling
          **kwargs: Torch specific model kwargs.
        """
        super(EfficientNetEncoder, self).__init__(**kwargs)
        if model_variant not in _MODELS:
            raise ValueError(f"Unknown variant {model_variant}")
        self.model_variant = model_variant
        self.early_film = early_film
        self.freeze = freeze
        self.conv1x1 = nn.Conv2d(
            in_channels=1536,
            out_channels=512,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding_mode="same",
            bias=False,
        )
        nn.init.kaiming_normal_(self.conv1x1.weight, mode="fan_out")
        nn.init.zeros__(self.conv1x1.bias)
        self.net = _MODELS[model_variant](
            include_top=include_top,
            weights=weights,
            include_film=early_film,
        )
        self.film_layer = film_conditioning_layer.FilmConditioning(num_channels=512)
        self._pooling = pooling

    def _prepare_image(self, image: torch.Tensor) -> torch.Tensor:
        """Resize the input image and check that the range is correct."""
        if len(image.shape) != 4 or image.shape[-1] != 3:
            raise ValueError("Provided image should have shape (b, h, w, 3).")
        size = _SIZES[self.model_variant]
        if image.shape[1] < size / 4 or image.shape[2] < size / 4:
            raise ValueError("Provided image is too small.")
        if image.shape[1] > size * 4 or image.shape[2] > size * 4:
            raise ValueError("Provided image is too large.")
        image = image.permute(0, 3, 1, 2)  # rehape to (b, 3, h, w)
        image = torchvision.transforms.Resize((size, size))(image)
        assert torch.min(image) >= 0.0 and torch.max(image) <= 1.0
        image *= 255  # The image is expected to be in range(0, 255).
        return image

    def _encode(
        self, image: torch.Tensor, context: torch.Tensor, training: bool
    ) -> torch.Tensor:
        """Run the image through the efficientnet encoder."""
        image = self._prepare_image(image)
        if self.early_film:
            return self.net((image, context), training=training)
        return self.net(image, training=training)

    def call(
        self,
        image: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        training: bool = True,
    ) -> torch.Tensor:
        if self.freeze:
            with torch.no_grad():
                features = self._encode(image, context, training)
        else:
            features = self._encode(image, context, training)
        if context is not None:
            features = self.conv1x1(features)
            features = self.film_layer(features, context)

        # Global average pool.
        if self._pooling:
            features = nn.functional.avg_pool2d(
                features, (features.shape[2], features.shape[3])
            )

        return features
