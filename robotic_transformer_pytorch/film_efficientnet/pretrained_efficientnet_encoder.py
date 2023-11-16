"""Encoder based on Efficientnet."""

from typing import Any, Optional

import torch
from torch import nn
from torchvision.models.efficientnet import EfficientNet_B3_Weights

from robotic_transformer_pytorch.film_efficientnet.film_conditioning_layer import (
    FilmConditioning,
)
from robotic_transformer_pytorch.film_efficientnet.film_efficientnet_encoder import (
    filmefficientnet_b3,
)

_MODELS = {
    "b3": filmefficientnet_b3,
}


class FilmEfficientNetEncoder(nn.Module):
    """Applies a pretrained Efficientnet based encoder with FiLM layers."""

    def __init__(
        self,
        model_variant: str = "b3",
        weights: Optional[Any] = "DEFAULT",
        include_top: bool = False,
        embedding_dim: int = 512,
    ):
        """Initialize the model.

        Args:
          model_variant: One of 'b0-b7' of the efficient encoders. See
            https://arxiv.org/abs/1905.11946 to understand the variants.
          weights: : One of "DEFAULT" or "IMAGENET1K".
          include_top: Whether to add the top fully connected layer. If True, this
            will cause encoding to fail and is used only for unit testing purposes.
          embedding_dim: Dimension of the embedding space.
        """
        super().__init__()
        if model_variant not in _MODELS:
            raise ValueError(f"Unknown variant {model_variant}")
        self.net = _MODELS[model_variant](
            include_top=include_top,
            weights=weights,
            embedding_dim=embedding_dim,
        )
        self.include_top = include_top
        self.conv1x1 = nn.Conv2d(
            in_channels=1536,
            out_channels=embedding_dim,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding="same",
            bias=False,
        )
        nn.init.kaiming_normal_(self.conv1x1.weight)
        self.film_layer = FilmConditioning(embedding_dim, embedding_dim)
        self.embedding_dim = embedding_dim

    def forward(
        self, image: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.include_top:
            assert context is None, "Cannot use context with include_top=True"
        elif context is None:
            context = torch.zeros(image.shape[0], self.embedding_dim)
        if len(image.shape) == 3:
            # Add batch dimension
            image = image.unsqueeze(0)
        assert len(image.shape) == 4, f"Unexpected image shape: {image.shape}"
        if image.shape[-1] == 3:
            # (B, H, W, C) -> (B, C, H, W)
            image = image.permute(0, 3, 1, 2)
        if torch.max(image) >= 1.0:
            # Normalize to [0, 1]
            image = image / 255.0
        assert torch.min(image) >= 0.0 and torch.max(image) <= 1.0
        preprocess = EfficientNet_B3_Weights.DEFAULT.transforms(antialias=True)
        image = preprocess(image)

        features = self.net(image, context)
        if not self.include_top:
            features = self.conv1x1(features)
            features = self.film_layer(features, context)

        return features
