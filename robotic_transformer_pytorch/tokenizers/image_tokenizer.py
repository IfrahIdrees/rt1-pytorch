"""A FiLM Efficientnet contextual image tokenizer used in Robotics Transformer 1.
"""
from typing import Optional

import torch
from torch import nn

from robotic_transformer_pytorch.film_efficientnet.pretrained_efficientnet_encoder import (
    FilmEfficientNetEncoder,
)
from robotic_transformer_pytorch.tokenizers.token_learner import TokenLearner


class RT1ImageTokenizer(nn.Module):
    """Tokenizes based on vocab size."""

    def __init__(
        self,
        embedding_dim: int,
        use_token_learner=True,
        token_learner_bottleneck_dim=64,
        token_learner_num_output_tokens=8,
        dropout_rate=0.1,
    ):
        """Instantiates a RT1ImageTokenizer.

        Args:
          embedding_dim: The embedding size of the tokens.
          use_token_learner: Whether to use token learner. See
            https://arxiv.org/abs/2106.11297
          num_tokens: Relevant only for token learner - the number of learned
            tokens.
        """
        super().__init__()

        self._tokenizer = FilmEfficientNetEncoder()

        self._use_token_learner = use_token_learner
        if self._use_token_learner:
            self._num_tokens = token_learner_num_output_tokens
            self._token_learner = TokenLearner(
                embedding_dim=embedding_dim,
                num_tokens=self._num_tokens,
                bottleneck_dim=token_learner_bottleneck_dim,
                dropout_rate=dropout_rate,
            )

    @property
    def tokens_per_context_image(self) -> int:
        if self._use_token_learner:
            num_tokens = self._num_tokens
        else:
            num_tokens = 100
        return num_tokens

    def forward(
        self,
        image: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Gets image tokens.

        Args:
          image: Images of shape (b, h, w, 3) to tokenize.
          context: An optional context vector (e.g., a natural language embedding).
            Expected to have shape (b, embedding_dim).

        Returns:
          tokens: has shape (batch, num_tokens_per_timestep, embedding_dim)
        """

        if context is not None:
            assert len(context.shape) == 2, f"Unexpected context shape: {context.shape}"
        tokens = self._tokenizer(image, context)
        if self._use_token_learner:
            tokens = self._token_learner(tokens)
        elif len(tokens.shape) == 4:
            # (b, c, h, w) -> (b, c, h*w)
            tokens = tokens.reshape(tokens.shape[0], tokens.shape[1], -1)
        return tokens
