"""A FiLM Efficientnet contextual image tokenizer used in Robotics Transformer 1.
"""
from typing import Optional

import torch
from torch import nn

from robotic_transformer_pytorch.film_efficientnet import FilmEfficientNetEncoder
from robotic_transformer_pytorch.tokenizers import token_learner


class RT1ImageTokenizer(nn.Module):
    """Tokenizes based on vocab size."""

    def __init__(
        self,
        embedding_output_dim: int,
        use_token_learner: bool = True,
        num_tokens: int = 8,
    ):
        """Instantiates a RT1ImageTokenizer.

        Args:
          embedding_output_dim: The output size of the tokens.
          use_token_learner: Whether to use token learner. See
            https://arxiv.org/abs/2106.11297
          num_tokens: Relevant only for token learner - the number of learned
            tokens.
        """
        super().__init__()

        self._tokenizer = FilmEfficientNetEncoder()

        self._use_token_learner = use_token_learner
        if self._use_token_learner:
            self._num_tokens = num_tokens
            self._token_learner = token_learner.TokenLearnerModule(
                num_tokens=self._num_tokens,
                embedding_dim=embedding_output_dim,
            )

    @property
    def tokens_per_context_image(self) -> int:
        if self._use_token_learner:
            num_tokens = self._num_tokens
        else:
            num_tokens = 100
        return num_tokens

    def __call__(
        self,
        image: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Gets image tokens.

        Args:
          image: Images of shape (b, t, h, w, 3) to tokenize.
          context: An optional context vector (e.g., a natural language embedding).
            Expected to have shape (b, t, embedding_dim).

        Returns:
          tokens: has shape (batch, t, num_tokens_per_timestep, embedding_dim)
        """
        b, t, h, w, c = image.shape

        # Fold the time axis into the batch axis.
        image = torch.reshape(image, [b * t, h, w, c])
        if context is not None:
            assert len(context.shape) == 3
            context = torch.reshape(context, [b * t, -1])
        tokens = self.get_image_embeddings(image, context)
        if self._use_token_learner:
            tokens = self._token_learner(tokens)
        # Unflatten the time axis, which was previously flattened into the batch.
        tokens = torch.reshape(tokens, [b, t, self.tokens_per_context_image, -1])
        return tokens

    def get_image_embeddings(
        self, image: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Gets embeddings from image.

        Args:
          image: Expected to be float32 in range [0, 1] with shape (b, h, w, 3).
          context: Expected to be float32 with shape (b, embedding_dim)

        Returns:
          tokens of shape (b, num_tokens, emedding_dim)
        """
        image_tokens = self._tokenizer(image, context=context)
        image_tokens = torch.reshape(image_tokens, [-1, 100, 512])
        return image_tokens
