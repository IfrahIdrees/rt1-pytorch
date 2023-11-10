import torch
from torch import nn

EMBEDDING_DIM = 512


class FilmConditioning(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self._projection_add = nn.Linear(EMBEDDING_DIM, num_channels)
        self._projection_mult = nn.Linear(EMBEDDING_DIM, num_channels)
        self.num_channels = num_channels
        # From the paper
        nn.init.zeros_(self._projection_add.weight)
        nn.init.zeros_(self._projection_mult.weight)
        nn.init.zeros_(self._projection_add.bias)
        nn.init.zeros_(self._projection_mult.bias)

    def forward(
        self, conv_filters: torch.Tensor, conditioning: torch.Tensor
    ) -> torch.Tensor:
        assert len(conditioning.shape) == 2
        projected_cond_add = self._projection_add(conditioning)
        projected_cond_mult = self._projection_mult(conditioning)

        if len(conv_filters.shape) == 4:
            projected_cond_add = projected_cond_add.unsqueeze(2).unsqueeze(3)
            projected_cond_mult = projected_cond_mult.unsqueeze(2).unsqueeze(3)
        else:
            assert len(conv_filters.shape) == 2

        # Original FiLM paper argues that 1 + gamma centers the initialization at
        # identity transform.
        result = (1 + projected_cond_mult) * conv_filters + projected_cond_add
        return result
