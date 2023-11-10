"""Tests for film_conditioning_layer."""
import torch
from absl.testing import absltest, parameterized
from robotic_transformer_pytorch.film_efficientnet import film_conditioning_layer


class FilmConditioningLayerTest(parameterized.TestCase):
    @parameterized.parameters([2, 4])
    def test_film_conditioning_rank_two_and_four(self, conv_rank):
        batch = 2
        num_channels = 3
        if conv_rank == 2:
            conv_layer = torch.randn(batch, num_channels)
        elif conv_rank == 4:
            conv_layer = torch.randn(batch, 1, 1, num_channels)
        else:
            raise ValueError(f"Unexpected conv rank: {conv_rank}")
        context = torch.rand(batch, num_channels)
        film_layer = film_conditioning_layer.FilmConditioning(num_channels)
        out = film_layer(conv_layer, context)
        assert len(out.shape) == conv_rank


if __name__ == "__main__":
    absltest.main()
