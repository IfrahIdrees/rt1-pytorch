from typing import List, Optional, Union

import numpy as np
import torch
from einops import pack, rearrange, reduce, repeat, unpack
from einops.layers.torch import Rearrange
from PIL import Image
from torch import nn

from robotic_transformer_pytorch.film_efficientnet import FilmEfficientNetEncoder


# helpers
def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def cast_tuple(val, length=1):
    return val if isinstance(val, tuple) else ((val,) * length)


def pack_one(x, pattern):
    return pack([x], pattern)


def unpack_one(x, ps, pattern):
    return unpack(x, ps, pattern)[0]


# sinusoidal positions
def posemb_sincos_1d(seq, dim, temperature=10000, device=None, dtype=torch.float32):
    n = torch.arange(seq, device=device)
    omega = torch.arange(dim // 2, device=device) / (dim // 2 - 1)
    omega = 1.0 / (temperature**omega)

    n = n[:, None] * omega[None, :]
    pos_emb = torch.cat((n.sin(), n.cos()), dim=1)
    return pos_emb.type(dtype)


# token learner module
class TokenLearner(nn.Module):
    """
    https://arxiv.org/abs/2106.11297
    using the 1.1 version with the MLP (2 dense layers with gelu) for generating attention map
    """

    def __init__(self, *, dim, ff_mult=2, num_output_tokens=8):
        super().__init__()
        inner_dim = dim * ff_mult * num_output_tokens

        self.num_output_tokens = num_output_tokens
        self.net = nn.Sequential(
            nn.Conv2d(dim * num_output_tokens, inner_dim, 1, groups=num_output_tokens),
            nn.GELU(),
            nn.Conv2d(inner_dim, num_output_tokens, 1, groups=num_output_tokens),
        )

    def forward(self, x: torch.Tensor):
        x, ps = pack_one(x, "* c h w")
        x = repeat(x, "b c h w -> b (g c) h w", g=self.num_output_tokens)
        attn = self.net(x)

        attn = rearrange(attn, "b g h w -> b 1 g h w")
        x = rearrange(x, "b (g c) h w -> b c g h w", g=self.num_output_tokens)

        x = reduce(x * attn, "b c g h w -> b c g", "mean")
        x = unpack_one(x, ps, "* c n")
        return x


# Robotic Transformer
class RT1(nn.Module):
    def __init__(
        self,
        *,
        efficientnet_variant: str = "b3",
        encoder_dim=100,
        num_actions=11,
        action_bins=256,
        num_layers=8,
        layer_size=128,
        num_heads=8,
        feed_forward_size=512,
        dropout_rate=0.1,
        vocab_size=256,
        token_embedding_size=512,
        time_sequence_length=6,
        use_token_learner=True,
        token_learner_ff_mult=2,
        token_learner_num_layers=2,
        token_learner_num_output_tokens=8,
    ):
        super().__init__()
        self.encoder = FilmEfficientNetEncoder(model_variant=efficientnet_variant)

        self.token_learner = TokenLearner(
            dim=encoder_dim,
            ff_mult=token_learner_ff_mult,
            num_output_tokens=token_learner_num_output_tokens,
            num_layers=token_learner_num_layers,
        )

        self.num_learned_tokens = token_learner_num_output_tokens

        self.transformer = nn.Transformer(
            d_model=layer_size,
            nhead=num_heads,
            num_encoder_layers=0,
            num_decoder_layers=num_layers,
            dim_feedforward=feed_forward_size,
            dropout=dropout_rate,
            activation="gelu",
            batch_first=True,
        )

        self.to_logits = nn.Sequential(
            nn.LayerNorm(token_embedding_size),
            nn.Linear(token_embedding_size, num_actions * action_bins),
            Rearrange("... (a b) -> ... a b", b=action_bins),
        )

    def forward(
        self,
        videos: Union[np.ndarray, List[np.ndarray]],
        texts: Optional[List[str]] = None,
    ):
        if not isinstance(videos, np.ndarray):
            videos = np.stack(videos)

        frames = videos.shape[2]
        videos = rearrange(videos, "b c f h w -> b f c h w")
        images, packed_shape = pack_one(videos, "* c h w")

        tokens = self.encoder(images, texts)

        tokens = unpack_one(tokens, packed_shape, "* c h w")
        learned_tokens = self.token_learner(tokens)

        learned_tokens = rearrange(learned_tokens, "b f c n -> b (f n) c")

        # attention
        attended_tokens = self.transformer(learned_tokens)

        pooled = reduce(attended_tokens, "b (f n) d -> b f d", "mean", f=frames)

        logits = self.to_logits(pooled)
        return logits
