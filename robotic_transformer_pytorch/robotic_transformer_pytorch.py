from typing import List, Optional

import torch
from einops import pack, rearrange, reduce, repeat, unpack
from einops.layers.torch import Rearrange
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
        num_actions=11,
        action_bins=256,
        depth=6,
        heads=8,
        dim_head=64,
        token_learner_ff_mult=2,
        token_learner_num_layers=2,
        token_learner_num_output_tokens=8,
        cond_drop_prob=0.2,
    ):
        super().__init__()
        self.encoder = FilmEfficientNetEncoder(model_variant=efficientnet_variant)

        self.token_learner = TokenLearner(
            dim=100,
            ff_mult=token_learner_ff_mult,
            num_output_tokens=token_learner_num_output_tokens,
            num_layers=token_learner_num_layers,
        )

        self.num_learned_tokens = token_learner_num_output_tokens

        self.transformer_depth = depth

        self.transformer = nn.Transformer(
            d_model=512,
            nhead=heads,
            num_encoder_layers=0,
            num_decoder_layers=depth,
            dim_feedforward=dim_head,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
        )

        self.cond_drop_prob = cond_drop_prob

        self.to_logits = nn.Sequential(
            LayerNorm(vit.embed_dim),
            nn.Linear(vit.embed_dim, num_actions * action_bins),
            Rearrange("... (a b) -> ... a b", b=action_bins),
        )

    def forward(self, video, texts: Optional[List[str]] = None, cond_drop_prob=0.0):
        depth = self.transformer_depth
        cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)

        frames, device = video.shape[2], video.device

        cond_fns = self.conditioner(
            texts,
            cond_drop_prob=cond_drop_prob,
            repeat_batch=(
                *((frames,) * self.num_vit_stages),
                *((1,) * self.transformer_depth * 2),
            ),
        )

        vit_cond_fns, transformer_cond_fns = (
            cond_fns[: -(depth * 2)],
            cond_fns[-(depth * 2) :],
        )

        video = rearrange(video, "b c f h w -> b f c h w")
        images, packed_shape = pack_one(video, "* c h w")

        tokens = self.vit(
            images,
            texts=texts,
            cond_fns=vit_cond_fns,
            cond_drop_prob=cond_drop_prob,
            return_embeddings=True,
        )

        tokens = unpack_one(tokens, packed_shape, "* c h w")
        learned_tokens = self.token_learner(tokens)

        learned_tokens = rearrange(learned_tokens, "b f c n -> b (f n) c")

        # causal attention mask

        attn_mask = torch.ones((frames, frames), dtype=torch.bool, device=device).triu(
            1
        )
        attn_mask = repeat(
            attn_mask,
            "i j -> (i r1) (j r2)",
            r1=self.num_learned_tokens,
            r2=self.num_learned_tokens,
        )

        # sinusoidal positional embedding

        pos_emb = posemb_sincos_1d(
            frames,
            learned_tokens.shape[-1],
            dtype=learned_tokens.dtype,
            device=learned_tokens.device,
        )

        learned_tokens = learned_tokens + repeat(
            pos_emb, "n d -> (n r) d", r=self.num_learned_tokens
        )

        # attention

        attended_tokens = self.transformer(
            learned_tokens, cond_fns=transformer_cond_fns, attn_mask=~attn_mask
        )

        pooled = reduce(attended_tokens, "b (f n) d -> b f d", "mean", f=frames)

        logits = self.to_logits(pooled)
        return logits
