from typing import List, Optional, Union

import gymnasium as gym
import numpy as np
import torch
from einops import rearrange, reduce
from einops.layers.torch import Rearrange
from sentence_transformers import SentenceTransformer
from torch import nn

from robotic_transformer_pytorch.tokenizers.action_tokenizer import RT1ActionTokenizer
from robotic_transformer_pytorch.tokenizers.image_tokenizer import RT1ImageTokenizer


# Robotic Transformer
class RT1(nn.Module):
    def __init__(
        self,
        *,
        action_space: gym.spaces.Dict,
        text_encoder: str = "all-MiniLM-L6-v2",
        num_actions=11,
        action_bins=256,
        num_layers=8,
        num_heads=8,
        feed_forward_size=256,
        dropout_rate=0.1,
        vocab_size=256,
        time_sequence_length=6,
        embedding_dim=384,
        use_token_learner=True,
        token_learner_bottleneck_dim=64,
        token_learner_num_output_tokens=8,
    ):
        super().__init__()
        self.time_sequence_length = time_sequence_length
        self.text_encoder = SentenceTransformer(text_encoder)
        self.image_tokenizer = RT1ImageTokenizer(
            embedding_dim=embedding_dim,
            use_token_learner=use_token_learner,
            token_learner_bottleneck_dim=token_learner_bottleneck_dim,
            token_learner_num_output_tokens=token_learner_num_output_tokens,
            dropout_rate=dropout_rate,
        )
        self.action_tokenizer = RT1ActionTokenizer(
            action_space=action_space,
            vocab_size=vocab_size,
        )

        self.num_learned_tokens = token_learner_num_output_tokens

        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=embedding_dim,
                nhead=num_heads,
                dim_feedforward=feed_forward_size,
                dropout=dropout_rate,
                activation="gelu",
                batch_first=True,
            ),
            num_layers=num_layers,
        )

        self.to_logits = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, num_actions * action_bins),
            Rearrange("... (a b) -> ... a b", b=action_bins),
        )

    def forward(
        self,
        videos: Union[np.ndarray, List[np.ndarray]],
        texts: Optional[List[str]] = None,
    ):
        if not isinstance(videos, np.ndarray):
            videos = np.stack(videos)

        b, f, h, w, c = videos.shape
        print("b f h w c: ", b, f, h, w, c)
        videos = torch.tensor(videos).reshape(b * f, h, w, c)

        if texts is not None:
            texts = self.text_encoder.encode(texts)
            assert len(texts) == b
            texts = np.repeat(texts, f, axis=0)
            texts = torch.tensor(texts)
        tokens = self.image_tokenizer(videos, texts)
        tokens = torch.tile(tokens, (1, self.time_sequence_length, 1, 1))

        tokens = rearrange(tokens, "b f c n -> b (f n) c")

        attended_tokens = self.transformer(tokens, torch.zeros_like(tokens))

        pooled = reduce(attended_tokens, "b n d -> b d", "mean")
        pooled = pooled.reshape(b, f, -1)

        logits = self.to_logits(pooled)
        return logits
