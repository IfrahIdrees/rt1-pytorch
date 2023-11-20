from typing import Optional

import torch
from einops import rearrange, reduce
from torch import nn

from robotic_transformer_pytorch.tokenizers.image_tokenizer import RT1ImageTokenizer


def posemb_sincos_1d(seq, dim, temperature=10000, device=None, dtype=torch.float32):
    """
    Generate positional embeddings using sine and cosine functions for a 1-dimensional sequence.

    Parameters:
        seq (int): The length of the sequence.
        dim (int): The dimension of the positional embeddings.
        temperature (float, optional): The temperature parameter for the sine function. Defaults to 10000.
        device (torch.device, optional): The device for tensor operations. Defaults to None.
        dtype (torch.dtype, optional): The data type of the positional embeddings. Defaults to torch.float32.

    Returns:
        torch.Tensor: The positional embeddings of shape (seq, dim), with each element computed as the concatenation of the sine and cosine values.

    """
    n = torch.arange(seq, device=device)
    omega = torch.arange(dim // 2, device=device) / (dim // 2 - 1)
    omega = 1.0 / (temperature**omega)

    n = n[:, None] * omega[None, :]
    pos_emb = torch.cat((n.sin(), n.cos()), dim=1)
    return pos_emb.type(dtype)


# Robotic Transformer
class RT1Model(nn.Module):
    def __init__(
        self,
        tokens_per_action=11,
        action_bins=256,
        vocab_size=256,
        num_layers=6,
        num_heads=8,
        feed_forward_size=512,
        dropout_rate=0.1,
        time_sequence_length=6,
        embedding_dim=512,
        use_token_learner=True,
        token_learner_bottleneck_dim=64,
        token_learner_num_output_tokens=8,
    ):
        """
        Initializes the RT1Model.

        Parameters:
            tokens_per_action (int): The number of tokens per action. Default is 11.
            action_bins (int): The number of action bins. Default is 256.
            vocab_size (int): The size of the vocabulary. Default is 256.
            num_layers (int): The number of transformer layers. Default is 6.
            num_heads (int): The number of attention heads. Default is 8.
            feed_forward_size (int): The size of the feed-forward layer. Default is 512.
            dropout_rate (float): The dropout rate. Default is 0.1.
            time_sequence_length (int): The length of the time sequence. Default is 6.
            embedding_dim (int): The dimension of the embedding. Default is 512.
            use_token_learner (bool): Whether to use token learner. Default is True.
            token_learner_bottleneck_dim (int): The dimension of the token learner bottleneck. Default is 64.
            token_learner_num_output_tokens (int): The number of output tokens of the token learner. Default is 8.

        Returns:
            None
        """
        super().__init__()
        self.time_sequence_length = time_sequence_length
        self.action_encoder = nn.Linear(vocab_size, embedding_dim)
        self.image_tokenizer = RT1ImageTokenizer(
            embedding_dim=embedding_dim,
            use_token_learner=use_token_learner,
            token_learner_bottleneck_dim=token_learner_bottleneck_dim,
            token_learner_num_output_tokens=token_learner_num_output_tokens,
            dropout_rate=dropout_rate,
        )

        self.num_tokens = self.image_tokenizer._num_tokens

        self.transformer = nn.Transformer(
            d_model=embedding_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=feed_forward_size,
            dropout=dropout_rate,
            activation="gelu",
            batch_first=True,
        )

        self.to_logits = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, tokens_per_action * action_bins),
        )
        self.tokens_per_action = tokens_per_action
        self.action_bins = action_bins
        self.embedding_dim = embedding_dim

    def forward(
        self,
        videos: torch.Tensor,
        texts: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass of the model.

        Args:
            videos (torch.Tensor): The input videos. Shape is (b, t, h, w, c) or (b, t, c, h, w).
            texts (Optional[torch.Tensor]): The input texts. Shape is (b, t, d).
            actions (Optional[torch.Tensor]): The input actions. Shape is (b, t, a).

        Returns:
            torch.Tensor: The output logits. Shape is (b, t, a, d).
        """
        b, t, *_ = videos.shape
        assert (
            t >= self.time_sequence_length
        ), f"Please provide at least {self.time_sequence_length} frames"

        # pack time dimension into batch dimension
        videos = rearrange(videos, "b t h w c -> (b t) h w c")
        if texts is not None:
            texts = rearrange(texts, "b t d -> (b t) d")

        # tokenize images and texts
        tokens = self.image_tokenizer(videos, texts)

        # unpack time dimension
        tokens = rearrange(tokens, "(b t) c n -> b t c n", b=b, t=t)

        # repeat tokens by rolling self.time_sequence_length times over t
        indices = torch.stack(
            [
                torch.arange(s, s + t - self.time_sequence_length + 1)
                for s in range(self.time_sequence_length)
            ],
            dim=-1,
        )
        tokens = tokens[:, indices, :, :]  # (b, t-5, 6, c, n)

        # pack time dimension into batch dimension
        tokens = rearrange(tokens, "b t f c n -> (b t) (f n) c")

        # sinusoidal positional embedding
        pos_emb = posemb_sincos_1d(tokens.shape[1], tokens.shape[2])
        tokens = tokens + pos_emb

        # causal mask for tokens
        token_mask = torch.ones(
            tokens.shape[1], tokens.shape[1], dtype=torch.bool
        ).tril(0)

        # encode actions to have the same embedding dimension as tokens
        if actions is None:
            actions = torch.zeros((b, t, self.tokens_per_action, self.action_bins))
        action_tokens = self.action_encoder(actions)

        # index previous action
        action_tokens = action_tokens[:, indices, :, :]

        # pack time dimension into batch dimension
        action_tokens = rearrange(action_tokens, "b t f n a -> (b t) (f n) a")

        pos_emb = posemb_sincos_1d(action_tokens.shape[1], action_tokens.shape[2])
        action_tokens = action_tokens + pos_emb

        # action mask: do not let actions attend to previous actions,
        # a_t is independent of a_{t-1} given pi and s_t
        action_mask = torch.ones(
            self.time_sequence_length, self.time_sequence_length, dtype=torch.bool
        ).tril(0)
        action_mask = torch.kron(
            torch.eye(self.tokens_per_action, self.tokens_per_action, dtype=torch.bool),
            action_mask,
        )

        # causal mask between tokens and actions;
        # a_t attends to s_t' for all t'<=t
        memory_mask = torch.ones(
            self.time_sequence_length, self.time_sequence_length, dtype=torch.bool
        ).tril(0)
        memory_mask = torch.kron(
            memory_mask,
            torch.ones(self.tokens_per_action, self.num_tokens, dtype=torch.bool),
        )

        attended_tokens = self.transformer(
            src=tokens,
            src_mask=token_mask,
            tgt=action_tokens,
            tgt_mask=action_mask,
            memory_mask=memory_mask,
        )

        pooled = reduce(attended_tokens, "b n d -> b d", "mean")

        logits = self.to_logits(pooled)
        logits = rearrange(
            logits,
            "(b t) (n d) -> b t n d",
            b=b,
            t=t - self.time_sequence_length + 1,
            n=self.tokens_per_action,
            d=self.action_bins,
        )

        # Add zero actions for first self.time_sequence_length timesteps
        logits = torch.cat(
            [
                torch.zeros(
                    (
                        b,
                        self.time_sequence_length - 1,
                        self.tokens_per_action,
                        self.action_bins,
                    )
                ),
                logits,
            ],
            dim=1,
        )

        return logits