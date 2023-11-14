from typing import Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from robotic_transformer_pytorch.rt1_model import RT1Model
from robotic_transformer_pytorch.tokenizers.action_tokenizer import RT1ActionTokenizer


class RT1Policy:
    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        action_space: gym.spaces.Dict,
        text_encoder: str = "all-MiniLM-L6-v2",
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
        lr=1e-4,
    ):
        self.observation_space = observation_space
        self.action_space = action_space
        self.action_bins = action_bins
        self.text_encoder = SentenceTransformer(text_encoder).eval()
        self.action_tokenizer = RT1ActionTokenizer(
            action_space=action_space,
            vocab_size=vocab_size,
            action_order=list(action_space.keys()),
        )

        self.model = RT1Model(
            tokens_per_action=self.action_tokenizer.tokens_per_action,
            action_bins=action_bins,
            num_layers=num_layers,
            num_heads=num_heads,
            feed_forward_size=feed_forward_size,
            dropout_rate=dropout_rate,
            vocab_size=vocab_size,
            time_sequence_length=time_sequence_length,
            embedding_dim=embedding_dim,
            use_token_learner=use_token_learner,
            token_learner_bottleneck_dim=token_learner_bottleneck_dim,
            token_learner_num_output_tokens=token_learner_num_output_tokens,
        )

        self.embedding_dim = embedding_dim

    def preprocess(
        self,
        videos: Union[np.ndarray, List[np.ndarray]],
        texts: Optional[List[str]] = None,
        prev_actions: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not isinstance(videos, np.ndarray):
            videos = np.stack(videos, axis=0)

        b, t, *_ = videos.shape
        videos = torch.tensor(videos)

        if texts is not None:
            with torch.no_grad():
                texts = self.text_encoder.encode(texts)
            assert (
                texts.shape[0] == b and texts.shape[1] == self.embedding_dim
            ), f"Incorrect text shape {texts.shape}"
            texts = texts.reshape(b, 1, self.embedding_dim)
            texts = np.repeat(texts, t, axis=1)
            texts = torch.tensor(texts)
        if prev_actions is not None:
            prev_actions = self.action_tokenizer.tokenize(prev_actions)
            prev_actions = torch.tensor(prev_actions)
        return videos, texts, prev_actions

    def forward(
        self,
        videos: torch.Tensor,
        texts: Optional[torch.Tensor] = None,
        prev_actions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        action_logits = self.model(videos, texts, prev_actions)
        actions = torch.distributions.Categorical(logits=action_logits)
        actions = actions.sample()
        return actions, action_logits

    def loss(
        self,
        videos: np.ndarray,
        texts: List[str],
        prev_actions: Dict,
        target_actions: Dict,
    ) -> torch.Tensor:
        videos, texts, prev_actions = self.preprocess(
            videos,
            texts,
            prev_actions,
        )
        action_logits = self.forward(videos, texts, prev_actions)

        target_actions = self.action_tokenizer.tokenize(target_actions)
        target_actions = torch.tensor(target_actions)

        loss = torch.nn.functional.cross_entropy(action_logits, target_actions)
        return loss

    def act(
        self,
        videos: np.ndarray,
        texts: Optional[List[str]] = None,
        prev_actions: Optional[Dict] = None,
    ) -> Dict[str, np.ndarray]:
        videos, texts, prev_actions = self.preprocess(videos, texts, prev_actions)
        actions, _ = self.forward(videos, texts, prev_actions)
        actions = actions.detach().cpu().numpy()
        actions = self.action_tokenizer.detokenize(actions)
        return actions

    def parameters(self):
        return self.model.parameters()

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()
