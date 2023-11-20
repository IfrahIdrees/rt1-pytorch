from typing import Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import torch
from einops import rearrange

from robotic_transformer_pytorch.rt1_model import RT1Model
from robotic_transformer_pytorch.tokenizers.action_tokenizer import RT1ActionTokenizer


class RT1Policy:
    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        action_space: gym.spaces.Dict,
        action_bins=256,
        num_layers=8,
        num_heads=8,
        feed_forward_size=256,
        dropout_rate=0.1,
        time_sequence_length=6,
        embedding_dim=512,
        use_token_learner=True,
        token_learner_bottleneck_dim=64,
        token_learner_num_output_tokens=8,
    ):
        """
        Initializes an instance of the class.

        Args:
            observation_space (gym.spaces.Dict): The observation space of the environment.
            action_space (gym.spaces.Dict): The action space of the environment.
            action_bins (int, optional): The number of bins for discretizing continuous action spaces. Defaults to 256.
            num_layers (int, optional): The number of transformer layers in the model. Defaults to 8.
            num_heads (int, optional): The number of attention heads in each transformer layer. Defaults to 8.
            feed_forward_size (int, optional): The size of the feed-forward layer in the transformer. Defaults to 256.
            dropout_rate (float, optional): The dropout rate for the transformer layers. Defaults to 0.1.
            time_sequence_length (int, optional): The length of the time sequence for the model. Defaults to 6.
            embedding_dim (int, optional): The dimensionality of the input embeddings. Defaults to 512.
            use_token_learner (bool, optional): Whether to use the token learner module. Defaults to True.
            token_learner_bottleneck_dim (int, optional): The dimensionality of the bottleneck layer in the token learner. Defaults to 64.
            token_learner_num_output_tokens (int, optional): The number of output tokens from the token learner. Defaults to 8.

        Returns:
            None
        """
        self.observation_space = observation_space
        self.action_space = action_space
        self.action_bins = action_bins
        self.action_tokenizer = RT1ActionTokenizer(
            action_space=action_space,
            action_bins=action_bins,
            action_order=list(action_space.keys()),
        )

        self.model = RT1Model(
            tokens_per_action=self.action_tokenizer.tokens_per_action,
            action_bins=action_bins,
            num_layers=num_layers,
            num_heads=num_heads,
            feed_forward_size=feed_forward_size,
            dropout_rate=dropout_rate,
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
        texts: Union[np.ndarray, List[np.ndarray]],
        actions: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Preprocesses the given videos, texts, and actions.

        Args:
            videos (Union[np.ndarray, List[np.ndarray]]): The input videos to preprocess.
              shape: (b, t, c, h, w) or (b, t, h, w, c)
            texts (Union[np.ndarray, List[np.ndarray]]): The input texts to preprocess
              shape: (b, d)
            actions (Optional[Dict]): The input actions to preprocess. Defaults to None.
              shape: (b, t, a)

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing the preprocessed videos, texts, and actions.
        """
        if not isinstance(videos, np.ndarray):
            videos = np.stack(videos, axis=0)
        videos = torch.tensor(videos)

        if not isinstance(texts, np.ndarray):
            texts = np.stack(texts, axis=0)
        texts = torch.tensor(texts)

        if actions is not None:
            actions = self.action_tokenizer.tokenize(actions)
            actions = torch.tensor(actions)

        return videos, texts, actions

    def forward(
        self,
        videos: torch.Tensor,
        texts: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            videos (torch.Tensor): Input videos.
            texts (torch.Tensor): input contexts.
            actions (Optional[torch.Tensor]): Optional input actions.

        Returns:
            action_logits (Tuple[torch.Tensor, torch.Tensor]):
              A tuple containing the sampled actions and the action logits.
        """
        action_logits = self.model(videos, texts, actions)
        actions = torch.distributions.Categorical(logits=action_logits)
        actions = actions.sample()
        return actions, action_logits

    def loss(
        self,
        videos: Union[np.ndarray, List[np.ndarray]],
        texts: Union[np.ndarray, List[np.ndarray]],
        actions: Dict,
        target_actions: Dict,
    ) -> torch.Tensor:
        """
        Calculates the loss function for the given inputs.

        Parameters:
            videos (Union[np.ndarray, List[np.ndarray]]): The input videos. It can be a single numpy array or a list of numpy arrays.
            texts (Union[np.ndarray, List[np.ndarray]]): The input texts. It can be a single numpy array or a list of numpy arrays.
            actions (Dict): A dictionary containing the actions.
            target_actions (Dict): A dictionary containing the target actions.

        Returns:
            torch.Tensor: The calculated loss value.

        Raises:
            None
        """
        videos, texts, actions = self.preprocess(
            videos,
            texts,
            actions,
        )
        action_logits = self.forward(videos, texts, actions)

        target_actions = rearrange(target_actions, "b t a -> (b t) a")
        target_actions = self.action_tokenizer.tokenize(target_actions)
        target_actions = rearrange(
            target_actions, "(b t) ... -> b t ...", b=videos.shape[0]
        )
        target_actions = torch.tensor(target_actions)

        loss = torch.nn.functional.cross_entropy(action_logits, target_actions)
        return loss

    def act(self, observations: Dict) -> Dict[str, np.ndarray]:
        """
        Performs an action based on the given observations.

        Args:
            observations (Dict): A dictionary containing the observations. It should have the following keys:
                - "image" (np.ndarray): The video observations.
                - "context" (np.ndarray): The context.

        Returns:
            Dict[str, np.ndarray]: A dictionary containing the actions. It has the following keys:
                - "actions" (np.ndarray): The actions performed based on the observations.
        """
        videos = observations["image"]
        texts = observations["context"]
        videos, texts, _ = self.preprocess(videos, texts)
        actions, _ = self.forward(videos, texts)
        actions = actions.detach().cpu().numpy()
        actions = self.action_tokenizer.detokenize(actions)
        return actions
