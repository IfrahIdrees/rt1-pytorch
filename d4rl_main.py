import argparse
import os
from typing import Dict

import d4rl
import gymnasium as gym
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from torch.optim import Adam

from rt1_pytorch.rt1_policy import RT1Policy


class RT1Env(gym.Env):
    def __init__(
        self,
        env_id: str,
        embedding: np.ndarray,
        embedding_dim=384,
        num_frames=6,
    ):
        self.env = gym.wrappers.FrameStack(
            gym.make("GymV21Environment-v0", env_id=env_id), num_frames
        )
        self.action_space = gym.spaces.Dict({"actions": self.env.action_space})
        self.embedding = embedding
        self.observation_space = gym.spaces.Dict(
            {
                "observations": self.env.observation_space,
                "embedding": gym.spaces.Box(
                    low=-1.0, high=1.0, shape=(embedding_dim,), dtype=np.float32
                ),
            }
        )

    def reset(self):
        obs, info = self.env.reset()
        return ({"image": obs, "embedding": self.embedding}, info)

    def step(self, action):
        o, r, term, trunc, info = self.env.step(action)
        return ({"image": o, "embedding": self.embedding}, r, term, trunc, info)

    def get_dataset(self):
        dataset = self.env.get_dataset()
        breakpoint()
        return dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="halfcheetah-expert-v2",
    )
    parser.add_argument(
        "--context",
        type=str,
        default="run forward as smoothly as possible without falling",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="number of training epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="learning rate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="batch size",
    )
    parser.add_argument(
        "--trajectory-length",
        type=int,
        default=6,
        help="number of frames per trajectory",
    )
    parser.add_argument(
        "--sentence-transformer",
        type=str,
        default="all-MiniLM-L6-v2",
        help="SentenceTransformer to use for text embedding",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="device to use for training",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=None,
        help="eval frequency in number of batches; defaults to None",
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=None,
        help="checkpoint frequency in number of batches; defaults to None",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="d4rl_checkpoints",
        help="directory to save checkpoints",
    )
    parser.add_argument(
        "--load-checkpoint",
        type=str,
        default=None,
        help="checkpoint to load from; defaults to None",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    text_embedding_model = SentenceTransformer(args.sentence_transformer)
    embedding = text_embedding_model.encode(args.context)

    print("Loading environment...")
    env = RT1Env(
        env_id=args.dataset,
        embedding=embedding,
        embedding_dim=text_embedding_model.get_sentence_embedding_dimension(),
        num_frames=args.trajectory_length,
    )

    print("Loading dataset...")
    train_dataset = env.get_dataset()

    print("Building policy...")
    policy = RT1Policy(
        observation_space=env.unwrapped.observation_space,
        action_space=env.action_space,
        device=args.device,
        checkpoint_path=args.load_checkpoint,
    )
    policy.model.train()
    optimizer = Adam(policy.model.parameters(), lr=args.lr)
    # Total number of params
    total_params = sum(p.numel() for p in policy.model.parameters())
    # Transformer params
    transformer_params = sum(p.numel() for p in policy.model.transformer.parameters())
    # FiLM-EfficientNet and TokenLearner params
    tokenizer_params = sum(p.numel() for p in policy.model.image_tokenizer.parameters())
    print(f"Total params: {total_params}")
    print(f"Transformer params: {transformer_params}")
    print(f"FiLM-EfficientNet+TokenLearner params: {tokenizer_params}")

    def get_text_embedding(observation: Dict):
        return observation["embedding"]

    print("Training...")
    for epoch in range(args.epochs):
        num_batches = 0
        for batch in train_dataset:
            policy.model.train()
            num_batches += 1
            observations = {
                "image": batch["observation"]["image"],
                "context": get_text_embedding(batch["observation"]),
            }
            actions = batch["action"]
            loss = policy.loss(observations, actions)
            print(f"Training loss Epoch {epoch} Batch {num_batches}: {loss.item()}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if args.eval_freq and num_batches % args.eval_freq == 0:
                print("Evaluating...")
                breakpoint()
                policy.model.eval()
                obs, _ = env.reset()
                observations = []
                actions = []
                done = False
                while not done:
                    obs = {
                        "image": obs["image"],
                        "context": get_text_embedding(obs),
                    }
                    act = policy.act(obs)
                    observations, _, _, _, _ = env.step(obs)
                    observations.append(obs)
                    actions.append(act)
                eval_loss = policy.loss(observations, actions)
                eval_loss = eval_loss.item()
                print(f"eval loss: {eval_loss}")
            if args.checkpoint_freq and num_batches % args.checkpoint_freq == 0:
                checkpoint_path = (
                    f"{args.checkpoint_dir}/checkpoint_{num_batches}"
                    + f"_loss_{loss.item():.3f}.pt"
                )
                torch.save(policy.model.state_dict(), checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
