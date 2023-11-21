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
        env_id="halfcheetah-expert-v2",
        context="run forwards",
        context_dim=512,
    ):
        self.env = gym.make("GymV21Environment-v0", env_id=env_id)
        self.action_space = gym.spaces.Dict({"actions": self.env.action_space})
        self.context = context
        self.observation_space = gym.spaces.Dict(
            {
                "observations": self.env.observation_space,
                "context": gym.spaces.Box(
                    low=-1.0, high=1.0, shape=(context_dim,), dtype=np.float32
                ),
            }
        )


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

    print("Loading environment...")
    env = RT1Env(
        env_id=args.dataset,
        context=args.context,
    )

    print("Loading dataset...")
    train_dataset = create_dataset(
        datasets=args.datasets,
        split=args.train_split,
        trajectory_length=args.trajectory_length,
        batch_size=args.train_batch_size,
    )

    print("Building policy...")
    policy = RT1Policy(
        observation_space=observation_space,
        action_space=action_space,
        device=args.device,
        checkpoint_path=args.load_checkpoint,
    )
    policy.model.train()
    optimizer = Adam(policy.model.parameters(), lr=args.lr)
    text_embedding_model = (
        SentenceTransformer(args.sentence_transformer)
        if args.sentence_transformer
        else None
    )
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
        if text_embedding_model is not None:
            return text_embedding_model.encode(observation["instruction"])
        else:
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
                policy.model.eval()
                batch = next(eval_dataset)
                observations = {
                    "image": batch["observation"]["image"],
                    "context": get_text_embedding(batch["observation"]),
                }
                actions = batch["action"]
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
