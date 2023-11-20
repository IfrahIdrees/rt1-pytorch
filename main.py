import argparse
import os
from typing import Dict

import gymnasium as gym
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from torch.optim import Adam

from data import create_dataset
from rt1_pytorch.rt1_policy import RT1Policy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        type=list,
        default=["fractal20220817_data"],
    )
    parser.add_argument(
        "--train-split",
        type=str,
        default="train[:-1000]",
        help="use e.g. train[:100] for the first 100 episodes",
    )
    parser.add_argument(
        "--eval-split",
        type=str,
        default="train[-1000:]",
        help="use e.g. eval[:100] for the first 100 episodes",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="number of training epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="learning rate",
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=32,
        help="train batch size",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=32,
        help="eval batch size",
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
        default=None,
        help="SentenceTransformer to use; default is None for original USE embeddings",
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
        default="checkpoints",
        help="directory to save checkpoints",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print("Loading dataset...")
    train_dataset = create_dataset(
        datasets=args.datasets,
        split=args.train_split,
        trajectory_length=args.trajectory_length,
        batch_size=args.train_batch_size,
    )
    eval_dataset = create_dataset(
        datasets=args.datasets,
        split=args.eval_split,
        trajectory_length=args.trajectory_length,
        batch_size=args.eval_batch_size,
    )

    observation_space = gym.spaces.Dict(
        image=gym.spaces.Box(low=0, high=255, shape=(128, 128, 3)),
        context=gym.spaces.Box(low=0.0, high=1.0, shape=(512,), dtype=np.float32),
    )
    action_space = gym.spaces.Dict(
        world_vector=gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
        base_displacement_vertical_rotation=gym.spaces.Box(
            low=-np.pi / 2.0, high=np.pi / 2.0, shape=(1,), dtype=np.float32
        ),
        gripper_closedness_action=gym.spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        ),
        terminate_episode=gym.spaces.Discrete(3),
        base_displacement_vector=gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        ),
        rotation_delta=gym.spaces.Box(
            low=-np.pi / 2.0, high=np.pi / 2.0, shape=(3,), dtype=np.float32
        ),
    )

    print("Building policy...")
    policy = RT1Policy(
        observation_space=observation_space,
        action_space=action_space,
    )
    policy.model.train()
    optimizer = Adam(policy.model.parameters(), lr=args.lr)
    text_embedding_model = (
        SentenceTransformer(args.sentence_transformer)
        if args.sentence_transformer
        else None
    )

    def get_text_embedding(observation: Dict):
        if text_embedding_model is not None:
            return text_embedding_model.encode(observation["instruction"])
        else:
            return observation["embedding"]

    print("Training...")
    for _ in range(args.epochs):
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
            print(f"Training loss: {loss.item()}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if args.eval_freq and num_batches % args.eval_freq == 0:
                print("Evaluating...")
                policy.model.eval()
                eval_loss = 0
                num_eval_batches = 0
                for batch in eval_dataset:
                    num_eval_batches += 1
                    observations = {
                        "image": batch["observation"]["image"],
                        "context": get_text_embedding(batch["observation"]),
                    }
                    actions = batch["action"]
                    eval_loss += policy.loss(observations, actions).item()
                print(f"eval loss: {eval_loss / num_eval_batches}")
            if args.checkpoint_freq and num_batches % args.checkpoint_freq == 0:
                checkpoint_path = (
                    f"{args.checkpoint_dir}/checkpoint_{num_batches}"
                    + f"_loss_{loss.item():.3f}.pt"
                )
                torch.save(policy.model.state_dict(), checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
