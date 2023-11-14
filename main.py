import argparse

import gymnasium as gym
import numpy as np
import tensorflow_datasets as tfds
from torch.optim import Adam

from robotic_transformer_pytorch.rt1_policy import RT1Policy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name",
        type=str,
        default="fractal20220817_data",
        choices=["fractal20220817_data", "bc_z"],
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
        "--batch-size",
        type=int,
        default=32,
        help="batch size",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=100,
        help="number of frames per episode",
    )
    return parser.parse_args()


def load_dataset(name):
    if name == "robo_net":
        version = "1.0.0"
    elif name == "language_table":
        version = "0.0.1"
    else:
        version = "0.1.0"
    path = f"gs://gresearch/robotics/{name}/{version}"
    builder = tfds.builder_from_directory(path)
    builder.download_and_prepare()
    ds = builder.as_dataset(split="train[:100]")

    def map_fn(episode):
        return {
            "observation": {
                "image": [
                    step["observation"]["image"]
                    for step in episode["steps"].as_numpy_iterator()
                ],
                "context": [
                    step["observation"]["natural_language_instruction"]
                    for step in episode["steps"].as_numpy_iterator()
                ],
            },
            "action": [step["action"] for step in episode["steps"].as_numpy_iterator()],
        }

    ds = ds.map(map_fn, num_parallel_calls=-1).unbatch()

    # shuffle, repeat, pre-fetch, batch
    ds = ds.cache()  # optionally keep full dataset in memory
    ds = ds.shuffle(100)  # set shuffle buffer size
    ds = ds.repeat()  # repeat indefinitely

    breakpoint()

    return ds.as_numpy_iterator()


def main():
    args = parse_args()
    print("Loading dataset...")
    dataset = load_dataset(args.name)

    observation_space = gym.spaces.Dict(
        image=gym.spaces.Box(low=0, high=255, shape=(128, 128, 3)),
        context=gym.spaces.Text(max_length=256),
    )
    action_space = gym.spaces.Dict(
        world_vector=gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
        rotation_delta=gym.spaces.Box(
            low=-np.pi / 2.0, high=np.pi / 2.0, shape=(3,), dtype=np.float32
        ),
        gripper_closedness_action=gym.spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        ),
    )

    print("Building policy...")
    policy = RT1Policy(
        observation_space=observation_space,
        action_space=action_space,
    )
    policy.train()
    optimizer = Adam(policy.parameters(), lr=args.lr)

    print("Starting training...")
    for _ in range(args.epochs):
        for batch in dataset:
            videos = batch["observation"]["image"]
            texts = batch.get("context")
            prev_actions = batch["action"]
            breakpoint()
            loss = policy.loss(videos, texts, prev_actions, batch["action"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(loss)


if __name__ == "__main__":
    main()
