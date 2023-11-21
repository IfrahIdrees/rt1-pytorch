import d4rl
import gymnasium as gym

from rt1_pytorch.rt1_policy import RT1Policy

env = gym.make("GymV21Environment-v0", env_id="halfcheetah-expert-v2")
dataset = env.get_dataset()
