import unittest

import numpy as np
from gymnasium.spaces import Box, Dict, Discrete, Text
from skimage import data

from robotic_transformer_pytorch.rt1_policy import RT1Policy


class RT1PolicyTest(unittest.TestCase):
    def test_encoding_single_image(self):
        """Test that we get a correctly shaped encoding."""
        observation_space = Dict(
            image=Box(low=0, high=255, shape=(300, 451, 3), dtype=np.uint8),
            context=Box(low=0.0, high=1.0, shape=(512,), dtype=np.float32),
        )
        action_space = Dict(
            world_vector=Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
            rotation_delta=Box(
                low=-np.pi / 2.0, high=np.pi / 2.0, shape=(3,), dtype=np.float32
            ),
            gripper_closedness_action=Box(
                low=-1.0, high=1.0, shape=(1,), dtype=np.float32
            ),
            terminate_episode=Discrete(2),
        )
        policy = RT1Policy(observation_space, action_space)

        image = data.chelsea()
        videos = np.reshape(image, (1, 1, *image.shape)).repeat(6, axis=1)
        # videos (b, f, h, w, c) = (1, 6, 300, 451, 3)
        context = np.random.rand(1, 512).astype(np.float32)
        # context (b, d) = (1, 512)
        observation = {"image": videos, "context": context}
        actions = policy.act(observation)
        action_tokens = policy.action_tokenizer.tokenize(actions)

        self.assertEqual(action_tokens.shape, (1, 6, 8))
        self.assertTrue(action_space.contains(actions))
        print(actions)

    # TODO (Rohan138): Add more tests


if __name__ == "__main__":
    unittest.main()
