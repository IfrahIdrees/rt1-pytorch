import unittest

import numpy as np
from gymnasium.spaces import Box, Dict, Discrete
from skimage import data

from robotic_transformer_pytorch.rt1 import RT1


class RT1Test(unittest.TestCase):
    def test_encoding(self):
        """Test that we get a correctly shaped encoding."""

        image = data.chelsea()
        image = np.reshape(image, (1, 1, *image.shape))
        # image (b, f, h, w, c) = (1, 1, 300, 451, 3)
        context = ["Move X to Y"]
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
        model = RT1(action_space=action_space).eval()
        preds = model(image, context)
        self.assertEqual(preds.shape, (1, 1, 11, 256))


if __name__ == "__main__":
    unittest.main()
