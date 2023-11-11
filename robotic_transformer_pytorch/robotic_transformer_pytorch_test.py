import unittest

import torch
from skimage import data

from robotic_transformer_pytorch import RT1


class RT1Test(unittest.TestCase):
    def test_encoding(self):
        """Test that we get a correctly shaped encoding."""
        image = torch.tensor(data.chelsea()).repeat(6, 1, 1, 1).unsqueeze(0)
        context = torch.FloatTensor(size=(1, 512)).uniform_(-1, 1)
        model = RT1().eval()
        preds = model(image, context)
        self.assertEqual(preds.shape, (1, 11))


if __name__ == "__main__":
    unittest.main()
