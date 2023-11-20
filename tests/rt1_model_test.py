import unittest

import torch

from robotic_transformer_pytorch.rt1_model import RT1Model


class RT1ModelTest(unittest.TestCase):
    def test_videos(self):
        model = RT1Model()

        videos = torch.rand(2, 6, 3, 224, 224)
        logits = model(videos)
        self.assertFalse(torch.isnan(logits).any())
        self.assertEqual(logits.shape, (2, 11, 256))

    def test_videos_and_texts(self):
        model = RT1Model()

        videos = torch.rand(2, 6, 3, 224, 224)
        texts = torch.rand(2, 6, 512)
        logits = model(videos, texts)
        self.assertFalse(torch.isnan(logits).any())
        self.assertEqual(logits.shape, (2, 11, 256))

    def test_videos_and_actions(self):
        model = RT1Model()

        videos = torch.rand(2, 6, 3, 224, 224)
        actions = torch.rand(2, 11, 256)
        logits = model(videos, actions=actions)
        self.assertFalse(torch.isnan(logits).any())
        self.assertEqual(logits.shape, (2, 11, 256))

    def test_videos_and_texts_and_actions(self):
        model = RT1Model()

        videos = torch.rand(2, 6, 3, 224, 224)
        texts = torch.rand(2, 6, 512)
        actions = torch.rand(2, 11, 256)
        logits = model(videos, texts, actions)
        self.assertFalse(torch.isnan(logits).any())
        self.assertEqual(logits.shape, (2, 11, 256))

    # TODO (Rohan138): Add more tests


if __name__ == "__main__":
    unittest.main()
