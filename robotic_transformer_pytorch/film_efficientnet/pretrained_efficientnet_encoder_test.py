"""Tests for pretrained_efficientnet_encoder."""

import unittest

import torch
from skimage import data

from robotic_transformer_pytorch.film_efficientnet import (
    FilmEfficientNetEncoder,
    decode_predictions,
)


class PretrainedEfficientnetEncoderTest(unittest.TestCase):
    def test_encoding(self):
        """Test that we get a correctly shaped encoding."""
        image = torch.tensor(data.chelsea()).repeat(10, 1, 1, 1)
        context = torch.FloatTensor(size=(10, 512)).uniform_(-1, 1)
        model = FilmEfficientNetEncoder().eval()
        preds = model(image, context)
        self.assertEqual(preds.shape, (10, 512, 10, 10))

    def test_imagenet_classification(self):
        """Test that we can correctly classify an image of a cat."""
        image = torch.tensor(data.chelsea())
        model = FilmEfficientNetEncoder(
            include_top=True,
        ).eval()
        preds = model(image)
        predicted_names = [n[0] for n in decode_predictions(preds, top=3)[0]]
        self.assertIn("tabby", predicted_names)


if __name__ == "__main__":
    unittest.main()
