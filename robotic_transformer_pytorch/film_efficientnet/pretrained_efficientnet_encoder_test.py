"""Tests for pretrained_efficientnet_encoder."""

import unittest

import torch
from skimage import data

from robotic_transformer_pytorch.film_efficientnet import film_efficientnet_encoder
from robotic_transformer_pytorch.film_efficientnet import (
    pretrained_efficientnet_encoder as eff,
)


class PretrainedEfficientnetEncoderTest(unittest.TestCase):
    def test_encoding(self):
        """Test that we get a correctly shaped decoding."""
        context = torch.FloatTensor((10, 512)).uniform_(-1, 1)
        model = eff.EfficientNetEncoder()
        image = torch.tensor(data.chelsea()).permute(2, 0, 1).unsqueeze(0) / 255
        preds = model(image, context, training=False).numpy()
        self.assertEqual(preds.shape, (10, 512))

    def test_imagenet_classification(self):
        """Test that we can correctly classify an image of a cat."""
        context = torch.FloatTensor((10, 512)).uniform_(-1, 1)
        model = eff.EfficientNetEncoder(include_top=True)
        image = torch.tensor(data.chelsea()).permute(2, 0, 1).unsqueeze(0) / 255
        preds = model._encode(image, context, training=False).numpy()
        predicted_names = [
            n[1] for n in film_efficientnet_encoder.decode_predictions(preds, top=3)[0]
        ]
        self.assertIn("tabby", predicted_names)


if __name__ == "__main__":
    unittest.main()
