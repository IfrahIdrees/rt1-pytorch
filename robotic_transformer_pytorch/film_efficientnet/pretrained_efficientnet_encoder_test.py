"""Tests for pretrained_efficientnet_encoder."""

import unittest

import torch
from robotic_transformer_pytorch.film_efficientnet import (
    film_efficientnet_encoder,
    pretrained_efficientnet_encoder,
)
from skimage import data


class PretrainedEfficientnetEncoderTest(unittest.TestCase):
    def test_encoding(self):
        """Test that we get a correctly shaped encoding."""
        image = torch.tensor(data.chelsea())
        context = torch.FloatTensor(size=(10, 512)).uniform_(-1, 1)
        model = pretrained_efficientnet_encoder.EfficientNetEncoder().eval()
        preds = model(image, context)
        self.assertEqual(preds.shape, (10, 512))

    def test_imagenet_classification(self):
        """Test that we can correctly classify an image of a cat."""
        image = torch.tensor(data.chelsea())
        model = pretrained_efficientnet_encoder.EfficientNetEncoder(
            include_top=True,
        ).eval()
        preds = model(image)
        predicted_names = [
            n[0] for n in film_efficientnet_encoder.decode_predictions(preds, top=3)[0]
        ]
        self.assertIn("tabby", predicted_names)


if __name__ == "__main__":
    unittest.main()
