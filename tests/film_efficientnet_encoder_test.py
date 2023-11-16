"""Tests for pretrained_efficientnet_encoder."""

import unittest

import torch
from skimage import data

from robotic_transformer_pytorch.film_efficientnet.film_efficientnet_encoder import (
    decode_predictions,
)
from robotic_transformer_pytorch.film_efficientnet.pretrained_efficientnet_encoder import (
    FilmEfficientNetEncoder,
)


class PretrainedEfficientnetEncoderTest(unittest.TestCase):
    def test_encoding(self):
        """Test that we get a correctly shaped encoding."""
        embedding_dim = 512
        image = torch.tensor(data.chelsea()).repeat(10, 1, 1, 1)
        context = torch.FloatTensor(size=(10, embedding_dim)).uniform_(-1, 1)
        model = FilmEfficientNetEncoder(embedding_dim=embedding_dim).eval()
        preds = model(image, context)
        self.assertEqual(preds.shape, (10, 512, 10, 10))

    def test_imagenet_classification(self):
        """Test that we can correctly classify an image of a cat."""
        embedding_dim = 512
        image = torch.tensor(data.chelsea())
        model = FilmEfficientNetEncoder(
            include_top=True,
            embedding_dim=embedding_dim,
        ).eval()
        preds = model(image)
        predicted_names = [n[0] for n in decode_predictions(preds, top=3)[0]]
        self.assertIn("tabby", predicted_names)


if __name__ == "__main__":
    unittest.main()
