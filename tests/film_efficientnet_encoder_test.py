"""Tests that film_efficientnet can detect an image of a cat."""

import unittest

import torch
from skimage import data
from torchvision.models import EfficientNet_B3_Weights, efficientnet_b3

from robotic_transformer_pytorch.film_efficientnet.film_efficientnet_encoder import (
    decode_predictions,
    filmefficientnet_b3,
)


class FilmEfficientnetTest(unittest.TestCase):
    def test_equivalence_b3(self):
        image = torch.tensor(data.chelsea()).permute(2, 0, 1).unsqueeze(0) / 255
        preprocess = EfficientNet_B3_Weights.DEFAULT.transforms(antialias=True)
        image = preprocess(image)
        context = torch.zeros(1, 384)

        model = efficientnet_b3(weights="DEFAULT").eval()
        model_output = model(image)
        model_preds = decode_predictions(model_output, top=3)
        print(model_preds)
        self.assertIn("tabby", [f[0] for f in model_preds[0]])

        encoder = filmefficientnet_b3(
            weights="DEFAULT",
            include_top=True,
        ).eval()
        eff_output = encoder(image, context)
        film_preds = decode_predictions(eff_output, top=3)
        print(film_preds)
        self.assertIn("tabby", [f[0] for f in film_preds[0]])


if __name__ == "__main__":
    unittest.main()
