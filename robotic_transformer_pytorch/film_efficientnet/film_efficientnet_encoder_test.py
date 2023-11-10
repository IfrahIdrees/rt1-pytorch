"""Tests that film_efficientnet can detect an image of a cat."""

import unittest

import torch
from skimage import data
from torchvision.models import efficientnet_b3
from torchvision.models._meta import _IMAGENET_CATEGORIES

from robotic_transformer_pytorch.film_efficientnet import film_efficientnet_encoder


def decode_predictions(preds, top=5):
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [(_IMAGENET_CATEGORIES[i], pred[i]) for i in top_indices]
        results.append(result)
    return results


class FilmEfficientnetTest(unittest.TestCase):
    def test_equivalence_b3(self):
        image = torch.tensor(data.chelsea()).permute(2, 0, 1).unsqueeze(0) / 255
        context = torch.zeros(1, 512)

        model = efficientnet_b3(weights="DEFAULT").eval()
        model_output = model(image)
        model_preds = decode_predictions(model_output.cpu().detach().numpy(), top=10)
        print(model_preds)
        self.assertIn("tabby", [f[0] for f in model_preds[0]])

        encoder = film_efficientnet_encoder.filmefficientnet_b3(
            weights="DEFAULT",
            include_top=True,
        ).eval()
        eff_output = encoder(image, context)
        film_preds = decode_predictions(eff_output.cpu().detach().numpy(), top=10)
        print(film_preds)
        self.assertIn("tabby", [f[0] for f in film_preds[0]])


if __name__ == "__main__":
    unittest.main()
