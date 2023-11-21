"""Tests for image_tokenizer."""
import unittest

import torch
from absl.testing import parameterized

from rt1_pytorch.tokenizers.image_tokenizer import RT1ImageTokenizer


class ImageTokenizerTest(parameterized.TestCase):
    @parameterized.named_parameters(
        ("sample_image", 512, 224, False, 8),
        ("sample_image_token_learner", 512, 224, True, 8),
    )
    def testTokenize(
        self, embedding_dim, image_resolution, use_token_learner, num_tokens
    ):
        batch = 2
        device = "cuda"
        tokenizer = RT1ImageTokenizer(
            embedding_dim=embedding_dim,
            use_token_learner=use_token_learner,
            token_learner_num_output_tokens=num_tokens,
            device=device,
        )

        image = torch.randn((batch, image_resolution, image_resolution, 3))
        image = torch.clip(image, 0.0, 1.0)
        image = image.to(device)
        context_vector = torch.FloatTensor(size=(batch, 512)).uniform_()
        context_vector = context_vector.to(device)
        image_tokens = tokenizer(image, context_vector)
        if use_token_learner:
            self.assertEqual(image_tokens.shape, (batch, 512, num_tokens))
        else:
            self.assertEqual(image_tokens.shape, (batch, 512, 100))


if __name__ == "__main__":
    unittest.main()
