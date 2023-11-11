"""Tests for image_tokenizer."""
import unittest

import torch
from absl.testing import parameterized

from robotic_transformer_pytorch.tokenizers import RT1ImageTokenizer


class ImageTokenizerTest(parameterized.TestCase):
    @parameterized.named_parameters(
        ("sample_image", 512, 224, False, 8),
        ("sample_image_token_learner", 512, 224, True, 8),
    )
    def testTokenize(self, output_dim, image_resolution, use_token_learner, num_tokens):
        batch = 1
        seq = 2
        tokenizer = RT1ImageTokenizer(
            embedding_output_dim=output_dim,
            use_token_learner=use_token_learner,
            num_tokens=num_tokens,
        )

        image = torch.randn((batch, seq, image_resolution, image_resolution, 3))
        image = torch.clip(image, 0.0, 1.0)
        context_vector = torch.FloatTensor(size=(batch, seq, 512)).uniform_()
        image_tokens = tokenizer(image, context_vector)
        if use_token_learner:
            self.assertEqual(image_tokens.shape, (batch, seq, num_tokens, 512))
        else:
            self.assertEqual(image_tokens.shape, (batch, seq, 100, 512))


if __name__ == "__main__":
    unittest.main()
