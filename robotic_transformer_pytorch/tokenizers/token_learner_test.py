"""Tests for token_learner."""
import unittest

import torch
from absl.testing import parameterized

from robotic_transformer_pytorch.tokenizers import TokenLearnerModule


class TokenLearnerTest(parameterized.TestCase):
    @parameterized.named_parameters(("sample_input", 512, 8))
    def testTokenLearner(self, embedding_dim, num_tokens):
        batch = 1
        seq = 2
        token_learner_layer = TokenLearnerModule(
            embedding_dim=embedding_dim, num_tokens=num_tokens
        )

        inputvec = torch.randn((batch * seq, 100, embedding_dim))

        learnedtokens = token_learner_layer(inputvec)
        self.assertEqual(learnedtokens.shape, (batch * seq, num_tokens, embedding_dim))


if __name__ == "__main__":
    unittest.main()
