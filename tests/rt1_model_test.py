import torch
from absl.testing import absltest, parameterized

from rt1_pytorch.rt1_model import RT1Model


class RT1ModelTest(parameterized.TestCase):
    @parameterized.parameters(["cpu", "cuda"])
    def test_videos(self, device="cpu"):
        model = RT1Model(device=device)

        videos = torch.rand(2, 6, 3, 224, 224, device=device)
        logits = model(videos)
        self.assertFalse(torch.isnan(logits).any())
        self.assertEqual(logits.shape, (2, 6, 11, 256))

    @parameterized.parameters(["cpu", "cuda"])
    def test_videos_and_texts(self, device="cpu"):
        model = RT1Model(device=device)

        videos = torch.rand(2, 6, 3, 224, 224, device=device)
        texts = torch.rand(2, 6, 512, device=device)
        logits = model(videos, texts)
        self.assertFalse(torch.isnan(logits).any())
        self.assertEqual(logits.shape, (2, 6, 11, 256))

    @parameterized.parameters(["cpu", "cuda"])
    def test_videos_and_actions(self, device="cpu"):
        model = RT1Model(device=device)

        videos = torch.rand(2, 6, 3, 224, 224, device=device)
        actions = torch.rand(2, 6, 11, 256, device=device)
        logits = model(videos, actions=actions)
        self.assertFalse(torch.isnan(logits).any())
        self.assertEqual(logits.shape, (2, 6, 11, 256))

    @parameterized.parameters(["cpu", "cuda"])
    def test_videos_and_texts_and_actions(self, device="cpu"):
        model = RT1Model(device=device)

        videos = torch.rand(2, 6, 3, 224, 224, device=device)
        texts = torch.rand(2, 6, 512, device=device)
        actions = torch.rand(2, 6, 11, 256, device=device)
        logits = model(videos, texts, actions)
        self.assertFalse(torch.isnan(logits).any())
        self.assertEqual(logits.shape, (2, 6, 11, 256))


if __name__ == "__main__":
    absltest.main()
