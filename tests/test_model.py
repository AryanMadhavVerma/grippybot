"""Tests for ACT model — verifies shapes, train/eval modes, and components."""

import torch
import pytest
from grippybot.model.act import (
    ACTPolicy, VisionBackbone, CVAEEncoder, ObservationFuser, ActionDecoder,
    sinusoidal_pos_embedding, sinusoidal_2d_pos_embedding,
)


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def model(device):
    return ACTPolicy(d_model=64, latent_dim=8, chunk_size=10, state_dim=5,
                     n_encoder_layers=1, n_decoder_layers=1).to(device)


class TestACTPolicy:
    def test_train_forward_shapes(self, model, device):
        """Training forward pass produces correct output shapes."""
        image = torch.randn(2, 3, 480, 640, device=device)
        state = torch.randn(2, 5, device=device)
        actions = torch.randn(2, 10, 5, device=device)

        model.train()
        pred, mu, logvar = model(image, state, actions)

        assert pred.shape == (2, 10, 5), f"Expected (2, 10, 5), got {pred.shape}"
        assert mu.shape == (2, 8), f"Expected (2, 8), got {mu.shape}"
        assert logvar.shape == (2, 8), f"Expected (2, 8), got {logvar.shape}"

    def test_eval_forward_shapes(self, model, device):
        """Inference forward pass produces correct shapes, no CVAE output."""
        image = torch.randn(2, 3, 480, 640, device=device)
        state = torch.randn(2, 5, device=device)

        model.eval()
        pred, mu, logvar = model(image, state)

        assert pred.shape == (2, 10, 5)
        assert mu is None
        assert logvar is None

    def test_eval_is_deterministic(self, model, device):
        """Same input produces same output in eval mode."""
        image = torch.randn(1, 3, 480, 640, device=device)
        state = torch.randn(1, 5, device=device)

        model.eval()
        pred1, _, _ = model(image, state)
        pred2, _, _ = model(image, state)

        assert torch.allclose(pred1, pred2), "Eval mode should be deterministic"

    def test_gradients_flow(self, model, device):
        """Loss backward produces gradients on all parameters."""
        image = torch.randn(2, 3, 480, 640, device=device)
        state = torch.randn(2, 5, device=device)
        actions = torch.randn(2, 10, 5, device=device)

        model.train()
        pred, mu, logvar = model(image, state, actions)
        loss = pred.mean() + mu.mean() + logvar.mean()
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_batch_size_one(self, model, device):
        """Works with batch size 1."""
        image = torch.randn(1, 3, 480, 640, device=device)
        state = torch.randn(1, 5, device=device)

        model.eval()
        pred, _, _ = model(image, state)
        assert pred.shape == (1, 10, 5)


class TestVisionBackbone:
    def test_output_shape(self, device):
        backbone = VisionBackbone(d_model=64).to(device)
        image = torch.randn(2, 3, 480, 640, device=device)
        tokens = backbone(image)
        assert tokens.shape == (2, 300, 64), f"Expected (2, 300, 64), got {tokens.shape}"


class TestPositionalEmbeddings:
    def test_1d_shape(self):
        pe = sinusoidal_pos_embedding(50, 64)
        assert pe.shape == (50, 64)

    def test_2d_shape(self):
        pe = sinusoidal_2d_pos_embedding(15, 20, 64)
        assert pe.shape == (300, 64)

    def test_2d_values_bounded(self):
        pe = sinusoidal_2d_pos_embedding(15, 20, 64)
        assert pe.abs().max() <= 1.0, "Sinusoidal embeddings should be in [-1, 1]"
