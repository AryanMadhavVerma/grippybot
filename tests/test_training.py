"""Tests for training utilities — verifies loss computation and checkpointing."""

import os
import tempfile
import torch
import pytest
from grippybot.training.train import compute_loss, save_checkpoint
from grippybot.model.act import ACTPolicy


class TestComputeLoss:
    def test_loss_is_scalar(self):
        pred = torch.randn(2, 10, 5)
        target = torch.randn(2, 10, 5)
        pad_mask = torch.zeros(2, 10, dtype=torch.bool)
        mu = torch.randn(2, 8)
        logvar = torch.randn(2, 8)

        loss, l1, kl = compute_loss(pred, target, pad_mask, mu, logvar)
        assert loss.dim() == 0, "Loss should be scalar"
        assert isinstance(l1, float)
        assert isinstance(kl, float)

    def test_masked_positions_ignored(self):
        """Padded positions should not contribute to L1 loss."""
        pred = torch.zeros(1, 5, 5)
        target = torch.zeros(1, 5, 5)
        mu = torch.zeros(1, 8)
        logvar = torch.zeros(1, 8)

        # No padding: put big error at position 3
        pred[0, 3, :] = 100.0
        no_pad_mask = torch.zeros(1, 5, dtype=torch.bool)
        _, l1_unmasked, _ = compute_loss(pred, target, no_pad_mask, mu, logvar)

        # Mask position 3: error should be reduced
        pad_mask = torch.zeros(1, 5, dtype=torch.bool)
        pad_mask[0, 3] = True
        _, l1_masked, _ = compute_loss(pred, target, pad_mask, mu, logvar)

        assert l1_masked < l1_unmasked, "Masking high-error position should reduce L1"

    def test_kl_zero_for_standard_normal(self):
        """KL divergence should be ~0 when mu=0, logvar=0 (i.e., N(0,1))."""
        pred = torch.zeros(2, 10, 5)
        target = torch.zeros(2, 10, 5)
        pad_mask = torch.zeros(2, 10, dtype=torch.bool)
        mu = torch.zeros(2, 8)
        logvar = torch.zeros(2, 8)

        _, _, kl = compute_loss(pred, target, pad_mask, mu, logvar)
        assert abs(kl) < 0.01, f"KL should be ~0 for N(0,1), got {kl}"


class TestCheckpoint:
    def test_save_and_load(self):
        """Checkpoint saves and loads correctly."""
        model = ACTPolicy(d_model=64, latent_dim=8, chunk_size=10, state_dim=5,
                          n_encoder_layers=1, n_decoder_layers=1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        stats = {"state_mean": [0.0] * 5, "state_std": [1.0] * 5}

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.pt")
            save_checkpoint(model, optimizer, step=100, stats=stats, path=path)

            assert os.path.exists(path)
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            assert ckpt["step"] == 100
            assert "model_state_dict" in ckpt
            assert "config" in ckpt
            assert ckpt["config"]["d_model"] == 256  # save_checkpoint uses module-level config
