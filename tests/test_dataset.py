"""Tests for ACT dataset — verifies data loading, normalization, and shapes."""

import json
import os
import tempfile
import numpy as np
import pytest
from PIL import Image
from grippybot.model.dataset import ACTDataset, step_to_state


@pytest.fixture
def sample_data_dir():
    """Create a minimal episode for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ep_dir = os.path.join(tmpdir, "episode_000")
        os.makedirs(ep_dir)

        steps = []
        for i in range(10):
            frame_name = f"frame_{i:04d}.jpg"
            # Create a small dummy JPEG
            img = Image.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
            img.save(os.path.join(ep_dir, frame_name), quality=50)
            steps.append({
                "timestamp": i * 0.167,
                "joint_angles": {
                    "base": float(i * 2),
                    "shoulder": float(10 + i),
                    "elbow": float(20 - i),
                    "wrist": float(50 + i * 3),
                },
                "gripper_state": 1 if i < 5 else 0,
                "delta": {"base": 2.0, "shoulder": 1.0, "elbow": -1.0, "wrist": 3.0, "gripper": 0},
                "frame": frame_name,
            })

        with open(os.path.join(ep_dir, "episode.json"), "w") as f:
            json.dump({"steps": steps}, f)

        yield tmpdir


class TestStepToState:
    def test_extracts_5_dims(self):
        step = {
            "joint_angles": {"base": 10.0, "shoulder": 20.0, "elbow": 30.0, "wrist": 40.0},
            "gripper_state": 1,
        }
        state = step_to_state(step)
        assert state.shape == (5,)
        np.testing.assert_array_equal(state, [10.0, 20.0, 30.0, 40.0, 1.0])


class TestACTDataset:
    def test_loads_episodes(self, sample_data_dir):
        ds = ACTDataset(sample_data_dir, chunk_size=5)
        assert len(ds) == 9  # 10 steps, exclude last

    def test_sample_shapes(self, sample_data_dir):
        ds = ACTDataset(sample_data_dir, chunk_size=5)
        sample = ds[0]
        assert sample["image"].shape == (3, 480, 640)
        assert sample["state"].shape == (5,)
        assert sample["actions"].shape == (5, 5)
        assert sample["action_is_pad"].shape == (5,)

    def test_padding_at_end(self, sample_data_dir):
        ds = ACTDataset(sample_data_dir, chunk_size=5)
        # Last valid sample (step 8) has only 1 future step, rest padded
        sample = ds[len(ds) - 1]
        pad_count = sample["action_is_pad"].sum().item()
        assert pad_count == 4, f"Expected 4 padded actions, got {pad_count}"

    def test_no_padding_early(self, sample_data_dir):
        ds = ACTDataset(sample_data_dir, chunk_size=5)
        sample = ds[0]  # step 0 has 9 future steps, chunk=5, no padding
        assert sample["action_is_pad"].sum().item() == 0

    def test_normalization_stats(self, sample_data_dir):
        stats = ACTDataset.compute_stats(sample_data_dir)
        assert "state_mean" in stats
        assert "state_std" in stats
        assert len(stats["state_mean"]) == 5
        assert all(s > 0 for s in stats["state_std"]), "Std should be positive"

    def test_normalization_is_applied(self, sample_data_dir):
        ds = ACTDataset(sample_data_dir, chunk_size=5)
        sample = ds[0]
        # Normalized state should have roughly zero mean (for single episode, close enough)
        # Just verify it's not the raw values
        raw = step_to_state(ds.episodes[0]["steps"][0])
        assert not np.allclose(sample["state"].numpy(), raw), "State should be normalized"
