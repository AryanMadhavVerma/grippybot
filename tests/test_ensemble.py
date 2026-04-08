"""Tests for temporal ensemble — verifies averaging and state management."""

import numpy as np
import pytest
from grippybot.model.ensemble import TemporalEnsemble


class TestTemporalEnsemble:
    def test_single_chunk(self):
        """Single chunk returns first action."""
        ens = TemporalEnsemble(chunk_size=5, state_dim=3)
        actions = np.array([[1, 2, 3]] * 5, dtype=np.float32)
        ens.add_chunk(actions)
        action = ens.get_action()
        np.testing.assert_array_almost_equal(action, [1, 2, 3])

    def test_multiple_chunks_smooth(self):
        """Overlapping chunks produce smoothed output."""
        ens = TemporalEnsemble(chunk_size=5, state_dim=1, decay=0.0)

        # First chunk: all 10s
        ens.add_chunk(np.full((5, 1), 10.0))
        action1 = ens.get_action()

        # Second chunk: all 20s
        ens.add_chunk(np.full((5, 1), 20.0))
        action2 = ens.get_action()

        # With decay=0 (equal weights), step 1 has predictions from both chunks
        # Chunk 1 predicted 10 for step 1, chunk 2 predicts 20 for step 1
        assert action1[0] == pytest.approx(10.0)  # only one chunk
        assert action2[0] == pytest.approx(15.0)  # average of 10 and 20

    def test_reset(self):
        """Reset clears all state."""
        ens = TemporalEnsemble(chunk_size=5, state_dim=2)
        ens.add_chunk(np.ones((5, 2)))
        ens.get_action()
        ens.reset()
        assert ens.current_step == 0
        assert len(ens.buffer) == 0

    def test_get_action_advances_step(self):
        """Each get_action call advances the step counter."""
        ens = TemporalEnsemble(chunk_size=5, state_dim=2)
        ens.add_chunk(np.ones((5, 2)))
        assert ens.current_step == 0
        ens.get_action()
        assert ens.current_step == 1
        ens.get_action()
        assert ens.current_step == 2

    def test_empty_buffer_returns_none(self):
        """get_action returns None if no predictions for current step."""
        ens = TemporalEnsemble(chunk_size=2, state_dim=1)
        ens.add_chunk(np.ones((2, 1)))
        ens.get_action()  # step 0
        ens.get_action()  # step 1
        result = ens.get_action()  # step 2 — no data
        assert result is None

    def test_decay_weights_recent_more(self):
        """With nonzero decay, recent predictions weighted higher."""
        ens = TemporalEnsemble(chunk_size=3, state_dim=1, decay=1.0)

        # Step 0: chunk predicts [0, 0, 0]
        ens.add_chunk(np.zeros((3, 1)))
        ens.get_action()  # consume step 0

        # Step 1: new chunk predicts [10, 10, 10]
        ens.add_chunk(np.full((3, 1), 10.0))
        action = ens.get_action()

        # Step 1 has: old chunk (weight=exp(-1)=0.368, value=0) + new chunk (weight=1.0, value=10)
        # Expected: (0.368*0 + 1.0*10) / (0.368 + 1.0) = 7.31
        assert action[0] > 5.0, "Recent prediction should dominate with high decay"
        assert action[0] < 10.0, "Old prediction should still have some influence"
