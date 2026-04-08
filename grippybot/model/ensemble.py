"""
Temporal ensemble — averages overlapping action chunk predictions.

Used at inference time to smooth trajectories. Each step queries the policy
for a full chunk of future actions. Overlapping predictions are averaged
using exponential decay weights (recent predictions weighted more).
"""

import numpy as np


class TemporalEnsemble:
    """Averages overlapping action chunk predictions using exponential weighting."""

    def __init__(self, chunk_size, state_dim, decay=0.01):
        self.chunk_size = chunk_size
        self.state_dim = state_dim
        self.decay = decay
        self.buffer = {}  # timestep -> (weight_sum, weighted_action_sum)
        self.current_step = 0

    def add_chunk(self, actions):
        """Add a new chunk of predicted actions. actions: [chunk_size, state_dim]"""
        for i in range(self.chunk_size):
            future_step = self.current_step + i
            weight = np.exp(-self.decay * i)
            if future_step not in self.buffer:
                self.buffer[future_step] = (0.0, np.zeros(self.state_dim))
            w_sum, a_sum = self.buffer[future_step]
            self.buffer[future_step] = (w_sum + weight, a_sum + weight * actions[i])

    def get_action(self):
        """Get the ensembled action for the current timestep and advance."""
        if self.current_step not in self.buffer:
            return None
        w_sum, a_sum = self.buffer[self.current_step]
        action = a_sum / w_sum
        del self.buffer[self.current_step]
        self.current_step += 1
        return action

    def reset(self):
        """Reset the ensemble state."""
        self.buffer = {}
        self.current_step = 0
