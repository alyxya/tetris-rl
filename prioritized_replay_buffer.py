"""
Prioritized Experience Replay Buffer.

Samples transitions with probability proportional to their TD-error,
ensuring rare but important experiences (line clears, deaths) are learned from more frequently.

Based on: Schaul et al. 2015 - "Prioritized Experience Replay"
https://arxiv.org/abs/1511.05952
"""

import numpy as np


class SumTree:
    """Binary heap where parent = sum(children), used for efficient prioritized sampling."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write_idx = 0
        self.size = 0

    def _propagate(self, idx, change):
        """Update tree node and propagate change to parents."""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        """Find sample on leaf node given cumulative sum s."""
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        """Return sum of all priorities."""
        return self.tree[0]

    def add(self, priority, data):
        """Add new experience with given priority."""
        idx = self.write_idx + self.capacity - 1

        self.data[self.write_idx] = data
        self.update(idx, priority)

        self.write_idx = (self.write_idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update(self, idx, priority):
        """Update priority of existing experience."""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s):
        """Get experience with cumulative priority s."""
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay with importance sampling.

    Samples transitions proportional to their TD-error, with importance sampling
    weights to correct for bias introduced by non-uniform sampling.
    """

    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000):
        """
        Args:
            capacity: Maximum buffer size
            alpha: How much prioritization to use (0 = uniform, 1 = full priority)
            beta_start: Initial importance sampling weight (0 = no correction, 1 = full correction)
            beta_frames: Number of frames to anneal beta to 1.0
        """
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 0
        self.epsilon = 1e-6  # Small constant to ensure non-zero priorities
        self.max_priority = 1.0

    def _get_beta(self):
        """Anneal beta from beta_start to 1.0 over beta_frames."""
        return min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)

    def push(self, state_empty, state_filled, action, reward, next_empty, next_filled, done):
        """Add experience with maximum priority (will be updated after training)."""
        data = (state_empty, state_filled, action, reward, next_empty, next_filled, done)
        self.tree.add(self.max_priority, data)

    def sample(self, batch_size):
        """Sample batch with priorities and return importance weights."""
        batch = []
        indices = []
        priorities = []
        segment = self.tree.total() / batch_size

        beta = self._get_beta()
        self.frame += 1

        # Sample from each segment to ensure diversity
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)

            idx, priority, data = self.tree.get(s)

            batch.append(data)
            indices.append(idx)
            priorities.append(priority)

        # Compute importance sampling weights
        priorities = np.array(priorities)
        sampling_probs = priorities / self.tree.total()

        # IS weights: (N * P(i))^(-beta)
        weights = (self.tree.size * sampling_probs) ** (-beta)
        weights /= weights.max()  # Normalize by max weight for stability

        # Unpack batch
        states_empty = np.array([x[0] for x in batch])
        states_filled = np.array([x[1] for x in batch])
        actions = np.array([x[2] for x in batch])
        rewards = np.array([x[3] for x in batch])
        next_empty = np.array([x[4] for x in batch])
        next_filled = np.array([x[5] for x in batch])
        dones = np.array([x[6] for x in batch])

        return (states_empty, states_filled, actions, rewards, next_empty, next_filled, dones,
                indices, weights)

    def update_priorities(self, indices, td_errors):
        """Update priorities based on TD-errors after training step."""
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        return self.tree.size
