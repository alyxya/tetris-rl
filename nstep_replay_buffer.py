"""
N-step replay buffer for better credit assignment.

Stores sequences of N transitions and computes N-step returns:
  Q_target = r1 + γ*r2 + γ²*r3 + ... + γⁿ*max(Q(state_n))

This helps propagate rewards backward to positioning actions (LEFT, ROTATE)
that don't immediately receive rewards.
"""

import numpy as np
from collections import deque


class NStepReplayBuffer:
    """N-step replay buffer with uniform sampling."""

    def __init__(self, capacity, n_step=3, gamma=0.99):
        """
        Args:
            capacity: Maximum buffer size
            n_step: Number of steps to look ahead for returns
            gamma: Discount factor
        """
        self.capacity = capacity
        self.n_step = n_step
        self.gamma = gamma
        self.buffer = deque(maxlen=capacity)
        self.n_step_buffer = deque(maxlen=n_step)

    def push(self, state_empty, state_filled, action, reward, next_empty, next_filled, done):
        """Add experience to n-step buffer, then to main buffer when ready."""
        self.n_step_buffer.append((state_empty, state_filled, action, reward, next_empty, next_filled, done))

        # If we have n_step transitions OR episode ended, compute n-step return
        if len(self.n_step_buffer) == self.n_step or done:
            # Get the oldest transition
            state_empty_0, state_filled_0, action_0, _, _, _, _ = self.n_step_buffer[0]

            # Compute n-step return
            n_step_reward = 0.0
            for i, (_, _, _, r, _, _, d) in enumerate(self.n_step_buffer):
                n_step_reward += (self.gamma ** i) * r
                if d:
                    # Episode ended, use this as final state
                    final_next_empty = self.n_step_buffer[i][4]
                    final_next_filled = self.n_step_buffer[i][5]
                    final_done = True
                    break
            else:
                # No early termination, use the last transition's next_state
                final_next_empty = self.n_step_buffer[-1][4]
                final_next_filled = self.n_step_buffer[-1][5]
                final_done = False

            # Store n-step transition
            self.buffer.append((
                state_empty_0, state_filled_0, action_0, n_step_reward,
                final_next_empty, final_next_filled, final_done
            ))

            # If episode ended, clear n-step buffer
            if done:
                self.n_step_buffer.clear()

    def sample(self, batch_size):
        """Sample random batch from buffer."""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]

        states_empty = np.array([x[0] for x in batch])
        states_filled = np.array([x[1] for x in batch])
        actions = np.array([x[2] for x in batch])
        rewards = np.array([x[3] for x in batch])
        next_empty = np.array([x[4] for x in batch])
        next_filled = np.array([x[5] for x in batch])
        dones = np.array([x[6] for x in batch])

        return states_empty, states_filled, actions, rewards, next_empty, next_filled, dones

    def __len__(self):
        return len(self.buffer)


class NStepPrioritizedReplayBuffer:
    """N-step replay buffer with prioritized sampling."""

    def __init__(self, capacity, n_step=3, gamma=0.99, alpha=0.6, beta_start=0.4, beta_frames=100000):
        """
        Args:
            capacity: Maximum buffer size
            n_step: Number of steps to look ahead for returns
            gamma: Discount factor
            alpha: Prioritization exponent
            beta_start: Initial importance sampling weight
            beta_frames: Frames to anneal beta to 1.0
        """
        from prioritized_replay_buffer import SumTree

        self.capacity = capacity
        self.n_step = n_step
        self.gamma = gamma
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 0
        self.epsilon = 1e-6
        self.max_priority = 1.0

        self.tree = SumTree(capacity)
        self.n_step_buffer = deque(maxlen=n_step)

    def _get_beta(self):
        """Anneal beta from beta_start to 1.0 over beta_frames."""
        return min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)

    def push(self, state_empty, state_filled, action, reward, next_empty, next_filled, done):
        """Add experience to n-step buffer, then to main buffer when ready."""
        self.n_step_buffer.append((state_empty, state_filled, action, reward, next_empty, next_filled, done))

        # If we have n_step transitions OR episode ended, compute n-step return
        if len(self.n_step_buffer) == self.n_step or done:
            # Get the oldest transition
            state_empty_0, state_filled_0, action_0, _, _, _, _ = self.n_step_buffer[0]

            # Compute n-step return
            n_step_reward = 0.0
            for i, (_, _, _, r, _, _, d) in enumerate(self.n_step_buffer):
                n_step_reward += (self.gamma ** i) * r
                if d:
                    # Episode ended, use this as final state
                    final_next_empty = self.n_step_buffer[i][4]
                    final_next_filled = self.n_step_buffer[i][5]
                    final_done = True
                    break
            else:
                # No early termination, use the last transition's next_state
                final_next_empty = self.n_step_buffer[-1][4]
                final_next_filled = self.n_step_buffer[-1][5]
                final_done = False

            # Store n-step transition with max priority
            data = (
                state_empty_0, state_filled_0, action_0, n_step_reward,
                final_next_empty, final_next_filled, final_done
            )
            self.tree.add(self.max_priority, data)

            # If episode ended, clear n-step buffer
            if done:
                self.n_step_buffer.clear()

    def sample(self, batch_size):
        """Sample batch with priorities and return importance weights."""
        batch = []
        indices = []
        priorities = []
        segment = self.tree.total() / batch_size

        beta = self._get_beta()
        self.frame += 1

        # Sample from each segment
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

        weights = (self.tree.size * sampling_probs) ** (-beta)
        weights /= weights.max()

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
        """Update priorities based on TD-errors."""
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        return self.tree.size
