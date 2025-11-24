"""
Mixed teacher agent that combines random actions with heuristic agent.

The agent samples a random action probability and heuristic temperature once per episode.
Then for each action, it either takes a random action or uses the heuristic agent.
"""

import numpy as np
from base_agent import BaseAgent
from heuristic_agent import HeuristicAgent


class MixedTeacherAgent(BaseAgent):
    """Agent that mixes random actions with heuristic-based actions."""

    def __init__(self, n_rows=20, n_cols=10, heuristic_weights=None):
        """
        Initialize mixed teacher agent.

        Args:
            n_rows: Board height
            n_cols: Board width
            heuristic_weights: Weights for heuristic evaluation (or None for defaults)
        """
        super().__init__(n_rows, n_cols)

        # Store heuristic weights
        self.heuristic_weights = heuristic_weights

        # Per-episode parameters (sampled at reset)
        self.random_prob = 0.0
        self.heuristic_temperature = 0.0

        # Heuristic agent instance (will be created with temperature at reset)
        self.heuristic_agent = None

    def reset(self):
        """
        Reset agent state for new episode.

        Samples new random_prob and heuristic_temperature:
        - random_prob = (uniform(0, 1))^4
        - heuristic_temperature = (uniform(0, 1))^4
        """
        # Sample per-episode parameters
        self.random_prob = np.random.uniform(0, 1) ** 4
        self.heuristic_temperature = np.random.uniform(0, 1) ** 4

        # Create heuristic agent with sampled temperature
        self.heuristic_agent = HeuristicAgent(
            n_rows=self.n_rows,
            n_cols=self.n_cols,
            weights=self.heuristic_weights,
            temperature=self.heuristic_temperature
        )

    def choose_action(self, obs):
        """
        Choose action using mixed strategy.

        With probability random_prob, take random action.
        Otherwise, use heuristic agent with temperature.

        Args:
            obs: Flattened observation array from environment

        Returns:
            action: Integer action (0-6)
        """
        # Ensure we have initialized the per-episode parameters
        if self.heuristic_agent is None:
            self.reset()

        # Randomly choose between random action and heuristic action
        if np.random.uniform(0, 1) < self.random_prob:
            # Random action (uniform over all 7 actions)
            return np.random.randint(0, 7)
        else:
            # Use heuristic agent
            return self.heuristic_agent.choose_action(obs)
