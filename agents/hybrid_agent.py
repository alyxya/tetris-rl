"""
Hybrid agent that randomly switches between CNN and Heuristic modes.

This agent makes a 50/50 random choice for each action between:
- CNN-based policy (learned behavior)
- Heuristic-based policy (rule-based behavior)

This can be useful for:
- Exploring hybrid strategies
- Comparing agent behaviors in the same episode
- Creating diverse training data
"""

import numpy as np
import sys
import os

from agents.base_agent import BaseTetrisAgent
from agents.cnn_agent import CNNAgent
from agents.heuristic_agent import HeuristicAgent


class HybridAgent(BaseTetrisAgent):
    """Agent that randomly chooses between CNN and Heuristic policies."""

    def __init__(self, n_rows=20, n_cols=10, device='cpu', model_path=None):
        """
        Initialize hybrid agent with both CNN and heuristic sub-agents.

        Args:
            n_rows: Number of rows in board
            n_cols: Number of columns in board
            device: 'cpu' or 'cuda' (for CNN agent)
            model_path: Path to load pretrained CNN model (optional)
        """
        super().__init__(n_rows, n_cols)

        # Initialize both sub-agents
        self.cnn_agent = CNNAgent(n_rows, n_cols, device, model_path)
        self.heuristic_agent = HeuristicAgent(n_rows, n_cols)

        # Track which agent was used for statistics
        self.cnn_count = 0
        self.heuristic_count = 0

    def reset(self):
        """Reset both sub-agents."""
        self.cnn_agent.reset()
        self.heuristic_agent.reset()
        self.cnn_count = 0
        self.heuristic_count = 0

    def choose_action(self, obs, temperature=1.0, deterministic=False):
        """
        Choose action by randomly selecting between CNN and heuristic agents.

        Args:
            obs: flattened observation array
            temperature: sampling temperature (for CNN agent)
            deterministic: if True, use deterministic CNN policy

        Returns:
            action: integer action (0-6)
        """
        # 50/50 random choice
        if np.random.random() < 0.5:
            # Use CNN agent
            action = self.cnn_agent.choose_action(obs, temperature, deterministic)
            self.cnn_count += 1
        else:
            # Use heuristic agent
            action = self.heuristic_agent.choose_action(obs)
            self.heuristic_count += 1

        return action

    def get_usage_stats(self):
        """
        Get statistics on which agent was used more.

        Returns:
            dict with usage counts and percentages
        """
        total = self.cnn_count + self.heuristic_count
        if total == 0:
            return {
                'cnn_count': 0,
                'heuristic_count': 0,
                'cnn_percentage': 0.0,
                'heuristic_percentage': 0.0,
                'total_actions': 0
            }

        return {
            'cnn_count': self.cnn_count,
            'heuristic_count': self.heuristic_count,
            'cnn_percentage': (self.cnn_count / total) * 100,
            'heuristic_percentage': (self.heuristic_count / total) * 100,
            'total_actions': total
        }

    def save_model(self, path):
        """Save CNN model weights."""
        self.cnn_agent.save_model(path)

    def load_model(self, path):
        """Load CNN model weights."""
        self.cnn_agent.load_model(path)
