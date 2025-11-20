"""
Hybrid agent that randomly switches between Student (Value) and Teacher (Heuristic) modes.

This agent makes a random choice for each action between:
- Student: Value-based policy (learned behavior)
- Teacher: Heuristic-based policy (rule-based behavior)

This can be useful for:
- Exploring hybrid strategies
- Comparing agent behaviors in the same episode
- Creating diverse training data
"""

import numpy as np
import sys
import os

from agents.base_agent import BaseTetrisAgent
from agents.q_agent import QValueAgent
from agents.heuristic_agent import HeuristicAgent


class HybridAgent(BaseTetrisAgent):
    """Agent that randomly chooses between Student (Q-value), Teacher (Heuristic), and Random policies."""

    def __init__(self, n_rows=20, n_cols=10, device='cpu', model_path=None, student_probability=0.5, random_probability=0.0):
        """
        Initialize hybrid agent with student, teacher, and random sub-agents.

        Args:
            n_rows: Number of rows in board
            n_cols: Number of columns in board
            device: 'cpu' or 'cuda' (for student agent)
            model_path: Path to load pretrained student model (optional)
            student_probability: Probability of using student (default: 0.5)
            random_probability: Probability of using random action (default: 0.0)
        """
        super().__init__(n_rows, n_cols)

        # Initialize both sub-agents
        self.student_agent = QValueAgent(n_rows, n_cols, device, model_path)
        self.teacher_agent = HeuristicAgent(n_rows, n_cols)

        # Store probabilities
        self.student_probability = student_probability
        self.random_probability = random_probability

        # Track which agent was used for statistics
        self.student_count = 0
        self.teacher_count = 0
        self.random_count = 0

    def reset(self):
        """Reset both sub-agents."""
        self.student_agent.reset()
        self.teacher_agent.reset()
        self.student_count = 0
        self.teacher_count = 0
        self.random_count = 0

    def choose_action(self, obs, epsilon=0.0, deterministic=False):
        """
        Choose action by randomly selecting between student, teacher, and random.

        Args:
            obs: flattened observation array
            epsilon: epsilon-greedy exploration (for student agent)
            deterministic: if True, use deterministic student policy

        Returns:
            action: integer action (0-6)
        """
        # Three-way random choice
        rand = np.random.random()
        if rand < self.student_probability:
            # Use student agent (value-based)
            action = self.student_agent.choose_action(obs, epsilon, deterministic)
            self.student_count += 1
        elif rand < self.student_probability + self.random_probability:
            # Use random action
            action = np.random.randint(0, 7)
            self.random_count += 1
        else:
            # Use teacher agent (heuristic)
            action = self.teacher_agent.choose_action(obs)
            self.teacher_count += 1

        return action

    def get_usage_stats(self):
        """
        Get statistics on which agent was used.

        Returns:
            dict with usage counts and percentages
        """
        total = self.student_count + self.teacher_count + self.random_count
        if total == 0:
            return {
                'student_count': 0,
                'teacher_count': 0,
                'random_count': 0,
                'student_percentage': 0.0,
                'teacher_percentage': 0.0,
                'random_percentage': 0.0,
                'total_actions': 0
            }

        return {
            'student_count': self.student_count,
            'teacher_count': self.teacher_count,
            'random_count': self.random_count,
            'student_percentage': (self.student_count / total) * 100,
            'teacher_percentage': (self.teacher_count / total) * 100,
            'random_percentage': (self.random_count / total) * 100,
            'total_actions': total
        }

    def save_model(self, path):
        """Save student model weights."""
        self.student_agent.save_model(path)

    def load_model(self, path):
        """Load student model weights."""
        self.student_agent.load_model(path)
