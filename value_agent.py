"""
Value-based Tetris agent.

Uses a neural network to output Q-values for each action and selects greedily.
"""

import numpy as np
import torch
import os
from base_agent import BaseAgent
from model import ValueNetwork


class ValueAgent(BaseAgent):
    """Agent that selects actions based on learned Q-values."""

    def __init__(self, n_rows=20, n_cols=10, device='cpu', model_path=None):
        """
        Initialize value agent.

        Args:
            n_rows: Board height
            n_cols: Board width
            device: Device for model ('cpu' or 'cuda')
            model_path: Path to load model weights from
        """
        super().__init__(n_rows, n_cols)

        self.device = torch.device(device)
        self.model = ValueNetwork(n_rows, n_cols, n_actions=7).to(self.device)
        self.model.eval()

        if model_path is not None:
            self.load_model(model_path)

    def choose_action(self, obs, epsilon=0.0, temperature=0.001):
        """
        Choose action with epsilon-greedy and Boltzmann exploration.

        Args:
            obs: Flattened observation
            epsilon: Probability of random exploration
            temperature: Temperature for Boltzmann sampling (0 = greedy, higher = more exploration)

        Returns:
            action: Selected action (0-6)
        """
        # Epsilon-greedy exploration
        if np.random.random() < epsilon:
            return np.random.randint(0, 7)

        # Get Q-values
        q_values = self.get_q_values(obs)

        # Temperature = 0 means greedy action selection
        if temperature == 0.0:
            return int(np.argmax(q_values))

        # Boltzmann (softmax) action selection with temperature
        # Shift Q-values for numerical stability
        q_shifted = q_values - np.max(q_values)
        exp_q = np.exp(q_shifted / temperature)
        probs = exp_q / np.sum(exp_q)

        action = np.random.choice(len(probs), p=probs)
        return int(action)

    def get_q_values(self, obs):
        """
        Get Q-values for all actions.

        Args:
            obs: Flattened observation

        Returns:
            q_values: Numpy array of Q-values (7,)
        """
        board_empty, board_filled = self.prepare_board_inputs(obs)

        # Convert to tensors
        board_empty = torch.FloatTensor(board_empty).unsqueeze(0).unsqueeze(0).to(self.device)
        board_filled = torch.FloatTensor(board_filled).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_values = self.model(board_empty, board_filled)

        return q_values.squeeze(0).cpu().numpy()

    def save_model(self, path):
        """Save model weights to disk."""
        output_dir = os.path.dirname(path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"Value model saved to {path}")

    def load_model(self, path):
        """Load model weights from checkpoint or raw state_dict."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)

        print(f"Value model loaded from {path}")
