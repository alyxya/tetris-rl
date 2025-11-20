"""
Policy-based Tetris agent.

Uses a neural network to output action logits and samples from the distribution.
"""

import numpy as np
import torch
import torch.nn.functional as F
from base_agent import BaseAgent
from model import PolicyNetwork


class PolicyAgent(BaseAgent):
    """Agent that samples actions from a learned policy network."""

    def __init__(self, n_rows=20, n_cols=10, device='cpu', model_path=None):
        """
        Initialize policy agent.

        Args:
            n_rows: Board height
            n_cols: Board width
            device: Device for model ('cpu' or 'cuda')
            model_path: Path to load model weights from
        """
        super().__init__(n_rows, n_cols)

        self.device = torch.device(device)
        self.model = PolicyNetwork(n_rows, n_cols, n_actions=7).to(self.device)
        self.model.eval()

        if model_path is not None:
            self.load_model(model_path)

    def choose_action(self, obs, deterministic=False, temperature=1.0):
        """
        Choose action by sampling from policy distribution.

        Args:
            obs: Flattened observation
            deterministic: If True, take argmax instead of sampling
            temperature: Temperature for softmax (lower = more deterministic)

        Returns:
            action: Sampled action (0-6)
        """
        logits = self.get_logits(obs)

        if deterministic:
            return int(np.argmax(logits))

        # Sample from softmax distribution with temperature
        probs = F.softmax(torch.tensor(logits) / temperature, dim=0).numpy()
        action = np.random.choice(len(probs), p=probs)
        return int(action)

    def get_logits(self, obs):
        """
        Get action logits from policy network.

        Args:
            obs: Flattened observation

        Returns:
            logits: Numpy array of action logits (7,)
        """
        board_empty, board_filled = self.prepare_board_inputs(obs)

        # Convert to tensors
        board_empty = torch.FloatTensor(board_empty).unsqueeze(0).unsqueeze(0).to(self.device)
        board_filled = torch.FloatTensor(board_filled).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(board_empty, board_filled)

        return logits.squeeze(0).cpu().numpy()

    def save_model(self, path):
        """Save model weights to disk."""
        torch.save(self.model.state_dict(), path)
        print(f"Policy model saved to {path}")

    def load_model(self, path):
        """Load model weights from checkpoint or raw state_dict."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)

        print(f"Policy model loaded from {path}")
