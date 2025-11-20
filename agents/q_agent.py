"""
Unified Q-value CNN agent for Tetris.

This model merges the original policy CNN and value CNN approaches:
- Dual-board CNN backbone identical to the old policy network
- Padding scheme matches the value network (explicit wall padding)
- Output: 7 Q-values (one per discrete Tetris action)

The same network can be used for:
1. Supervised imitation learning (treat outputs as logits for teacher labels)
2. Reinforcement learning fine-tuning (interpret outputs as Q-values)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from agents.base_agent import BaseTetrisAgent


class TetrisQNetwork(nn.Module):
    """Dual-board CNN that predicts a Q-value for each discrete action."""

    def __init__(self, n_rows=20, n_cols=10, n_actions=7):
        super().__init__()

        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_actions = n_actions

        # Shared CNN backbone (processes a single 20x10 board)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # Flattened conv output after padding -> conv1 -> conv2 -> conv3
        self.conv_output_size = 64 * n_rows * (n_cols + 2)

        # MLP head operating on concatenated board features
        self.fc1 = nn.Linear(self.conv_output_size * 2, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, n_actions)

        self.dropout = nn.Dropout(0.3)

    def forward_cnn(self, x):
        """Process a single board tensor through the CNN backbone."""
        # Custom padding to mimic Tetris walls: left/right/bottom padded with ones
        x = F.pad(x, (2, 2, 0, 2), mode='constant', value=1.0)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x.view(x.size(0), -1)

    def forward(self, board_empty, board_filled):
        """Forward pass with dual board representation."""
        features_empty = self.forward_cnn(board_empty)
        features_filled = self.forward_cnn(board_filled)
        features = torch.cat([features_empty, features_filled], dim=1)

        x = F.relu(self.fc1(features))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        q_values = self.fc3(x)
        return q_values


class QValueAgent(BaseTetrisAgent):
    """Agent wrapper that exposes the unified Q-network with helper methods."""

    def __init__(self, n_rows=20, n_cols=10, device='cpu', model_path=None):
        super().__init__(n_rows, n_cols)

        self.device = torch.device(device)
        self.model = TetrisQNetwork(n_rows, n_cols, n_actions=7).to(self.device)

        if model_path is not None:
            self.load_model(model_path)

        self.model.eval()

    def prepare_board_inputs(self, obs):
        """Convert flattened observation into dual board tensors."""
        full_board, locked_board, active_piece = self.parse_observation(obs)

        board_empty = locked_board.copy()
        board_filled = locked_board.copy()
        board_filled[active_piece > 0] = 1.0

        board_empty = torch.FloatTensor(board_empty).unsqueeze(0).unsqueeze(0).to(self.device)
        board_filled = torch.FloatTensor(board_filled).unsqueeze(0).unsqueeze(0).to(self.device)
        return board_empty, board_filled

    def get_action_values(self, obs):
        """Return numpy array of predicted Q-values for all actions."""
        board_empty, board_filled = self.prepare_board_inputs(obs)
        with torch.no_grad():
            values = self.model(board_empty, board_filled)
        return values.squeeze(0).cpu().numpy()

    def choose_action(self, obs, epsilon=0.0, deterministic=False):
        """Epsilon-greedy or greedy action selection using predicted Q-values."""
        if not deterministic and np.random.random() < epsilon:
            return np.random.randint(0, 7)

        values = self.get_action_values(obs)
        return int(np.argmax(values))

    def save_model(self, path):
        """Persist model weights to disk."""
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        """Load model weights from checkpoint or raw state_dict."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)

        print(f"Model loaded from {path}")
