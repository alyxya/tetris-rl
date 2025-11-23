"""
Neural network architectures for Tetris.

Contains:
- Shared CNN backbone for processing dual-board representation
- ValueNetwork: outputs Q-values for each action
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedCNN(nn.Module):
    """Shared CNN backbone that processes a single 20x10 board."""

    def __init__(self, n_rows=20, n_cols=10):
        super().__init__()

        self.n_rows = n_rows
        self.n_cols = n_cols

        # CNN layers with wall padding strategy
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # Output size after padding and convolutions
        self.output_size = 64 * n_rows * (n_cols + 2)

    def forward(self, x):
        """Process a single board through CNN backbone."""
        # Custom padding: left/right/bottom padded with 1s to simulate walls
        x = F.pad(x, (2, 2, 0, 2), mode='constant', value=1.0)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x.view(x.size(0), -1)


class ValueNetwork(nn.Module):
    """Value network that outputs Q-values for each action."""

    def __init__(self, n_rows=20, n_cols=10, n_actions=7):
        super().__init__()

        self.n_actions = n_actions
        self.cnn = SharedCNN(n_rows, n_cols)

        # MLP head for dual board features -> Q-values
        feature_size = self.cnn.output_size * 2  # Concatenate empty + filled
        self.fc1 = nn.Linear(feature_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, n_actions)

        self.dropout = nn.Dropout(0.3)

    def forward(self, board_empty, board_filled):
        """Forward pass with dual board representation.

        Args:
            board_empty: Board with locked pieces only (Bx1x20x10)
            board_filled: Board with locked + active piece (Bx1x20x10)

        Returns:
            q_values: Q-value for each action (Bx7)
        """
        features_empty = self.cnn(board_empty)
        features_filled = self.cnn(board_filled)
        features = torch.cat([features_empty, features_filled], dim=1)

        x = F.relu(self.fc1(features))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        q_values = self.fc3(x)
        return q_values
