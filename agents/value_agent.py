"""
Value-based CNN agent for Tetris.

Architecture:
- Dual-input CNN: processes board with piece as 0 (empty) and piece as 1 (filled)
- Shared CNN backbone extracts features from both representations
- Takes action as one-hot vector input
- Outputs scalar value prediction representing expected reward for (state, action) pair

Key differences from policy-based CNN:
- Input: board state + action (one-hot)
- Output: scalar value (predicted reward)
- Action selection: evaluate all actions, pick highest value
- Training: supervised learning on (state, action, reward) tuples
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from agents.base_agent import BaseTetrisAgent


class TetrisValueCNN(nn.Module):
    """
    Value-based CNN model for Tetris.

    Architecture:
    - Input: Two 20x10 boards (piece as empty, piece as filled) + action one-hot
    - Shared CNN: 3 conv layers with increasing channels
    - MLP: Combines board features + action features and outputs scalar value
    """

    def __init__(self, n_rows=20, n_cols=10, n_actions=7):
        super().__init__()

        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_actions = n_actions

        # Shared CNN backbone (processes both board representations)
        # Input: (batch, 1, 20, 10)
        # First conv uses no padding - we'll apply custom padding in forward_cnn
        # Padding: left=2, right=2, top=0, bottom=2
        # After padding: (20 + 0 + 2, 10 + 2 + 2) = (22, 14)
        # After conv (kernel=3, no padding): (22-3+1, 14-3+1) = (20, 12)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=0)  # Will be (32, 20, 12) after custom padding
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # -> (64, 20, 12)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1) # -> (64, 20, 12)

        # Calculate flattened size after conv layers
        # After custom padding + conv1: 20 rows, 12 cols
        self.conv_output_size = 64 * n_rows * (n_cols + 2)

        # MLP head (combines board features + action features)
        # Input: concatenated board features (64*20*10 * 2) + action one-hot (n_actions)
        self.fc1 = nn.Linear(self.conv_output_size * 2 + n_actions, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)  # Output scalar value

        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)

    def forward_cnn(self, x):
        """
        Process single board through CNN backbone.

        Args:
            x: (batch, 1, n_rows, n_cols) board tensor

        Returns:
            features: (batch, conv_output_size) flattened features
        """
        # Apply custom padding for first conv layer
        # Tetris has walls on left, right, and bottom (value=1), open top (no padding)
        # Padding format: (left, right, top, bottom)
        # Pad width 2 on left/right/bottom with 1s (walls), no top padding
        x = F.pad(x, (2, 2, 0, 2), mode='constant', value=1.0)  # Pad with 1s (walls)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        return x

    def forward(self, board_empty, board_filled, action_onehot):
        """
        Forward pass with dual board representation + action.

        Args:
            board_empty: (batch, 1, n_rows, n_cols) - piece treated as empty (0)
            board_filled: (batch, 1, n_rows, n_cols) - piece treated as filled (1)
            action_onehot: (batch, n_actions) - one-hot encoded action

        Returns:
            value: (batch, 1) predicted reward value for (state, action) pair
        """
        # Process both representations through shared CNN
        features_empty = self.forward_cnn(board_empty)
        features_filled = self.forward_cnn(board_filled)

        # Concatenate board features + action
        features = torch.cat([features_empty, features_filled, action_onehot], dim=1)

        # MLP head
        x = F.relu(self.fc1(features))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        value = self.fc3(x)

        return value


class ValueAgent(BaseTetrisAgent):
    """Value-based agent that learns to predict rewards for (state, action) pairs."""

    def __init__(self, n_rows=20, n_cols=10, device='cpu', model_path=None):
        """
        Initialize value-based agent.

        Args:
            n_rows: Number of rows in board
            n_cols: Number of columns in board
            device: 'cpu' or 'cuda'
            model_path: Path to load pretrained model (optional)
        """
        super().__init__(n_rows, n_cols)

        self.device = torch.device(device)
        self.model = TetrisValueCNN(n_rows, n_cols, n_actions=7).to(self.device)

        if model_path is not None:
            self.load_model(model_path)

        self.model.eval()  # Start in eval mode

    def prepare_board_inputs(self, obs):
        """
        Prepare dual board representation for CNN.

        Args:
            obs: flattened observation array

        Returns:
            board_empty: board with piece as 0 (empty)
            board_filled: board with piece as 1 (filled)
        """
        full_board, locked_board, active_piece = self.parse_observation(obs)

        # Board with piece as empty (only locked blocks)
        board_empty = locked_board.copy()

        # Board with piece as filled (locked blocks + piece)
        board_filled = locked_board.copy()
        board_filled[active_piece > 0] = 1

        # Convert to tensors (batch_size=1, channels=1, height, width)
        board_empty = torch.FloatTensor(board_empty).unsqueeze(0).unsqueeze(0).to(self.device)
        board_filled = torch.FloatTensor(board_filled).unsqueeze(0).unsqueeze(0).to(self.device)

        return board_empty, board_filled

    def get_action_values(self, obs):
        """
        Get value predictions for all actions.

        Args:
            obs: flattened observation array

        Returns:
            values: numpy array of shape (n_actions,) with predicted values
        """
        board_empty, board_filled = self.prepare_board_inputs(obs)

        with torch.no_grad():
            values = []

            # Evaluate each action
            for action in range(7):
                # Create one-hot encoding for action
                action_onehot = torch.zeros(1, 7).to(self.device)
                action_onehot[0, action] = 1.0

                # Get value prediction
                value = self.model(board_empty, board_filled, action_onehot)
                values.append(value.item())

        return np.array(values)

    def choose_action(self, obs, epsilon=0.0, deterministic=False):
        """
        Choose action by selecting highest predicted value.

        Args:
            obs: flattened observation array
            epsilon: probability of random action (for epsilon-greedy exploration)
            deterministic: if True, always choose best action (ignores epsilon)

        Returns:
            action: integer action (0-6)
        """
        # Epsilon-greedy exploration
        if not deterministic and np.random.random() < epsilon:
            return np.random.randint(0, 7)

        # Get value predictions for all actions
        values = self.get_action_values(obs)

        # Choose action with highest value
        action = np.argmax(values)

        return action

    def save_model(self, path):
        """Save model weights."""
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        """Load model weights."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        # Handle both full checkpoints and model-only weights
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)

        print(f"Model loaded from {path}")
