"""
CNN-based RL agent for Tetris.

Architecture:
- Dual-input CNN: processes board with piece as 0 (empty) and piece as 1 (filled)
- Shared CNN backbone extracts features from both representations
- MLP head combines features and outputs action probabilities
- Can be trained via supervised learning (imitation) then fine-tuned with RL
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base_agent import BaseTetrisAgent


class TetrisCNN(nn.Module):
    """
    CNN model for Tetris with dual board representation.

    Architecture:
    - Input: Two 20x10 boards (piece as empty, piece as filled)
    - Shared CNN: 3 conv layers with increasing channels
    - MLP: Combines features and outputs action probabilities
    """

    def __init__(self, n_rows=20, n_cols=10, n_actions=7):
        super().__init__()

        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_actions = n_actions

        # Shared CNN backbone (processes both board representations)
        # Input: (batch, 1, 20, 10)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # -> (32, 20, 10)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # -> (64, 20, 10)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1) # -> (64, 20, 10)

        # Calculate flattened size after conv layers
        self.conv_output_size = 64 * n_rows * n_cols

        # MLP head (combines features from both representations)
        # Input: concatenated features from both boards (64*20*10 * 2)
        self.fc1 = nn.Linear(self.conv_output_size * 2, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, n_actions)

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
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        return x

    def forward(self, board_empty, board_filled):
        """
        Forward pass with dual board representation.

        Args:
            board_empty: (batch, 1, n_rows, n_cols) - piece treated as empty (0)
            board_filled: (batch, 1, n_rows, n_cols) - piece treated as filled (1)

        Returns:
            logits: (batch, n_actions) action logits
        """
        # Process both representations through shared CNN
        features_empty = self.forward_cnn(board_empty)
        features_filled = self.forward_cnn(board_filled)

        # Concatenate features
        features = torch.cat([features_empty, features_filled], dim=1)

        # MLP head
        x = F.relu(self.fc1(features))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        logits = self.fc3(x)

        return logits


class CNNAgent(BaseTetrisAgent):
    """CNN-based agent that learns from experience."""

    def __init__(self, n_rows=20, n_cols=10, device='cpu', model_path=None):
        """
        Initialize CNN agent.

        Args:
            n_rows: Number of rows in board
            n_cols: Number of columns in board
            device: 'cpu' or 'cuda'
            model_path: Path to load pretrained model (optional)
        """
        super().__init__(n_rows, n_cols)

        self.device = torch.device(device)
        self.model = TetrisCNN(n_rows, n_cols).to(self.device)

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

    def choose_action(self, obs, temperature=1.0, deterministic=False):
        """
        Choose action using CNN policy.

        Args:
            obs: flattened observation array
            temperature: sampling temperature (higher = more exploration)
            deterministic: if True, choose argmax action

        Returns:
            action: integer action (0-6)
        """
        board_empty, board_filled = self.prepare_board_inputs(obs)

        with torch.no_grad():
            logits = self.model(board_empty, board_filled)

            if deterministic:
                # Choose best action
                action = logits.argmax(dim=1).item()
            else:
                # Sample from softmax distribution with temperature
                probs = F.softmax(logits / temperature, dim=1)
                action = torch.multinomial(probs, num_samples=1).item()

        return action

    def get_action_probs(self, obs, temperature=1.0):
        """
        Get action probability distribution.

        Args:
            obs: flattened observation array
            temperature: sampling temperature

        Returns:
            probs: numpy array of action probabilities
        """
        board_empty, board_filled = self.prepare_board_inputs(obs)

        with torch.no_grad():
            logits = self.model(board_empty, board_filled)
            probs = F.softmax(logits / temperature, dim=1)

        return probs.cpu().numpy()[0]

    def save_model(self, path):
        """Save model weights."""
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        """Load model weights."""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Model loaded from {path}")
