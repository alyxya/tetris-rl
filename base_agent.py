"""
Abstract base class for Tetris agents.
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseAgent(ABC):
    """Abstract base class defining the agent interface."""

    def __init__(self, n_rows=20, n_cols=10):
        self.n_rows = n_rows
        self.n_cols = n_cols

    @abstractmethod
    def choose_action(self, obs):
        """
        Choose an action given the current observation.

        Args:
            obs: Flattened observation array from environment
                 (0=empty, 1=locked, 2=active piece)

        Returns:
            action: Integer action (0-6)
        """
        pass

    def reset(self):
        """Reset agent state for new episode (optional)."""
        pass

    def parse_observation(self, obs):
        """
        Parse flattened observation into board components.

        Args:
            obs: Flattened observation array (at least n_rows * n_cols elements)
                 First n_rows * n_cols elements are the board

        Returns:
            full_board: Full board state (n_rows x n_cols)
            locked_board: Board with only locked pieces (n_rows x n_cols)
            active_piece: Board with only active piece (n_rows x n_cols)
        """
        board_size = self.n_rows * self.n_cols
        board_obs = obs[:board_size]
        full_board = board_obs.reshape(self.n_rows, self.n_cols)
        locked_board = (full_board == 1).astype(np.float32)
        active_piece = (full_board == 2).astype(np.float32)
        return full_board, locked_board, active_piece

    def prepare_board_inputs(self, obs):
        """
        Prepare dual board representation for neural networks.

        Args:
            obs: Flattened observation array

        Returns:
            board_empty: Board with locked pieces only (n_rows x n_cols)
            board_filled: Board with locked + active piece (n_rows x n_cols)
        """
        _, locked_board, active_piece = self.parse_observation(obs)

        board_empty = locked_board.copy()
        board_filled = locked_board.copy()
        board_filled[active_piece > 0] = 1.0

        return board_empty, board_filled

    def extract_piece_shape(self, active_piece):
        """
        Extract piece shape from active piece board.

        Args:
            active_piece: Board with only active piece

        Returns:
            piece_shape: Minimal bounding box containing piece (or None if no piece)
        """
        piece_positions = np.argwhere(active_piece > 0)

        if len(piece_positions) == 0:
            return None

        # Get bounding box
        min_row, min_col = piece_positions.min(axis=0)
        max_row, max_col = piece_positions.max(axis=0)

        # Extract shape within bounding box
        piece_shape = active_piece[min_row:max_row+1, min_col:max_col+1]
        return piece_shape
