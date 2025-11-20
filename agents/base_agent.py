"""
Base agent class for Tetris agents.

Defines the interface and common functionality for all Tetris agents.
"""

import numpy as np
from abc import ABC, abstractmethod


class BaseTetrisAgent(ABC):
    """Abstract base class for Tetris agents."""

    def __init__(self, n_rows=20, n_cols=10):
        """Initialize base agent."""
        self.n_rows = n_rows
        self.n_cols = n_cols

    def parse_observation(self, obs):
        """
        Parse observation into board components.

        Args:
            obs: flattened observation array

        Returns:
            full_board: complete board (n_rows x n_cols) with 0=empty, 1=locked, 2=active
            locked_board: board with only locked blocks (1s become 1, rest become 0)
            active_piece: board with only active piece (2s become 1, rest become 0)
        """
        # Board: first n_rows * n_cols values
        full_board = obs[0:self.n_rows * self.n_cols].reshape(self.n_rows, self.n_cols)

        # Separate locked blocks and active piece
        locked_board = (full_board == 1).astype(float)
        active_piece = (full_board == 2).astype(float)

        return full_board, locked_board, active_piece

    def get_column_heights(self, board):
        """
        Get the height of each column (counting from bottom).

        Args:
            board: board array (n_rows x n_cols)

        Returns:
            heights: array of height for each column
        """
        heights = np.zeros(self.n_cols)
        for col in range(self.n_cols):
            for row in range(self.n_rows):
                if board[row, col] > 0:
                    heights[col] = self.n_rows - row
                    break
        return heights

    def count_holes(self, board):
        """
        Count holes (empty cells with filled cells above them).

        Args:
            board: board array (n_rows x n_cols)

        Returns:
            holes: number of holes
        """
        holes = 0
        for col in range(self.n_cols):
            found_block = False
            for row in range(self.n_rows):
                if board[row, col] > 0:
                    found_block = True
                elif found_block and board[row, col] == 0:
                    holes += 1
        return holes

    def count_complete_lines(self, board):
        """
        Count number of complete lines.

        Args:
            board: board array (n_rows x n_cols)

        Returns:
            complete_lines: number of complete lines
        """
        complete_lines = 0
        for row in range(self.n_rows):
            if np.all(board[row, :] > 0):
                complete_lines += 1
        return complete_lines

    def calculate_bumpiness(self, heights):
        """
        Calculate bumpiness (sum of height differences between adjacent columns).

        Args:
            heights: array of column heights

        Returns:
            bumpiness: sum of absolute height differences
        """
        bumpiness = 0
        for i in range(len(heights) - 1):
            bumpiness += abs(heights[i] - heights[i + 1])
        return bumpiness

    def reset(self):
        """Reset agent state. Override if agent has state to reset."""
        pass

    @abstractmethod
    def choose_action(self, obs):
        """
        Choose an action based on observation.

        Args:
            obs: flattened observation array

        Returns:
            action: integer action to take (0-6)
        """
        pass
