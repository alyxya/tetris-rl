"""
Heuristic-based Tetris agent.

Uses the shared heuristic reward function to select actions.
Plans multi-step sequences (rotate + move + drop) to reach best placement.
"""

import numpy as np
from base_agent import BaseAgent
import heuristic

# Action constants
ACTION_NOOP = 0
ACTION_LEFT = 1
ACTION_RIGHT = 2
ACTION_ROTATE = 3
ACTION_SOFT_DROP = 4
ACTION_HARD_DROP = 5
ACTION_HOLD = 6


class HeuristicAgent(BaseAgent):
    """Agent that selects actions based on heuristic evaluation."""

    def __init__(self, n_rows=20, n_cols=10, weights=None):
        """
        Initialize heuristic agent.

        Args:
            n_rows: Board height
            n_cols: Board width
            weights: Heuristic weights dict (or None for defaults)
        """
        super().__init__(n_rows, n_cols)
        # Use original weight tuned for aggregate height
        # Since we now use placement_height, keep the gentler -0.51 penalty
        if weights is None:
            weights = {
                'lines': 10.0,
                'height': -0.51,    # Original weight (gentler than default -1.0)
                'holes': -0.36,
                'bumpiness': -0.18,
            }
        self.weights = weights

        # State tracking for multi-step plans
        self.target_column = None
        self.target_rotation = None
        self.current_rotations = 0

    def reset(self):
        """Reset agent state for new episode."""
        self.target_column = None
        self.target_rotation = None
        self.current_rotations = 0

    def choose_action(self, obs):
        """
        Choose action to move piece to best placement and drop.

        Multi-step planning:
        1. Find best placement (rotation + column)
        2. Rotate to target orientation
        3. Move to target column
        4. Drop piece
        """
        _, locked_board, active_piece = self.parse_observation(obs)
        piece_shape = self.extract_piece_shape(active_piece)

        # If no piece visible, reset and wait
        if piece_shape is None:
            self.target_column = None
            self.target_rotation = None
            self.current_rotations = 0
            return ACTION_NOOP

        # Find best placement if we don't have a target
        if self.target_column is None or self.target_rotation is None:
            self.target_rotation, self.target_column, _ = heuristic.find_best_placement(
                locked_board, piece_shape, self.weights
            )
            self.current_rotations = 0

            if self.target_column is None:
                # No valid placement found, just drop
                return ACTION_HARD_DROP

        # Step 1: Rotate to target orientation
        if self.current_rotations < self.target_rotation:
            self.current_rotations += 1
            return ACTION_ROTATE

        # Step 2: Move to target column
        # Only calculate current position after rotation is complete
        piece_positions = np.argwhere(active_piece > 0)
        current_left_col = piece_positions[:, 1].min()

        if current_left_col > self.target_column:
            return ACTION_LEFT
        elif current_left_col < self.target_column:
            return ACTION_RIGHT

        # Step 3: Drop piece
        self.target_column = None
        self.target_rotation = None
        self.current_rotations = 0
        return ACTION_HARD_DROP
