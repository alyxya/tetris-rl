"""Utility functions for computing rewards."""

import numpy as np


ACTION_NO_OP = 0
ACTION_LEFT = 1
ACTION_RIGHT = 2
ACTION_ROTATE = 3
ACTION_SOFT_DROP = 4
ACTION_HARD_DROP = 5
ACTION_HOLD = 6

ACTION_NAMES = {
    ACTION_NO_OP: "NO_OP",
    ACTION_LEFT: "LEFT",
    ACTION_RIGHT: "RIGHT",
    ACTION_ROTATE: "ROTATE",
    ACTION_SOFT_DROP: "SOFT_DROP",
    ACTION_HARD_DROP: "HARD_DROP",
    ACTION_HOLD: "HOLD",
}


def compute_lines_cleared(locked_board, active_piece, next_locked_board):
    """
    Calculate the number of lines cleared based on board state change.

    Args:
        locked_board: Previous locked board state
        active_piece: The active piece board
        next_locked_board: Next locked board state after the action

    Returns:
        lines_cleared: int, number of lines cleared (0-4)
    """
    piece_shape_positions = np.argwhere(active_piece > 0)
    if len(piece_shape_positions) == 0:
        return 0

    # Count blocks
    prev_locked_count = np.sum(locked_board > 0)
    next_locked_count = np.sum(next_locked_board > 0)
    piece_blocks = len(piece_shape_positions)

    # Calculate lines cleared
    # Delta = prev_count + piece_blocks - cleared_lines * 10
    # So: cleared_lines = (prev_count + piece_blocks - next_count) / 10
    delta = prev_locked_count + piece_blocks - next_locked_count
    lines_cleared = max(0, delta // 10)  # Integer division

    return lines_cleared


def compute_simple_reward(lines_cleared):
    """
    Compute simplified reward based only on lines cleared.

    Reward structure:
    - 1 line: 0.1
    - 2 lines: 0.3
    - 3 lines: 0.6
    - 4 lines: 1.0
    - Otherwise: 0.0

    Args:
        lines_cleared: int, number of lines cleared (0-4)

    Returns:
        reward: float
    """
    reward_map = {
        0: 0.0,
        1: 0.1,
        2: 0.3,
        3: 0.6,
        4: 1.0,
    }
    return reward_map.get(lines_cleared, 0.0)
