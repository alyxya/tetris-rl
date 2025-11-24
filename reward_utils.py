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


def compute_column_heights(board):
    """
    Compute the height of the tallest filled cell in each column.

    Args:
        board: 2D numpy array (rows, cols) where >0 means filled

    Returns:
        heights: list of int, height of each column (0 if empty)
    """
    n_rows, n_cols = board.shape
    heights = []

    for col in range(n_cols):
        # Find highest filled row in this column
        filled_rows = np.where(board[:, col] > 0)[0]
        if len(filled_rows) > 0:
            # Height is distance from bottom (row index from bottom)
            height = n_rows - filled_rows[0]
        else:
            height = 0
        heights.append(height)

    return heights


def compute_holes(board):
    """
    Count holes (empty cells with filled cells above them) in each column.

    Args:
        board: 2D numpy array (rows, cols) where >0 means filled

    Returns:
        total_holes: int, total number of holes across all columns
    """
    n_rows, n_cols = board.shape
    total_holes = 0

    for col in range(n_cols):
        column = board[:, col]
        # Find first filled cell from top
        filled_indices = np.where(column > 0)[0]
        if len(filled_indices) > 0:
            first_filled = filled_indices[0]
            # Count empty cells below first filled cell
            holes_in_col = np.sum(column[first_filled:] == 0)
            total_holes += holes_in_col

    return total_holes


def compute_bumpiness(heights):
    """
    Compute bumpiness (sum of absolute height differences between adjacent columns).

    Args:
        heights: list of int, column heights

    Returns:
        bumpiness: int, sum of |heights[i] - heights[i+1]|
    """
    if len(heights) <= 1:
        return 0

    bumpiness = 0
    for i in range(len(heights) - 1):
        bumpiness += abs(heights[i] - heights[i+1])

    return bumpiness


def compute_shaped_reward(old_board, new_board, lines_cleared):
    """
    Compute shaped reward using only squared height penalty.

    Reward = -sum(new_heights²) + sum(old_heights²)

    Simple and clean: any action that reduces sum of squared heights gets positive reward.
    Clearing lines reduces height, so gets rewarded implicitly.
    Stacking high gets heavily penalized due to quadratic growth.

    Args:
        old_board: 2D numpy array, previous board state (locked cells only)
        new_board: 2D numpy array, new board state (locked cells only)
        lines_cleared: int, number of lines cleared (0-4) - not used

    Returns:
        reward: float (negative change in sum of squared heights)
    """
    # Compute heights for both boards
    old_heights = compute_column_heights(old_board)
    new_heights = compute_column_heights(new_board)

    # Compute sum of squares
    old_sum_sq = sum(h * h for h in old_heights)
    new_sum_sq = sum(h * h for h in new_heights)

    # Reward = reduction in sum of squares (negative change)
    # If new_sum_sq < old_sum_sq (height decreased), reward is positive
    # If new_sum_sq > old_sum_sq (height increased), reward is negative
    # Scaled by 0.1 to make values more manageable for learning
    return (old_sum_sq - new_sum_sq) * 0.1
