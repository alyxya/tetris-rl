"""Utility functions for computing rewards."""

import numpy as np
import heuristic


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


def compute_heuristic_normalized_reward(old_board, new_board, active_piece, lines_cleared):
    """
    Compute reward based on heuristic evaluation of placement quality.

    This provides dense feedback by comparing the chosen placement against all possible placements:
    1. Enumerate all possible placements (rotations × columns) for the current piece
    2. Score each placement using the heuristic function
    3. Find which placement matches the actual result (new_board)
    4. Normalize: reward = (chosen_score - mean_score) / std_score
    5. Add line clear bonus

    This solves the sparse reward problem by giving meaningful feedback for positioning moves,
    not just HARD_DROP.

    Args:
        old_board: 2D numpy array, previous board state (locked cells only)
        new_board: 2D numpy array, new board state after piece locked
        active_piece: 2D numpy array, active piece before locking
        lines_cleared: int, number of lines cleared (0-4)

    Returns:
        reward: float, normalized placement quality + line clear bonus
    """
    # Extract piece shape from active_piece board
    piece_positions = np.argwhere(active_piece > 0)
    if len(piece_positions) == 0:
        return 0.0  # No piece visible

    # Get bounding box of piece
    min_row, min_col = piece_positions.min(axis=0)
    max_row, max_col = piece_positions.max(axis=0)
    piece_shape = active_piece[min_row:max_row+1, min_col:max_col+1].copy()

    # Simple heuristic: negative sum of squared heights
    # Lower sum of squares = better placement
    # Line clears reduce heights, so they naturally score better
    weights = None  # Not used anymore, we'll compute directly

    # Enumerate all possible placements and find the one that matches new_board
    n_cols = old_board.shape[1]
    all_scores = []
    chosen_score = None

    for rot in range(4):
        rotated_shape = piece_shape.copy()
        for _ in range(rot):
            rotated_shape = heuristic.rotate_piece_cw(rotated_shape)

        piece_width = rotated_shape.shape[1]

        for col in range(n_cols - piece_width + 1):
            # Simulate placement and compute negative sum of squared heights
            simulated_board, lines, _ = heuristic.simulate_drop(old_board, rotated_shape, col)
            if simulated_board is not None:
                heights = heuristic.get_column_heights(simulated_board)
                sum_sq = sum(h * h for h in heights)

                # Adjust for line clears: each line cleared reduces height by 1 for all columns
                # Approximate the reduction in sum of squares
                if lines > 0:
                    # After clearing lines, all columns reduce by 'lines' rows
                    adjusted_heights = [max(0, h - lines) for h in heights]
                    adjusted_sum_sq = sum(h * h for h in adjusted_heights)
                    score = -adjusted_sum_sq
                else:
                    score = -sum_sq  # Negative because lower is better

                all_scores.append(score)

                # Check if this placement matches the actual result
                if chosen_score is None and np.array_equal(simulated_board, new_board):
                    chosen_score = score

    if len(all_scores) == 0:
        return 0.0  # No valid placements

    if chosen_score is None:
        # Couldn't match the placement - use a fallback approach
        # This can happen if lines were cleared (new_board has fewer filled cells)
        # In this case, just use average score
        chosen_score = np.mean(all_scores)

    # Normalize: center around -0.5 (shifted), scale to std=1
    # This makes positive rewards require better-than-average placements
    all_scores = np.array(all_scores)
    mean_score = np.mean(all_scores)
    std_score = np.std(all_scores)

    if std_score < 1e-6:
        # All placements have same score, this placement is average (shifted to -0.5)
        normalized_reward = -0.5
    else:
        normalized_reward = (chosen_score - mean_score) / std_score - 0.5

    # Add line clear bonus (5x scaling to make line clears highly rewarding)
    line_bonus_map = {0: 0.0, 1: 5.0, 2: 10.0, 3: 20.0, 4: 40.0}
    line_bonus = line_bonus_map.get(lines_cleared, 0.0)

    total_reward = normalized_reward + line_bonus

    # Clip reward to prevent destabilization
    return np.clip(total_reward, -20.0, +20.0)
