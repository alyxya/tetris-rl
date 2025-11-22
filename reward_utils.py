"""Utility functions for computing heuristic rewards."""

import numpy as np
import heuristic as heuristic_module
from heuristic import rotate_piece_cw


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


def extract_piece_shape_from_board(active_piece):
    """Extract piece shape from active piece board."""
    piece_positions = np.argwhere(active_piece > 0)
    if len(piece_positions) == 0:
        return None

    min_row = piece_positions[:, 0].min()
    max_row = piece_positions[:, 0].max()
    min_col = piece_positions[:, 1].min()
    max_col = piece_positions[:, 1].max()

    piece_shape = active_piece[min_row:max_row+1, min_col:max_col+1].copy()
    return piece_shape


def compute_all_heuristic_rewards(locked_board, active_piece, next_locked_board):
    """
    Compute heuristic-based rewards for all actions.

    Reward structure:
    1. Large immediate reward for line clears (if they happened)
    2. Otherwise, normalize action-specific heuristic scores (mean 0, std 0.01)
       based on placements across identity and single-rotation orientations.

    Returns:
        rewards_by_action: np.ndarray of shape (7,) with rewards for each action
        lines_cleared: int, number of lines cleared (0 if no line clears)
    """
    piece_shape = extract_piece_shape_from_board(active_piece)
    if piece_shape is None:
        return np.zeros(7, dtype=np.float32), 0

    # Check if lines were cleared (compare locked block counts)
    prev_locked_count = np.sum(locked_board > 0)
    next_locked_count = np.sum(next_locked_board > 0)

    # Calculate lines cleared (each cleared line removes 10 blocks, piece adds ~4 blocks)
    # Delta = prev_count + piece_blocks - cleared_lines * 10
    # So: cleared_lines = (prev_count + piece_blocks - next_count) / 10
    piece_blocks = np.sum(piece_shape > 0)
    delta = prev_locked_count + piece_blocks - next_locked_count
    lines_cleared = max(0, delta // 10)  # Integer division

    # Line clear rewards (scaled down by 10x from original)
    line_clear_rewards = {0: 0.0, 1: 1.0, 2: 3.0, 3: 6.0, 4: 10.0}
    line_reward = line_clear_rewards.get(lines_cleared, 0.0)

    # If lines were cleared, return the large reward for all actions
    if lines_cleared > 0:
        rewards = np.full(7, line_reward, dtype=np.float32)
        return rewards, lines_cleared

    # Otherwise, compute action-based heuristic scores
    # Get current piece position
    piece_positions = np.argwhere(active_piece > 0)
    current_left_col = piece_positions[:, 1].min()

    # Evaluate placements for identity and single clockwise rotation only
    n_cols = locked_board.shape[1]
    rotations_to_consider = (0, 1)
    placements = []  # List of (rotation, col, score)

    for rotation in rotations_to_consider:
        rotated_shape = piece_shape.copy()
        for _ in range(rotation):
            rotated_shape = rotate_piece_cw(rotated_shape)

        piece_width = rotated_shape.shape[1]

        for col in range(n_cols - piece_width + 1):
            score, _ = heuristic_module.evaluate_placement(
                locked_board, piece_shape, rotation, col
            )

            # Only consider valid placements
            if score > float('-inf'):
                placements.append((rotation, col, score))

    if len(placements) == 0:
        return np.zeros(7, dtype=np.float32), 0

    scores = np.array([p[2] for p in placements])
    cols = np.array([p[1] for p in placements])
    rotations = np.array([p[0] for p in placements])

    # Filter to only non-rotated pieces (rotation 0) for NO_OP and SOFT_DROP
    nonrotated_mask = rotations == 0
    nonrotated_scores = scores[nonrotated_mask]
    nonrotated_cols = cols[nonrotated_mask]

    # NO_OP: Weighted average skewed towards current column
    if len(nonrotated_scores) > 0:
        # Weight inversely proportional to distance from current column
        distances = np.abs(nonrotated_cols - current_left_col)
        # Use moderate exponential decay: weight = exp(-distance)
        weights_no_op = np.exp(-distances.astype(float))
        weights_no_op /= np.sum(weights_no_op)  # Normalize
        no_op_score = float(np.sum(weights_no_op * nonrotated_scores))
    else:
        no_op_score = 0.0

    # SOFT_DROP: Score at current column position
    current_col_mask = nonrotated_cols == current_left_col
    if current_col_mask.any():
        soft_drop_score = float(nonrotated_scores[current_col_mask][0])
    else:
        soft_drop_score = None

    left_mask = cols < current_left_col
    mean_left = float(np.mean(scores[left_mask])) if left_mask.any() else None

    right_mask = cols > current_left_col
    mean_right = float(np.mean(scores[right_mask])) if right_mask.any() else None

    rotation_mask = rotations > 0
    mean_rotation = float(np.mean(scores[rotation_mask])) if rotation_mask.any() else None

    raw_scores = {
        ACTION_NO_OP: no_op_score,
        ACTION_LEFT: mean_left,
        ACTION_RIGHT: mean_right,
        ACTION_ROTATE: mean_rotation,
        ACTION_SOFT_DROP: soft_drop_score,
    }

    # Only use non-None values for normalization
    valid_scores = {k: v for k, v in raw_scores.items() if v is not None}
    raw_values = np.array(list(valid_scores.values()), dtype=np.float32)
    raw_mean = float(raw_values.mean())
    raw_std = float(raw_values.std())
    if raw_std == 0.0:
        raw_std = 1.0

    normalized = (raw_values - raw_mean) / raw_std
    target_std = 0.01
    target_mean = 0.001
    normalized = normalized * target_std + target_mean

    # Map normalized values back to actions
    normalized_dict = {}
    for idx, act in enumerate(valid_scores.keys()):
        normalized_dict[act] = float(normalized[idx])

    # Assign rewards: use normalized value if available, otherwise use 0.0
    rewards_by_action = np.zeros(7, dtype=np.float32)
    for act in raw_scores.keys():
        if act in normalized_dict:
            rewards_by_action[act] = normalized_dict[act]
        # else: remains 0.0 from initialization

    return rewards_by_action, 0


def compute_heuristic_reward(locked_board, active_piece, next_locked_board, action):
    """
    Compute heuristic-based reward for the chosen action.

    This is a convenience wrapper around compute_all_heuristic_rewards.
    """
    rewards, _ = compute_all_heuristic_rewards(locked_board, active_piece, next_locked_board)
    return float(rewards[action])
