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
    2. Otherwise:
       - HARD_DROP: score from dropping at current position/rotation
       - NO_OP/SOFT_DROP: 0.0 reward
       - LEFT/RIGHT/ROTATE: weighted average of all placement scores,
         weighted by the proportion of that action needed to reach each placement

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
    # Get current piece position (assume rotation 0 since we can't easily infer it)
    piece_positions = np.argwhere(active_piece > 0)
    current_left_col = piece_positions[:, 1].min()
    current_rotation = 0  # Assumption: current piece is at rotation 0

    # Evaluate placements for all rotations (0-3)
    n_cols = locked_board.shape[1]
    rotations_to_consider = (0, 1, 2, 3)
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

    # Apply softmax to all placement scores for normalization
    scores = np.array([p[2] for p in placements], dtype=np.float32)
    # Subtract max for numerical stability
    scores_shifted = scores - np.max(scores)
    exp_scores = np.exp(scores_shifted)
    softmax_scores = exp_scores / np.sum(exp_scores)

    # Create normalized placements list
    normalized_placements = [(placements[i][0], placements[i][1], softmax_scores[i])
                             for i in range(len(placements))]

    # Initialize rewards
    rewards_by_action = np.zeros(7, dtype=np.float32)

    # NO_OP and SOFT_DROP: 0.0 reward
    rewards_by_action[ACTION_NO_OP] = 0.0
    rewards_by_action[ACTION_SOFT_DROP] = 0.0

    # HARD_DROP: softmax-normalized score at current position and rotation
    hard_drop_placements = [(r, c, s) for r, c, s in normalized_placements
                           if r == current_rotation and c == current_left_col]
    if hard_drop_placements:
        rewards_by_action[ACTION_HARD_DROP] = float(hard_drop_placements[0][2])
    else:
        rewards_by_action[ACTION_HARD_DROP] = 0.0

    # LEFT, RIGHT, ROTATE: weighted average based on action counts
    # For each placement, calculate the number of each action needed
    left_weighted_sum = 0.0
    left_total_weight = 0.0
    right_weighted_sum = 0.0
    right_total_weight = 0.0
    rotate_weighted_sum = 0.0
    rotate_total_weight = 0.0

    for rotation, col, norm_score in normalized_placements:
        # Calculate actions needed from current position
        num_rotates = rotation - current_rotation
        if num_rotates < 0:
            num_rotates += 4  # Wrap around

        num_horizontal = col - current_left_col
        num_left = max(0, -num_horizontal)
        num_right = max(0, num_horizontal)

        total_actions = num_left + num_right + num_rotates
        if total_actions == 0:
            continue  # This is the current position, skip

        # Weight for each action type is its proportion of total actions
        if num_left > 0:
            weight = num_left / total_actions
            left_weighted_sum += weight * norm_score
            left_total_weight += weight

        if num_right > 0:
            weight = num_right / total_actions
            right_weighted_sum += weight * norm_score
            right_total_weight += weight

        if num_rotates > 0:
            weight = num_rotates / total_actions
            rotate_weighted_sum += weight * norm_score
            rotate_total_weight += weight

    # Compute weighted averages
    if left_total_weight > 0:
        rewards_by_action[ACTION_LEFT] = left_weighted_sum / left_total_weight

    if right_total_weight > 0:
        rewards_by_action[ACTION_RIGHT] = right_weighted_sum / right_total_weight

    if rotate_total_weight > 0:
        rewards_by_action[ACTION_ROTATE] = rotate_weighted_sum / rotate_total_weight

    # HOLD action: always 0.0
    rewards_by_action[ACTION_HOLD] = 0.0

    return rewards_by_action, 0


def compute_heuristic_reward(locked_board, active_piece, next_locked_board, action):
    """
    Compute heuristic-based reward for the chosen action.

    This is a convenience wrapper around compute_all_heuristic_rewards.
    """
    rewards, _ = compute_all_heuristic_rewards(locked_board, active_piece, next_locked_board)
    return float(rewards[action])
