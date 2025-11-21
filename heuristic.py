"""
Shared heuristic reward function for Tetris.

This function evaluates state-action pairs and is used by:
1. HeuristicAgent - for action selection
2. Value-based training - as the reward signal for supervised/RL training

The heuristic primarily focuses on line clears with minor adjustments for
board quality (holes, height, bumpiness).
"""

import numpy as np


def get_column_heights(board):
    """Get height of each column (distance from bottom to highest filled cell)."""
    heights = np.zeros(board.shape[1])
    for col in range(board.shape[1]):
        column = board[:, col]
        filled_rows = np.where(column > 0)[0]
        if len(filled_rows) > 0:
            heights[col] = board.shape[0] - filled_rows[0]
    return heights


def count_holes(board):
    """Count holes (empty cells with filled cells above them)."""
    holes = 0
    for col in range(board.shape[1]):
        column = board[:, col]
        filled_rows = np.where(column > 0)[0]
        if len(filled_rows) > 0:
            # Count empty cells below the highest filled cell
            top_filled = filled_rows[0]
            holes += np.sum(column[top_filled:] == 0)
    return holes


def count_complete_lines(board):
    """Count number of complete (fully filled) rows."""
    return np.sum(np.all(board > 0, axis=1))


def calculate_bumpiness(heights):
    """Calculate bumpiness (sum of height differences between adjacent columns)."""
    if len(heights) < 2:
        return 0
    return np.sum(np.abs(np.diff(heights)))


def rotate_piece_cw(piece_shape):
    """Rotate piece shape 90 degrees clockwise."""
    return np.flip(piece_shape.T, axis=1)


def simulate_drop(board, piece_shape, target_col):
    """
    Simulate dropping piece at target column.

    Args:
        board: Current board state (locked pieces only)
        piece_shape: 2D array of piece to drop
        target_col: Left column position to drop at

    Returns:
        new_board: Resulting board after drop (or None if invalid)
        lines_cleared: Number of complete lines in result
    """
    n_rows, n_cols = board.shape
    piece_height, piece_width = piece_shape.shape

    # Check if piece fits horizontally
    if target_col + piece_width > n_cols or target_col < 0:
        return None, 0

    new_board = board.copy()

    # Find landing position
    for row in range(n_rows):
        collision = False

        # Check if piece would collide at this row
        if row + piece_height > n_rows:
            collision = True
        else:
            for pr in range(piece_height):
                for pc in range(piece_width):
                    if piece_shape[pr, pc] > 0:
                        if new_board[row + pr, target_col + pc] > 0:
                            collision = True
                            break
                if collision:
                    break

        if collision:
            # Place at previous row
            if row == 0:
                return None, 0  # Can't place

            place_row = row - 1
            for pr in range(piece_height):
                for pc in range(piece_width):
                    if piece_shape[pr, pc] > 0:
                        new_board[place_row + pr, target_col + pc] = 1

            lines_cleared = count_complete_lines(new_board)
            return new_board, lines_cleared

    # No collision, place at bottom
    place_row = n_rows - piece_height
    for pr in range(piece_height):
        for pc in range(piece_width):
            if piece_shape[pr, pc] > 0:
                new_board[place_row + pr, target_col + pc] = 1

    lines_cleared = count_complete_lines(new_board)
    return new_board, lines_cleared


def evaluate_placement(board, piece_shape, rotation, target_col, weights=None):
    """
    Evaluate a specific placement (rotation + column) using heuristics.

    Args:
        board: Current locked board state (20x10)
        piece_shape: Current piece shape (2D array)
        rotation: Number of clockwise rotations to apply (0-3)
        target_col: Left column to drop at
        weights: Dict with keys 'lines', 'height', 'holes', 'bumpiness'

    Returns:
        score: Heuristic score (higher is better)
        lines_cleared: Number of lines that would be cleared
    """
    if weights is None:
        # Default weights emphasize line clears
        weights = {
            'lines': 10.0,      # Strong positive for line clears
            'height': -0.51,    # Penalty for aggregate height
            'holes': -0.36,     # Penalty for holes
            'bumpiness': -0.18, # Penalty for uneven surface
        }

    # Apply rotations
    rotated_shape = piece_shape.copy()
    for _ in range(rotation % 4):
        rotated_shape = rotate_piece_cw(rotated_shape)

    # Simulate the drop
    new_board, lines_cleared = simulate_drop(board, rotated_shape, target_col)

    if new_board is None:
        return float('-inf'), 0

    # Calculate board metrics
    heights = get_column_heights(new_board)
    aggregate_height = np.sum(heights)
    holes = count_holes(new_board)
    bumpiness = calculate_bumpiness(heights)

    # Compute weighted score
    score = (
        weights['lines'] * lines_cleared +
        weights['height'] * aggregate_height +
        weights['holes'] * holes +
        weights['bumpiness'] * bumpiness
    )

    return score, lines_cleared


def find_best_placement(board, piece_shape, weights=None):
    """
    Find the best rotation and column to place the piece.

    Args:
        board: Current locked board state
        piece_shape: Current piece shape
        weights: Heuristic weights dict

    Returns:
        best_rotation: Best number of CW rotations (0-3)
        best_col: Best left column position
        best_score: Best heuristic score
    """
    n_cols = board.shape[1]
    best_options = []  # Store all (rotation, col) pairs with best score
    best_score = float('-inf')

    # Try all rotations and columns
    for rotation in range(4):
        rotated_shape = piece_shape.copy()
        for _ in range(rotation):
            rotated_shape = rotate_piece_cw(rotated_shape)

        piece_width = rotated_shape.shape[1]

        for col in range(n_cols - piece_width + 1):
            score, _ = evaluate_placement(board, piece_shape, rotation, col, weights)

            if score > best_score:
                best_score = score
                best_options = [(rotation, col)]
            elif score == best_score:
                best_options.append((rotation, col))

    # Deterministically choose among tied options based on board state hash
    if best_options:
        # First, prefer placements with fewer rotations
        min_rotations = min(rot for rot, col in best_options)
        fewest_rotation_options = [(rot, col) for rot, col in best_options if rot == min_rotations]

        # Then use hash-based selection among those with fewest rotations
        board_hash = hash(board.tobytes())
        idx = board_hash % len(fewest_rotation_options)
        best_rotation, best_col = fewest_rotation_options[idx]
    else:
        best_rotation, best_col = 0, None

    return best_rotation, best_col, best_score
