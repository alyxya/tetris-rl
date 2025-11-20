"""
Heuristic-based Tetris agent with rotation support.

The agent:
1. Extracts the current piece from the board (cells with value 2)
2. For each rotation (0-3) and horizontal position, simulates dropping the piece
3. Prioritizes moves that clear lines (most lines = best)
4. If tied or no lines, uses heuristics: minimize height, holes, bumpiness
5. Rotates piece to target rotation, moves to target column, then drops

Actions (verified empirically):
0 = no-op (piece doesn't move)
1 = left (move piece left one column)
2 = right (move piece right one column)
3 = rotate clockwise
4 = down (soft drop - move piece down one row)
5 = rotate counter-clockwise
6 = no-op (same as action 0)
"""

import numpy as np
import sys
import os

from agents.base_agent import BaseTetrisAgent


class HeuristicAgent(BaseTetrisAgent):
    """Heuristic agent that evaluates placements with rotation."""

    def __init__(self, n_rows=20, n_cols=10):
        """
        Initialize heuristic agent.

        Args:
            n_rows: Number of rows in board
            n_cols: Number of columns in board
        """
        super().__init__(n_rows, n_cols)

        # Heuristic weights (for when no lines are cleared)
        self.weights = {
            'height': -0.51,
            'lines': 0.76,
            'holes': -0.36,
            'bumpiness': -0.18,
        }

        # State tracking
        self.target_column = None
        self.target_rotation = None  # 0-3 (number of CW rotations from current)
        self.current_rotations = 0   # Track how many rotations we've done

    def reset(self):
        """Reset agent state for new episode."""
        self.target_column = None
        self.target_rotation = None
        self.current_rotations = 0

    def extract_piece_info(self, active_piece):
        """
        Extract piece bounding box and shape.

        Returns:
            piece_positions: (row, col) positions of piece blocks
            piece_cols: column positions of piece
            piece_shape: minimal bounding box containing the piece
        """
        piece_positions = np.argwhere(active_piece > 0)

        if len(piece_positions) == 0:
            return None, None, None

        # Get bounding box
        min_row, min_col = piece_positions.min(axis=0)
        max_row, max_col = piece_positions.max(axis=0)

        # Extract shape within bounding box
        piece_shape = active_piece[min_row:max_row+1, min_col:max_col+1]

        # Get column range
        piece_cols = piece_positions[:, 1]

        return piece_positions, piece_cols, piece_shape

    def rotate_piece_cw(self, piece_shape):
        """
        Rotate piece shape 90 degrees clockwise.

        Args:
            piece_shape: 2D array representing piece

        Returns:
            rotated_shape: piece rotated 90 degrees clockwise
        """
        # Rotate 90 degrees clockwise = transpose then flip horizontally
        return np.flip(piece_shape.T, axis=1)

    def simulate_drop(self, locked_board, piece_shape, target_col):
        """
        Simulate dropping piece straight down at target_col.

        Returns:
            new_board: resulting board (or None if invalid)
            lines_cleared: number of lines that would be cleared
        """
        piece_height, piece_width = piece_shape.shape

        # Check if piece fits horizontally
        if target_col + piece_width > self.n_cols or target_col < 0:
            return None, 0

        # Start from top, find where piece lands
        new_board = locked_board.copy()

        for row in range(self.n_rows):
            # Check collision
            collision = False
            if row + piece_height > self.n_rows:
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

                # Count complete lines
                lines_cleared = self.count_complete_lines(new_board)

                return new_board, lines_cleared

        # No collision, place at bottom
        place_row = self.n_rows - piece_height
        for pr in range(piece_height):
            for pc in range(piece_width):
                if piece_shape[pr, pc] > 0:
                    new_board[place_row + pr, target_col + pc] = 1

        lines_cleared = self.count_complete_lines(new_board)
        return new_board, lines_cleared

    def evaluate_board(self, board):
        """Evaluate board state using heuristics."""
        heights = self.get_column_heights(board)
        aggregate_height = np.sum(heights)
        complete_lines = self.count_complete_lines(board)
        holes = self.count_holes(board)
        bumpiness = self.calculate_bumpiness(heights)

        score = (
            self.weights['height'] * aggregate_height +
            self.weights['lines'] * complete_lines +
            self.weights['holes'] * holes +
            self.weights['bumpiness'] * bumpiness
        )

        return score

    def find_best_placement(self, locked_board, piece_shape):
        """
        Find the best rotation and column to drop the piece.

        Returns:
            best_rotation: number of CW rotations from current (0-3)
            best_col: best column to place piece's left edge
            best_score: evaluation score
        """
        if piece_shape is None:
            return 0, None, float('-inf')

        best_rotation = 0
        best_col = None
        best_score = float('-inf')
        best_lines_cleared = 0

        # Try all 4 rotations
        current_shape = piece_shape.copy()

        for rotation in range(4):
            piece_width = current_shape.shape[1]

            # Try each possible column for this rotation
            for col in range(self.n_cols - piece_width + 1):
                new_board, lines_cleared = self.simulate_drop(locked_board, current_shape, col)

                if new_board is None:
                    continue

                # Prioritize line clears
                if lines_cleared > best_lines_cleared:
                    best_lines_cleared = lines_cleared
                    best_rotation = rotation
                    best_col = col
                    best_score = lines_cleared * 1000  # High priority
                elif lines_cleared == best_lines_cleared:
                    # Use heuristic for ties
                    score = self.evaluate_board(new_board)
                    if lines_cleared > 0:
                        score += lines_cleared * 1000

                    if score > best_score:
                        best_score = score
                        best_rotation = rotation
                        best_col = col

            # Rotate for next iteration
            current_shape = self.rotate_piece_cw(current_shape)

        return best_rotation, best_col, best_score

    def choose_action(self, obs):
        """
        Choose action to move piece to best position and drop.

        Actions:
        0 = no-op
        1 = left
        2 = right
        3 = rotate cw
        4 = down
        5 = rotate ccw
        6 = no-op
        """
        full_board, locked_board, active_piece = self.parse_observation(obs)

        # Extract current piece info
        piece_positions, piece_cols, piece_shape = self.extract_piece_info(active_piece)

        # If no piece visible, no-op and reset state
        if piece_shape is None:
            self.target_column = None
            self.target_rotation = None
            self.current_rotations = 0
            return 0

        # Get current piece leftmost column
        current_left_col = piece_cols.min()

        # If we don't have a target, find the best placement
        if self.target_column is None or self.target_rotation is None:
            self.target_rotation, self.target_column, _ = self.find_best_placement(locked_board, piece_shape)
            self.current_rotations = 0

            if self.target_column is None:
                # No valid placement found, just drop
                return 4

        # Step 1: Rotate to target orientation first
        if self.current_rotations < self.target_rotation:
            self.current_rotations += 1
            return 3  # Rotate clockwise

        # Step 2: Move to target column
        if current_left_col > self.target_column:
            return 1  # Move left
        elif current_left_col < self.target_column:
            return 2  # Move right
        else:
            # Step 3: We're in position, drop!
            self.target_column = None
            self.target_rotation = None
            self.current_rotations = 0
            return 4  # Soft drop (let it fall)
