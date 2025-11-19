"""
Simple heuristic-based Tetris agent (no rotation version).

The agent:
1. Extracts the current piece from the board (cells with value 2)
2. For each horizontal position, simulates dropping the piece straight down
3. Prioritizes moves that clear lines (most lines = best)
4. If tied or no lines, uses heuristics: minimize height, holes, bumpiness
5. Moves to target position and drops

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
from pufferlib.ocean.tetris import tetris


class HeuristicAgent:
    def __init__(self, n_rows=20, n_cols=10):
        """Initialize heuristic agent."""
        self.n_rows = n_rows
        self.n_cols = n_cols

        # Heuristic weights (for when no lines are cleared)
        self.weights = {
            'height': -0.51,
            'lines': 0.76,
            'holes': -0.36,
            'bumpiness': -0.18,
        }

        # State tracking
        self.target_column = None
        self.ready_to_drop = False

    def parse_observation(self, obs):
        """Parse observation into board parts."""
        # Board: first n_rows * n_cols values
        # Values: 0 = empty, 1 = locked block, 2 = active piece
        full_board = obs[0:self.n_rows * self.n_cols].reshape(self.n_rows, self.n_cols)

        # Separate locked blocks and active piece
        locked_board = (full_board == 1).astype(float)
        active_piece = (full_board == 2).astype(float)

        return full_board, locked_board, active_piece

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

    def get_column_heights(self, board):
        """Get the height of each column (counting from bottom)."""
        heights = np.zeros(self.n_cols)
        for col in range(self.n_cols):
            for row in range(self.n_rows):
                if board[row, col] > 0:
                    heights[col] = self.n_rows - row
                    break
        return heights

    def count_holes(self, board):
        """Count holes (empty cells with filled cells above them)."""
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
        """Count number of complete lines that would be cleared."""
        complete_lines = 0
        for row in range(self.n_rows):
            if np.all(board[row, :] > 0):
                complete_lines += 1
        return complete_lines

    def calculate_bumpiness(self, heights):
        """Calculate bumpiness (sum of height differences between adjacent columns)."""
        bumpiness = 0
        for i in range(len(heights) - 1):
            bumpiness += abs(heights[i] - heights[i + 1])
        return bumpiness

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

    def find_best_drop_column(self, locked_board, piece_shape, current_piece_cols):
        """
        Find the best column to drop the piece.

        Returns:
            best_col: best column to place piece's left edge
            best_score: evaluation score
        """
        if piece_shape is None:
            return None, float('-inf')

        piece_width = piece_shape.shape[1]

        best_col = None
        best_score = float('-inf')
        best_lines_cleared = 0

        # Try each possible column
        for col in range(self.n_cols - piece_width + 1):
            new_board, lines_cleared = self.simulate_drop(locked_board, piece_shape, col)

            if new_board is None:
                continue

            # Prioritize line clears
            if lines_cleared > best_lines_cleared:
                best_lines_cleared = lines_cleared
                best_col = col
                best_score = lines_cleared * 1000  # High priority
            elif lines_cleared == best_lines_cleared:
                # Use heuristic for ties
                score = self.evaluate_board(new_board)
                if lines_cleared > 0:
                    score += lines_cleared * 1000

                if score > best_score:
                    best_score = score
                    best_col = col

        return best_col, best_score

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

        # If no piece visible, no-op
        if piece_shape is None:
            self.target_column = None
            self.ready_to_drop = False
            return 0

        # Get current piece leftmost column
        current_left_col = piece_cols.min()

        # If we don't have a target, find the best one
        if self.target_column is None:
            self.target_column, _ = self.find_best_drop_column(locked_board, piece_shape, piece_cols)
            self.ready_to_drop = False

            if self.target_column is None:
                # No valid placement found, just drop
                return 4

        # Move towards target column
        if current_left_col > self.target_column:
            return 1  # Move left
        elif current_left_col < self.target_column:
            return 2  # Move right
        else:
            # We're in position, drop!
            self.target_column = None
            self.ready_to_drop = False
            return 4  # Soft drop (let it fall)


def main():
    """Demo the heuristic agent."""
    env = tetris.Tetris()
    agent = HeuristicAgent()

    obs, info = env.reset()
    done = False
    total_reward = 0
    steps = 0
    pieces_placed = 0

    print("Running heuristic agent (no rotation)...")
    print("The agent evaluates horizontal placements and chooses the best one.")

    while not done:
        action = agent.choose_action(obs[0])
        obs, reward, terminated, truncated, info = env.step([action])

        # frame = env.render()

        # Uncomment to see the board
        # if steps % 10 == 0:
        #     board = obs[0, :200].reshape(20, 10).astype(int)
        #     print(f"\nStep {steps}:")
        #     print(board)

        total_reward += reward[0]
        steps += 1
        done = terminated[0] or truncated[0]

        if steps % 100 == 0:
            print(f"Step {steps}, Total reward: {total_reward:.2f}")

    print(f"\nEpisode finished after {steps} steps")
    print(f"Total reward: {total_reward:.2f}")
    env.close()


if __name__ == "__main__":
    main()
