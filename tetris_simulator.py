"""
Standalone Tetris simulation that doesn't depend on PufferLib environments.

Takes a grid and action, returns the new grid after simulating that action.
"""

import numpy as np


# Tetromino definitions (7 pieces, each with 4 rotations)
# Shape: [piece_type][rotation][row][col]
PIECES = {
    'I': [
        [[0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]],
        [[0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0]],
        [[0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0]],
        [[0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0]],
    ],
    'O': [
        [[0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        [[0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        [[0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        [[0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    ],
    'T': [
        [[0, 1, 0, 0], [1, 1, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        [[0, 1, 0, 0], [0, 1, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]],
        [[0, 0, 0, 0], [1, 1, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]],
        [[0, 1, 0, 0], [1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]],
    ],
    'S': [
        [[0, 1, 1, 0], [1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        [[0, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 0], [0, 0, 0, 0]],
        [[0, 0, 0, 0], [0, 1, 1, 0], [1, 1, 0, 0], [0, 0, 0, 0]],
        [[1, 0, 0, 0], [1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]],
    ],
    'Z': [
        [[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        [[0, 0, 1, 0], [0, 1, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]],
        [[0, 0, 0, 0], [1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 0, 0]],
        [[0, 1, 0, 0], [1, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]],
    ],
    'J': [
        [[1, 0, 0, 0], [1, 1, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        [[0, 1, 1, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]],
        [[0, 0, 0, 0], [1, 1, 1, 0], [0, 0, 1, 0], [0, 0, 0, 0]],
        [[0, 1, 0, 0], [0, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0]],
    ],
    'L': [
        [[0, 0, 1, 0], [1, 1, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        [[0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 1, 0], [0, 0, 0, 0]],
        [[0, 0, 0, 0], [1, 1, 1, 0], [1, 0, 0, 0], [0, 0, 0, 0]],
        [[1, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]],
    ],
}

PIECE_TYPES = ['I', 'O', 'T', 'S', 'Z', 'J', 'L']


class TetrisState:
    """Represents a Tetris game state with grid and current piece."""

    def __init__(self, grid: np.ndarray, piece_type: int = 0, rotation: int = 0, x: int = 3, y: int = 0):
        """
        Args:
            grid: 20x10 numpy array (0 = empty, >0 = filled)
            piece_type: 0-6 (I, O, T, S, Z, J, L)
            rotation: 0-3
            x: column position of piece (left edge of 4x4 bounding box)
            y: row position of piece (top edge of 4x4 bounding box)
        """
        self.grid = grid.copy()
        self.piece_type = piece_type
        self.rotation = rotation
        self.x = x
        self.y = y

    def get_piece_shape(self) -> np.ndarray:
        """Get the current piece shape."""
        return np.array(PIECES[PIECE_TYPES[self.piece_type]][self.rotation])

    def can_place(self, x: int, y: int, rotation: int = None) -> bool:
        """Check if piece can be placed at position."""
        if rotation is None:
            rotation = self.rotation

        piece_shape = np.array(PIECES[PIECE_TYPES[self.piece_type]][rotation])

        for py in range(4):
            for px in range(4):
                if piece_shape[py, px]:
                    grid_x = x + px
                    grid_y = y + py

                    # Check bounds
                    if grid_x < 0 or grid_x >= 10 or grid_y >= 20:
                        return False

                    # Check collision with existing blocks (but allow negative y for spawning)
                    if grid_y >= 0 and self.grid[grid_y, grid_x] != 0:
                        return False

        return True

    def place_piece(self) -> np.ndarray:
        """Place the current piece on the grid and return new grid."""
        new_grid = self.grid.copy()
        piece_shape = self.get_piece_shape()

        for py in range(4):
            for px in range(4):
                if piece_shape[py, px]:
                    grid_x = self.x + px
                    grid_y = self.y + py
                    if 0 <= grid_y < 20 and 0 <= grid_x < 10:
                        new_grid[grid_y, grid_x] = self.piece_type + 1

        return new_grid

    def clear_lines(self, grid: np.ndarray) -> np.ndarray:
        """Clear full lines and return new grid."""
        new_grid = []
        for row in grid:
            if not np.all(row != 0):  # If row is not full
                new_grid.append(row)

        # Add empty rows at top
        lines_cleared = 20 - len(new_grid)
        for _ in range(lines_cleared):
            new_grid.insert(0, np.zeros(10))

        return np.array(new_grid)


def simulate_step(grid: np.ndarray, action: int, piece_type: int = 0, rotation: int = 0, x: int = 3, y: int = 0) -> np.ndarray:
    """
    Simulate one Tetris action and return the resulting grid.

    Actions:
        0 - NO_OP (do nothing)
        1 - LEFT (move piece left)
        2 - RIGHT (move piece right)
        3 - ROTATE (rotate piece clockwise)
        4 - SOFT_DROP (move piece down one step)
        5 - HARD_DROP (drop piece to bottom and lock)
        6 - HOLD (treated as NO_OP)

    Args:
        grid: 20x10 numpy array representing the board
        action: int from 0-6
        piece_type: Current piece type (0-6 for I, O, T, S, Z, J, L)
        rotation: Current rotation (0-3)
        x: Current x position (column)
        y: Current y position (row)

    Returns:
        np.ndarray: The new 20x10 grid after the action
    """
    state = TetrisState(grid, piece_type, rotation, x, y)

    # Action 0: NO_OP
    if action == 0:
        return grid.copy()

    # Action 1: LEFT
    elif action == 1:
        if state.can_place(state.x - 1, state.y):
            state.x -= 1
        return grid.copy()

    # Action 2: RIGHT
    elif action == 2:
        if state.can_place(state.x + 1, state.y):
            state.x += 1
        return grid.copy()

    # Action 3: ROTATE
    elif action == 3:
        new_rotation = (state.rotation + 1) % 4
        if state.can_place(state.x, state.y, new_rotation):
            state.rotation = new_rotation
        return grid.copy()

    # Action 4: SOFT_DROP
    elif action == 4:
        if state.can_place(state.x, state.y + 1):
            state.y += 1
        else:
            # Lock piece
            new_grid = state.place_piece()
            return state.clear_lines(new_grid)
        return grid.copy()

    # Action 5: HARD_DROP
    elif action == 5:
        # Drop until can't drop anymore
        while state.can_place(state.x, state.y + 1):
            state.y += 1

        # Lock piece
        new_grid = state.place_piece()
        return state.clear_lines(new_grid)

    # Action 6: HOLD (treat as NO_OP)
    elif action == 6:
        return grid.copy()

    return grid.copy()


# Example usage
if __name__ == "__main__":
    print("Testing standalone Tetris simulation")
    print("=" * 50)

    # Create an empty grid
    grid = np.zeros((20, 10), dtype=int)

    # Add some blocks at the bottom for testing
    grid[19, :] = 1  # Bottom row filled
    grid[18, :5] = 1  # Part of second-to-bottom row

    print("Initial grid (last 3 rows):")
    print(grid[-3:])
    print(f"Filled cells: {np.sum(grid != 0)}")

    # Test different actions with a piece at position (3, 0)
    action_names = ["NO_OP", "LEFT", "RIGHT", "ROTATE", "SOFT_DROP", "HARD_DROP", "HOLD"]

    print("\nTesting actions (with I piece at x=3, y=0, rotation=0):")
    print("-" * 50)

    for action in range(7):
        new_grid = simulate_step(grid, action, piece_type=0, rotation=0, x=3, y=0)
        print(f"Action {action} ({action_names[action]:10s}): filled cells = {np.sum(new_grid != 0)}")

    # Test HARD_DROP more carefully
    print("\nDetailed HARD_DROP test:")
    new_grid = simulate_step(grid, action=5, piece_type=0, rotation=0, x=3, y=0)
    print("Grid after HARD_DROP (last 5 rows):")
    print(new_grid[-5:])
    print(f"Filled cells: {np.sum(new_grid != 0)}")

    print("\n" + "=" * 50)
    print("Simulation test complete!")
