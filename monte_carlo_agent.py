"""
Monte Carlo lookahead agent for Tetris.

Uses Monte Carlo rollouts to find action sequences that lead to high Q-values.
"""

import numpy as np
import torch
from value_agent import ValueAgent


class MonteCarloAgent(ValueAgent):
    """Value agent with Monte Carlo lookahead for action selection."""

    def __init__(self, device='cpu', num_rollouts=20, rollout_depth=10, temperature=0.1, epsilon=0.1):
        """
        Args:
            device: torch device
            num_rollouts: number of Monte Carlo rollouts to perform
            rollout_depth: maximum depth of each rollout
            temperature: temperature for action sampling (lower = more greedy)
            epsilon: probability of random action in rollout
        """
        super().__init__(device=device)
        self.num_rollouts = num_rollouts
        self.rollout_depth = rollout_depth
        self.temperature = temperature
        self.epsilon = epsilon

        # Action space for rollouts (exclude NO_OP and HARD_DROP)
        self.rollout_actions = [1, 2, 3, 4]  # LEFT, RIGHT, ROTATE, SOFT_DROP

    def simulate_action(self, locked, active, action):
        """
        Simulate a single action on the board.

        Args:
            locked: 20x10 locked board (empty cells)
            active: 20x10 active piece board
            action: action to take (1=LEFT, 2=RIGHT, 3=ROTATE, 4=SOFT_DROP)

        Returns:
            new_locked: updated locked board (if piece locked)
            new_active: updated active piece board
            piece_locked: whether piece locked after this action
        """
        # Find active piece position and shape
        piece_positions = np.argwhere(active > 0)
        if len(piece_positions) == 0:
            return locked.copy(), active.copy(), False

        min_row, min_col = piece_positions.min(axis=0)
        max_row, max_col = piece_positions.max(axis=0)
        piece_shape = active[min_row:max_row+1, min_col:max_col+1].copy()

        new_active = np.zeros_like(active)
        new_locked = locked.copy()

        if action == 1:  # LEFT
            # Try to move left
            new_col = max(0, min_col - 1)
            # Check collision
            collision = False
            for pr in range(piece_shape.shape[0]):
                for pc in range(piece_shape.shape[1]):
                    if piece_shape[pr, pc] > 0:
                        board_r = min_row + pr
                        board_c = new_col + pc
                        if board_c < 0 or locked[board_r, board_c] > 0:
                            collision = True
                            break
                if collision:
                    break

            if not collision:
                min_col = new_col

        elif action == 2:  # RIGHT
            # Try to move right
            new_col = min(10 - piece_shape.shape[1], min_col + 1)
            # Check collision
            collision = False
            for pr in range(piece_shape.shape[0]):
                for pc in range(piece_shape.shape[1]):
                    if piece_shape[pr, pc] > 0:
                        board_r = min_row + pr
                        board_c = new_col + pc
                        if board_c >= 10 or locked[board_r, board_c] > 0:
                            collision = True
                            break
                if collision:
                    break

            if not collision:
                min_col = new_col

        elif action == 3:  # ROTATE (clockwise, fixed top-left)
            # Rotate piece 90 degrees clockwise
            rotated = np.flip(piece_shape.T, axis=1)

            # Check if rotated piece fits at current position
            collision = False
            if min_row + rotated.shape[0] > 20 or min_col + rotated.shape[1] > 10:
                collision = True
            else:
                for pr in range(rotated.shape[0]):
                    for pc in range(rotated.shape[1]):
                        if rotated[pr, pc] > 0:
                            board_r = min_row + pr
                            board_c = min_col + pc
                            if board_r >= 20 or board_c >= 10 or locked[board_r, board_c] > 0:
                                collision = True
                                break
                    if collision:
                        break

            if not collision:
                piece_shape = rotated

        elif action == 4:  # SOFT_DROP
            # Move down one row
            new_row = min_row + 1
            # Check collision
            collision = False
            if new_row + piece_shape.shape[0] > 20:
                collision = True
            else:
                for pr in range(piece_shape.shape[0]):
                    for pc in range(piece_shape.shape[1]):
                        if piece_shape[pr, pc] > 0:
                            board_r = new_row + pr
                            board_c = min_col + pc
                            if locked[board_r, board_c] > 0:
                                collision = True
                                break
                    if collision:
                        break

            if collision:
                # Piece locks
                for pr in range(piece_shape.shape[0]):
                    for pc in range(piece_shape.shape[1]):
                        if piece_shape[pr, pc] > 0:
                            new_locked[min_row + pr, min_col + pc] = 1
                return new_locked, new_active, True
            else:
                min_row = new_row

        # Place piece at new position
        for pr in range(piece_shape.shape[0]):
            for pc in range(piece_shape.shape[1]):
                if piece_shape[pr, pc] > 0:
                    new_active[min_row + pr, min_col + pc] = 1

        return new_locked, new_active, False

    def rollout(self, locked, active):
        """
        Perform one Monte Carlo rollout.

        Returns:
            action_sequence: list of actions taken
            avg_q_value: average Q-value encountered during rollout
        """
        current_locked = locked.copy()
        current_active = active.copy()
        action_sequence = []
        q_values_seen = []

        for _ in range(self.rollout_depth):
            # Prepare observation for Q-value computation
            board_empty = current_locked.copy()
            board_filled = current_locked.copy()
            board_filled[current_active > 0] = 1.0

            # Get Q-values
            state_empty = torch.FloatTensor(board_empty).unsqueeze(0).unsqueeze(0).to(self.device)
            state_filled = torch.FloatTensor(board_filled).unsqueeze(0).unsqueeze(0).to(self.device)

            with torch.no_grad():
                q_values = self.model(state_empty, state_filled)[0].cpu().numpy()

            # Track Q-values (only for rollout actions)
            rollout_q_values = q_values[self.rollout_actions]
            q_values_seen.append(np.max(rollout_q_values))

            # Sample action with temperature and epsilon-greedy
            if np.random.random() < self.epsilon:
                # Random action
                action = np.random.choice(self.rollout_actions)
            else:
                # Temperature-based sampling on rollout actions
                rollout_q = q_values[self.rollout_actions]
                exp_q = np.exp(rollout_q / self.temperature)
                probs = exp_q / np.sum(exp_q)
                action = np.random.choice(self.rollout_actions, p=probs)

            action_sequence.append(action)

            # Simulate action
            new_locked, new_active, piece_locked = self.simulate_action(
                current_locked, current_active, action
            )

            # If piece locked, stop rollout
            if piece_locked:
                # Get final Q-value after piece locked
                final_board_empty = new_locked.copy()
                final_board_filled = new_locked.copy()  # No active piece after locking

                final_state_empty = torch.FloatTensor(final_board_empty).unsqueeze(0).unsqueeze(0).to(self.device)
                final_state_filled = torch.FloatTensor(final_board_filled).unsqueeze(0).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    final_q_values = self.model(final_state_empty, final_state_filled)[0].cpu().numpy()

                q_values_seen.append(np.max(final_q_values))
                break

            current_locked = new_locked
            current_active = new_active

        # Compute average Q-value
        avg_q_value = np.mean(q_values_seen) if len(q_values_seen) > 0 else float('-inf')
        return action_sequence, avg_q_value

    def choose_action(self, obs, epsilon=0.0, temperature=0.0):
        """
        Choose action using Monte Carlo lookahead.

        Args:
            obs: observation from environment
            epsilon: ignored (we use internal epsilon for rollouts)
            temperature: ignored (we use internal temperature for rollouts)

        Returns:
            action: best action to take
        """
        _, locked, active = self.parse_observation(obs)

        # Perform multiple rollouts
        best_action = None
        best_avg_q = float('-inf')

        for _ in range(self.num_rollouts):
            action_sequence, avg_q_value = self.rollout(locked, active)

            # If this rollout has better average Q-value, use its first action
            if avg_q_value > best_avg_q and len(action_sequence) > 0:
                best_avg_q = avg_q_value
                best_action = action_sequence[0]

        # Fallback to random rollout action if no rollouts succeeded
        if best_action is None:
            best_action = np.random.choice(self.rollout_actions)

        return best_action
