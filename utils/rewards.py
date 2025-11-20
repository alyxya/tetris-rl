"""Reward processing helpers shared across training scripts."""

from typing import Iterable, List

import numpy as np


LINES_TO_REWARD = {0: 0.0, 1: 0.1, 2: 0.3, 3: 0.6, 4: 1.0}


def extract_line_clear_reward(prev_board, next_board) -> float:
    """Return synthetic line-clear reward using board occupancy."""
    lines = count_lines_cleared(prev_board, next_board)
    return LINES_TO_REWARD.get(lines, 0.0)


def count_lines_cleared(prev_board, next_board) -> int:
    """Infer cleared lines from change in locked block counts."""
    prev_arr = _ensure_board(prev_board)
    next_arr = _ensure_board(next_board)

    prev_locked = int(np.count_nonzero(prev_arr == 1))
    next_locked = int(np.count_nonzero(next_arr == 1))
    cols = prev_arr.shape[1]

    delta = prev_locked + 4 - next_locked
    lines = max(0, min(4, delta // cols))
    return lines


def compute_discounted_returns(rewards: Iterable[float], gamma: float = 0.99) -> List[float]:
    """Compute discounted returns for a reward sequence."""
    returns: List[float] = [0.0] * len(rewards)
    future = 0.0
    for idx in reversed(range(len(rewards))):
        future = rewards[idx] + gamma * future
        returns[idx] = future
    return returns


def compute_penalty_returns(penalties: Iterable[float], growth: float = 1.005) -> List[float]:
    """
    Compute growing penalty returns for a penalty sequence.

    Unlike rewards which decay (gamma < 1), penalties grow (growth > 1)
    to make future movement penalties more costly than current ones.

    Args:
        penalties: Sequence of per-step penalties (0.01 for left/right/rotate, 0.0 otherwise)
        growth: Growth factor for future penalties (default: 1.005)

    Returns:
        List of cumulative penalty values with growth factor applied
    """
    returns: List[float] = [0.0] * len(penalties)
    future = 0.0
    for idx in reversed(range(len(penalties))):
        future = penalties[idx] + growth * future
        returns[idx] = future
    return returns


class LineClearPenaltyTracker:
    """Track translation/rotation actions to apply proximity penalties on clears."""

    def __init__(self, actions=None, penalty=0.08, decay=0.9):
        self.actions = set(actions if actions is not None else (1, 2, 3))
        self.penalty = float(penalty)
        self.decay = float(decay)
        self.pending: List[float] = []

    def reset(self):
        self.pending.clear()

    def step(self, action: int, line_reward: float) -> float:
        """Update tracker with the taken action and return shaped reward."""
        self.pending = [value * self.decay for value in self.pending if value * self.decay > 1e-6]

        if action in self.actions:
            self.pending.append(self.penalty)

        penalty_value = 0.0
        if line_reward > 0.0:
            penalty_value = sum(self.pending)
            self.pending.clear()

        shaped_reward = max(0.0, line_reward - penalty_value)
        return shaped_reward


def _ensure_board(board) -> np.ndarray:
    arr = np.asarray(board)
    if arr.ndim != 2:
        raise ValueError("Board arrays must be 2D (rows x cols)")
    return arr
