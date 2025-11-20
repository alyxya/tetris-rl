"""Reward processing helpers shared across training scripts."""

from typing import Iterable, List

import numpy as np


N_ROWS = 20
N_COLS = 10
LINES_TO_REWARD = {0: 0.0, 1: 0.1, 2: 0.3, 3: 0.6, 4: 1.0}


def extract_line_clear_reward(prev_board, next_board, valid=True) -> float:
    """Return synthetic line-clear reward using board occupancy."""
    if not valid:
        return 0.0

    lines = count_lines_cleared(prev_board, next_board)
    return LINES_TO_REWARD.get(lines, 0.0)


def count_lines_cleared(prev_board, next_board, valid=True) -> int:
    """Infer cleared lines from change in locked block counts."""
    if not valid:
        return 0

    prev_locked = _count_locked(prev_board)
    next_locked = _count_locked(next_board)

    delta = prev_locked + 4 - next_locked
    if delta < 0:
        return 0

    lines = max(0, min(4, delta // N_COLS))
    return lines


def compute_discounted_returns(rewards: Iterable[float], gamma: float = 0.99) -> List[float]:
    """Compute discounted returns for a reward sequence."""
    returns: List[float] = [0.0] * len(rewards)
    future = 0.0
    for idx in reversed(range(len(rewards))):
        future = rewards[idx] + gamma * future
        returns[idx] = future
    return returns


def _count_locked(board) -> int:
    arr = np.asarray(board)
    if arr.shape != (N_ROWS, N_COLS):
        arr = arr.reshape(N_ROWS, N_COLS)
    return int(np.count_nonzero(arr == 1))
