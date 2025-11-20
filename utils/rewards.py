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


def _ensure_board(board) -> np.ndarray:
    arr = np.asarray(board)
    if arr.ndim != 2:
        raise ValueError("Board arrays must be 2D (rows x cols)")
    return arr
