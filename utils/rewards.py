"""
Reward processing helpers shared across training scripts.
"""

from typing import Iterable, List


def extract_line_clear_reward(raw_reward: float) -> float:
    """Convert raw PufferLib reward into line-clear shaped reward."""
    step_reward = round(float(raw_reward), 2)
    if step_reward >= 0.09:
        return round(step_reward / 0.1) * 0.1
    return 0.0


def compute_discounted_returns(rewards: Iterable[float], gamma: float = 0.99) -> List[float]:
    """Compute discounted returns for a reward sequence."""
    returns: List[float] = [0.0] * len(rewards)
    future = 0.0
    for idx in reversed(range(len(rewards))):
        future = rewards[idx] + gamma * future
        returns[idx] = future
    return returns
