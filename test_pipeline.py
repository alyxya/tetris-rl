"""
Quick integration test for the training pipeline.
"""

import numpy as np
import torch
from pufferlib.ocean.tetris import tetris

from mixed_teacher_agent import MixedTeacherAgent
from reward_utils import compute_lines_cleared, compute_simple_reward
from model import ValueNetwork
from value_agent import ValueAgent

print("Testing Tetris Training Pipeline")
print("=" * 50)

# Test 1: Environment interaction with mixed teacher
print("\n1. Testing mixed teacher agent with environment...")
env = tetris.Tetris()
agent = MixedTeacherAgent()

num_test_episodes = 3
for ep in range(num_test_episodes):
    agent.reset()
    obs, _ = env.reset()
    done = False
    steps = 0
    total_reward = 0

    print(f"\n  Episode {ep+1}:")
    print(f"    Random prob: {agent.random_prob:.4f}")
    print(f"    Temperature: {agent.heuristic_temperature:.4f}")

    while not done and steps < 50:  # Limit steps for quick test
        # Extract single observation from batch
        obs_single = obs[0] if len(obs.shape) > 1 else obs

        action = agent.choose_action(obs_single)

        # Get state for reward computation
        _, locked, active = agent.parse_observation(obs_single)

        next_obs, _, terminated, truncated, _ = env.step([action])
        done = terminated[0] or truncated[0]

        # Compute reward
        if done:
            reward = 0.0
        else:
            next_obs_single = next_obs[0] if len(next_obs.shape) > 1 else next_obs
            _, next_locked, _ = agent.parse_observation(next_obs_single)
            lines = compute_lines_cleared(locked, active, next_locked)
            reward = compute_simple_reward(lines)

        total_reward += reward
        steps += 1
        obs = next_obs

    print(f"    Steps: {steps}, Total reward: {total_reward:.4f}")

print("\n✓ Mixed teacher test passed!")

# Test 2: Value network forward pass
print("\n2. Testing value network...")
device = torch.device('cpu')
model = ValueNetwork(n_rows=20, n_cols=10, n_actions=7).to(device)

# Create dummy batch
batch_size = 4
board_empty = torch.randn(batch_size, 1, 20, 10)
board_filled = torch.randn(batch_size, 1, 20, 10)

q_values = model(board_empty, board_filled)
print(f"  Input shape: {board_empty.shape}")
print(f"  Output shape: {q_values.shape}")
print(f"  Expected shape: ({batch_size}, 7)")
assert q_values.shape == (batch_size, 7), "Output shape mismatch!"

print("\n✓ Value network test passed!")

# Test 3: Value agent action selection
print("\n3. Testing value agent...")
value_agent = ValueAgent(device='cpu')
value_agent.model = model

obs, _ = env.reset()
obs_single = obs[0] if len(obs.shape) > 1 else obs
action = value_agent.choose_action(obs_single, epsilon=0.1)
print(f"  Selected action: {action}")
assert 0 <= action <= 6, "Invalid action!"

print("\n✓ Value agent test passed!")

# Test 4: Reward computation edge cases
print("\n4. Testing reward edge cases...")

# No active piece
locked = np.zeros((20, 10))
active = np.zeros((20, 10))  # No piece
next_locked = locked.copy()
lines = compute_lines_cleared(locked, active, next_locked)
reward = compute_simple_reward(lines)
print(f"  No active piece: {lines} lines, reward={reward}")
assert lines == 0 and reward == 0.0

# 2 lines cleared
locked = np.zeros((20, 10))
locked[18:20, :8] = 1  # 2 rows with 8 blocks each (16 blocks)
active = np.zeros((20, 10))
active[18:20, 8:10] = 1  # 4 blocks to complete 2 lines (total 20 blocks)
next_locked = np.zeros((20, 10))  # All cleared (0 blocks)
lines = compute_lines_cleared(locked, active, next_locked)
reward = compute_simple_reward(lines)
print(f"  2 lines cleared: {lines} lines, reward={reward}")
assert lines == 2 and reward == 0.3

# 3 lines cleared
locked = np.zeros((20, 10))
locked[17:20, :7] = 1  # 3 rows with 7 blocks each (21 blocks)
active = np.zeros((20, 10))
active[17:20, 7:10] = 1  # 9 blocks to complete 3 lines (total 30 blocks)
next_locked = np.zeros((20, 10))  # All cleared (0 blocks)
lines = compute_lines_cleared(locked, active, next_locked)
reward = compute_simple_reward(lines)
print(f"  3 lines cleared: {lines} lines, reward={reward}")
assert lines == 3 and reward == 0.6

print("\n✓ Reward edge cases test passed!")

print("\n" + "=" * 50)
print("All pipeline tests passed successfully!")
print("\nYou can now run:")
print("  1. python train_supervised_mixed.py --num-episodes 1000 --output models/supervised_value.pth --save-data data/supervised_dataset.pkl")
print("  2. python train_rl.py --num-episodes 1000 --init-model models/supervised_value.pth --output models/rl_value.pth")
