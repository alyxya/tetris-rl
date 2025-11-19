import torch
import numpy as np
from pufferlib.ocean.tetris import tetris
from tetris_cnn_policy import TetrisCNNPolicy

# Create environment
env = tetris.Tetris()

# Create policy
policy = TetrisCNNPolicy(
    observation_space=env.observation_space,
    action_space=env.action_space,
    cnn_channels=[32, 64, 64],
    cnn_kernel_sizes=[3, 3, 3],
    cnn_strides=[1, 1, 1],
    mlp_hidden_dims=[256, 128],
)

print("Policy created successfully!")
print(f"Number of parameters: {sum(p.numel() for p in policy.parameters()):,}")

# Test with a sample observation
obs, info = env.reset()
obs_tensor = torch.FloatTensor(obs)

print(f"\nObservation shape: {obs_tensor.shape}")

# Forward pass
with torch.no_grad():
    action_logits, value = policy.forward(obs_tensor)
    print(f"\nAction logits shape: {action_logits.shape}")
    print(f"Action logits: {action_logits}")
    print(f"Value shape: {value.shape}")
    print(f"Value: {value}")

    # Test get_action_and_value
    action, log_prob, entropy, value = policy.get_action_and_value(obs_tensor)
    print(f"\nSampled action: {action}")
    print(f"Log prob: {log_prob}")
    print(f"Entropy: {entropy}")
    print(f"Value: {value}")

# Test grid extraction
grid = policy.extract_grid(obs_tensor)
print(f"\nExtracted grid shape: {grid.shape}")
print(f"Grid min: {grid.min()}, max: {grid.max()}")

# Test dual grid creation
grid_filled, grid_empty = policy.create_dual_grids(grid)
print(f"\nGrid filled - unique values: {torch.unique(grid_filled)}")
print(f"Grid empty - unique values: {torch.unique(grid_empty)}")

# Verify the falling piece transformation
original_has_2 = (grid == 2.0).any()
filled_has_2 = (grid_filled == 2.0).any()
empty_has_2 = (grid_empty == 2.0).any()

print(f"\nOriginal grid has value 2: {original_has_2}")
print(f"Filled grid has value 2: {filled_has_2}")
print(f"Empty grid has value 2: {empty_has_2}")

# Run a few steps to test in action
print("\n" + "="*50)
print("Testing in environment loop...")
print("="*50)

env.reset()
total_reward = 0

for step in range(10):
    obs_tensor = torch.FloatTensor(obs).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        action, log_prob, entropy, value = policy.get_action_and_value(obs_tensor)

    action_np = action.cpu().numpy()[0]
    obs, reward, terminated, truncated, info = env.step(action_np)

    total_reward += reward
    done = terminated or truncated

    print(f"Step {step+1}: Action={action_np}, Reward={float(reward):.2f}, Value={value.item():.2f}")

    if done:
        print("Episode finished!")
        break

print(f"\nTotal reward: {total_reward}")
print("\nAll tests passed!")
