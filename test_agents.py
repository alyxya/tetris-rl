"""Quick test of all agent types."""

from pufferlib.ocean.tetris import tetris
from heuristic_agent import HeuristicAgent
import time

# Test heuristic agent for a few steps
print("Testing HeuristicAgent...")
env = tetris.Tetris(seed=int(time.time() * 1e6))
agent = HeuristicAgent()

obs, _ = env.reset()
print(f"Initial obs shape: {obs.shape}")

for step in range(10):
    obs_single = obs[0] if len(obs.shape) > 1 else obs
    action = agent.choose_action(obs_single)
    next_obs, reward, terminated, truncated, _ = env.step([action])
    print(f"Step {step}: action={action}, reward={reward[0]:.2f}, done={terminated[0] or truncated[0]}")

    if terminated[0] or truncated[0]:
        print("Episode terminated")
        break

    obs = next_obs

print("\nHeuristicAgent test completed successfully!")
