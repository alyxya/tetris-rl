"""
Analyze reward values from rollouts to understand RL training signals.

This script collects rewards from different agent types and prints statistics
to help sanity check the reward structure before RL training.
"""

import argparse
import numpy as np
import time
from pufferlib.ocean.tetris import tetris
from heuristic_agent import HeuristicAgent
from hybrid_agent import HybridAgent
from reward_utils import compute_lines_cleared, compute_simple_reward


def collect_episode_rewards(env, agent, verbose=False):
    """
    Collect detailed reward information for a single episode.

    Returns:
        episode_rewards: List of (step, action, env_reward, simple_reward) tuples
        total_steps: Total steps in episode
    """
    obs, _ = env.reset(seed=int(time.time() * 1e6))
    agent.reset()

    episode_data = []
    done = False
    steps = 0

    while not done:
        obs_single = obs[0] if len(obs.shape) > 1 else obs
        action = agent.choose_action(obs_single)

        # Parse current state
        _, locked, active = agent.parse_observation(obs_single)

        # Get environment reward and take step
        next_obs, env_reward, terminated, truncated, _ = env.step([action])
        done = terminated[0] or truncated[0]
        env_reward = env_reward[0] if hasattr(env_reward, '__getitem__') else env_reward

        # Parse next state
        next_obs_single = next_obs[0] if len(next_obs.shape) > 1 else next_obs
        _, next_locked, _ = agent.parse_observation(next_obs_single)

        # Compute simple reward (line clears only)
        lines_cleared = compute_lines_cleared(locked, active, next_locked)
        simple_reward = compute_simple_reward(lines_cleared)

        episode_data.append((steps, action, env_reward, simple_reward))

        if verbose and steps % 500 == 0:
            print(f"  Step {steps}: env_reward={env_reward:.3f}, simple_reward={simple_reward:.3f}")

        steps += 1
        obs = next_obs

    return episode_data, steps


def analyze_rewards(episodes_data):
    """Print reward statistics from collected episodes."""

    all_env_rewards = []
    all_simple_rewards = []
    all_steps = []

    for episode_data, total_steps in episodes_data:
        env_rewards = [r[2] for r in episode_data]
        simple_rewards = [r[3] for r in episode_data]

        all_env_rewards.extend(env_rewards)
        all_simple_rewards.extend(simple_rewards)
        all_steps.append(total_steps)

    print("\n" + "=" * 70)
    print("REWARD ANALYSIS")
    print("=" * 70)

    # Environment rewards
    print("\nEnvironment Rewards (from PufferLib Tetris env):")
    print(f"  Mean:    {np.mean(all_env_rewards):.6f}")
    print(f"  Std:     {np.std(all_env_rewards):.6f}")
    print(f"  Min:     {np.min(all_env_rewards):.6f}")
    print(f"  Max:     {np.max(all_env_rewards):.6f}")
    print(f"  Median:  {np.median(all_env_rewards):.6f}")

    # Count non-zero rewards
    nonzero_env = [r for r in all_env_rewards if r != 0]
    print(f"  Non-zero: {len(nonzero_env)}/{len(all_env_rewards)} ({100*len(nonzero_env)/len(all_env_rewards):.1f}%)")
    if nonzero_env:
        print(f"  Non-zero mean: {np.mean(nonzero_env):.6f}")

    # Simple rewards
    print("\nSimple Rewards (line clears only: 0.1/0.3/0.6/1.0):")
    print(f"  Mean:    {np.mean(all_simple_rewards):.6f}")
    print(f"  Std:     {np.std(all_simple_rewards):.6f}")
    print(f"  Min:     {np.min(all_simple_rewards):.6f}")
    print(f"  Max:     {np.max(all_simple_rewards):.6f}")
    print(f"  Median:  {np.median(all_simple_rewards):.6f}")

    # Episode statistics
    print("\nEpisode Statistics:")
    print(f"  Total episodes:  {len(all_steps)}")
    print(f"  Mean steps:      {np.mean(all_steps):.1f}")
    print(f"  Std steps:       {np.std(all_steps):.1f}")
    print(f"  Min steps:       {np.min(all_steps)}")
    print(f"  Max steps:       {np.max(all_steps)}")

    # Cumulative rewards
    print("\nCumulative Rewards per Episode:")
    for i, (episode_data, total_steps) in enumerate(episodes_data):
        env_sum = sum(r[2] for r in episode_data)
        simple_sum = sum(r[3] for r in episode_data)
        print(f"  Episode {i+1}: env={env_sum:.2f}, simple={simple_sum:.2f}, steps={total_steps}")

    # Reward distribution
    print("\nEnvironment Reward Distribution:")
    unique_rewards, counts = np.unique(all_env_rewards, return_counts=True)
    for reward, count in sorted(zip(unique_rewards, counts), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {reward:.4f}: {count} times ({100*count/len(all_env_rewards):.1f}%)")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Analyze reward values from rollouts")
    parser.add_argument('--agent', type=str, default='hybrid',
                        choices=['heuristic', 'hybrid'],
                        help='Agent type to analyze')
    parser.add_argument('--heuristic-prob', type=float, default=0.5,
                        help='Probability of heuristic agent in hybrid (default: 0.5)')
    parser.add_argument('--random-prob', type=float, default=0.5,
                        help='Probability of random agent in hybrid (default: 0.5)')
    parser.add_argument('--episodes', type=int, default=5,
                        help='Number of episodes to analyze')
    parser.add_argument('--verbose', action='store_true',
                        help='Print rewards every 500 steps')

    args = parser.parse_args()

    # Create environment (seed passed to reset())
    env = tetris.Tetris()

    # Create agent
    if args.agent == 'heuristic':
        agent = HeuristicAgent()
        print(f"Analyzing HeuristicAgent for {args.episodes} episodes...")
    else:
        # Create hybrid agent with heuristic + random
        heuristic_agent = HeuristicAgent()
        agents = [heuristic_agent, 'random']
        probs = [args.heuristic_prob, args.random_prob]

        # Normalize probabilities
        total_prob = sum(probs)
        probs = [p / total_prob for p in probs]

        agent = HybridAgent(agents, probs)
        print(f"Analyzing HybridAgent (heuristic={probs[0]:.1%}, random={probs[1]:.1%}) for {args.episodes} episodes...")

    # Collect data
    episodes_data = []

    for episode in range(args.episodes):
        print(f"\nCollecting Episode {episode + 1}/{args.episodes}...")
        episode_data, total_steps = collect_episode_rewards(env, agent, verbose=args.verbose)
        episodes_data.append((episode_data, total_steps))

        env_sum = sum(r[2] for r in episode_data)
        simple_sum = sum(r[3] for r in episode_data)
        print(f"  Finished: {total_steps} steps, env_reward={env_sum:.2f}, simple_reward={simple_sum:.2f}")

        # Reset hybrid agent stats for next episode
        if hasattr(agent, 'reset_stats'):
            agent.reset_stats()

    # Print usage stats for hybrid agent
    if args.agent == 'hybrid':
        print("\nHybrid Agent Usage (last episode):")
        stats = agent.get_usage_stats()
        for name, percentage in stats.items():
            print(f"  {name}: {percentage:.1%}")

    # Analyze collected data
    analyze_rewards(episodes_data)

    env.close()


if __name__ == '__main__':
    main()
