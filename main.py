"""
Main script for running Tetris agents.

Usage:
    python main.py --agent heuristic [--episodes 1] [--render]
"""

import argparse
from pufferlib.ocean.tetris import tetris
from agents import HeuristicAgent


def run_episode(env, agent, render=False, verbose=True):
    """
    Run a single episode with the given agent.

    Args:
        env: Tetris environment
        agent: Agent to use
        render: Whether to render the game
        verbose: Whether to print progress

    Returns:
        steps: Number of steps taken
        total_reward: Total reward achieved
    """
    obs, _ = env.reset()
    agent.reset()

    done = False
    total_reward = 0
    steps = 0

    while not done:
        action = agent.choose_action(obs[0])
        obs, reward, terminated, truncated, info = env.step([action])

        if render:
            env.render()

        total_reward += reward[0]
        steps += 1
        done = terminated[0] or truncated[0]

        if verbose and steps % 100 == 0:
            print(f"  Step {steps}, Total reward: {total_reward:.2f}")

    return steps, total_reward


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run Tetris agents')
    parser.add_argument('--agent', type=str, default='heuristic',
                        choices=['heuristic'],
                        help='Agent type to use')
    parser.add_argument('--episodes', type=int, default=1,
                        help='Number of episodes to run')
    parser.add_argument('--render', action='store_true',
                        help='Render the game (warning: slow)')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='Print progress during episodes')
    parser.add_argument('--no-rotation', action='store_true',
                        help='Disable rotation consideration for heuristic agent')

    args = parser.parse_args()

    # Create environment
    env = tetris.Tetris()

    # Create agent
    if args.agent == 'heuristic':
        use_rotation = not args.no_rotation
        agent = HeuristicAgent(use_rotation=use_rotation)
        print(f"Running HeuristicAgent for {args.episodes} episode(s)...")
        if use_rotation:
            print("The agent evaluates all rotations and horizontal placements.")
        else:
            print("The agent evaluates horizontal placements (no rotation).")
    else:
        raise ValueError(f"Unknown agent: {args.agent}")

    # Run episodes
    all_steps = []
    all_rewards = []

    for episode in range(args.episodes):
        if args.episodes > 1:
            print(f"\n=== Episode {episode + 1}/{args.episodes} ===")

        steps, reward = run_episode(env, agent, render=args.render, verbose=args.verbose)

        all_steps.append(steps)
        all_rewards.append(reward)

        print(f"Episode {episode + 1} finished: {steps} steps, reward: {reward:.2f}")

    # Print summary
    if args.episodes > 1:
        print(f"\n=== Summary ===")
        print(f"Average steps: {sum(all_steps) / len(all_steps):.2f}")
        print(f"Average reward: {sum(all_rewards) / len(all_rewards):.2f}")
        print(f"Best reward: {max(all_rewards):.2f}")
        print(f"Worst reward: {min(all_rewards):.2f}")

    env.close()


if __name__ == "__main__":
    main()
