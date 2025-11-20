"""
Main script for running Tetris agents.

Usage:
    python main.py --agent heuristic [--episodes 1] [--render]
    python main.py --agent cnn --model-path models/cnn_agent.pt [--episodes 1] [--render]
    python main.py --agent value --model-path models/value_agent.pt [--episodes 1] [--render]
    python main.py --agent hybrid --model-path models/cnn_agent.pt [--episodes 1] [--render]
"""

import argparse
from pufferlib.ocean.tetris import tetris
from agents import HeuristicAgent, CNNAgent, HybridAgent, ValueAgent


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
                        choices=['heuristic', 'cnn', 'value', 'hybrid'],
                        help='Agent type to use')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to CNN model weights (for cnn agent)')
    parser.add_argument('--episodes', type=int, default=1,
                        help='Number of episodes to run')
    parser.add_argument('--render', action='store_true',
                        help='Render the game (warning: slow)')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='Print progress during episodes')

    args = parser.parse_args()

    # Create environment
    env = tetris.Tetris()

    # Create agent
    if args.agent == 'heuristic':
        agent = HeuristicAgent()
        print(f"Running HeuristicAgent for {args.episodes} episode(s)...")
        print("The agent evaluates all rotations and horizontal placements.")
    elif args.agent == 'cnn':
        agent = CNNAgent(model_path=args.model_path)
        print(f"Running CNNAgent for {args.episodes} episode(s)...")
        if args.model_path:
            print(f"Loaded model from {args.model_path}")
        else:
            print("Using randomly initialized model")
    elif args.agent == 'value':
        agent = ValueAgent(model_path=args.model_path)
        print(f"Running ValueAgent for {args.episodes} episode(s)...")
        if args.model_path:
            print(f"Loaded model from {args.model_path}")
        else:
            print("Using randomly initialized model")
    elif args.agent == 'hybrid':
        agent = HybridAgent(model_path=args.model_path)
        print(f"Running HybridAgent for {args.episodes} episode(s)...")
        print("The agent randomly chooses between CNN and Heuristic with 50/50 probability.")
        if args.model_path:
            print(f"Loaded CNN model from {args.model_path}")
        else:
            print("Using randomly initialized CNN model")
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

        # Print usage stats for hybrid agent
        if args.agent == 'hybrid':
            stats = agent.get_usage_stats()
            print(f"  CNN used: {stats['cnn_count']} times ({stats['cnn_percentage']:.1f}%)")
            print(f"  Heuristic used: {stats['heuristic_count']} times ({stats['heuristic_percentage']:.1f}%)")

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
