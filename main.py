"""
Main script for running Tetris agents.

Usage:
    python main.py --agent heuristic [--episodes 1] [--render]
    python main.py --agent cnn --model-path models/cnn_agent.pt [--episodes 1] [--render]
    python main.py --agent value --model-path models/value_agent.pt [--episodes 1] [--render]
    python main.py --agent hybrid --model-path models/value_agent.pt [--episodes 1] [--render] [--student-probability 0.5]
"""

import argparse
from pufferlib.ocean.tetris import tetris
from agents import HeuristicAgent, CNNAgent, HybridAgent, ValueAgent


def run_episode(env, agent, render=False, verbose=True, debug_values=False):
    """
    Run a single episode with the given agent.

    Args:
        env: Tetris environment
        agent: Agent to use
        render: Whether to render the game
        verbose: Whether to print progress
        debug_values: Whether to print predicted values (ValueAgent only)

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
        # Get action (with optional debug output for ValueAgent)
        if debug_values and hasattr(agent, 'get_action_values'):
            values = agent.get_action_values(obs[0])
            action = agent.choose_action(obs[0])
            print(f"Step {steps}: Values={values.round(3)}, Best action={action} (value={values[action]:.3f})")
        else:
            action = agent.choose_action(obs[0])

        obs, reward, terminated, truncated, info = env.step([action])

        if render:
            env.render()

        # Extract line clear rewards only
        step_reward = round(reward[0], 2)
        if step_reward >= 0.09:
            step_reward = round(step_reward / 0.1) * 0.1
        else:
            step_reward = 0.0
        total_reward += step_reward
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
    parser.add_argument('--student-probability', type=float, default=0.5,
                        help='For hybrid agent: probability of using student (default: 0.5)')
    parser.add_argument('--random-probability', type=float, default=0.0,
                        help='For hybrid agent: probability of using random action (default: 0.0)')
    parser.add_argument('--debug-values', action='store_true',
                        help='Print predicted values for each action (value agent only)')

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
        agent = HybridAgent(
            model_path=args.model_path,
            student_probability=args.student_probability,
            random_probability=args.random_probability
        )
        print(f"Running HybridAgent for {args.episodes} episode(s)...")
        teacher_prob = 1.0 - args.student_probability - args.random_probability
        print(f"  Student: {args.student_probability:.1%}, Random: {args.random_probability:.1%}, Teacher: {teacher_prob:.1%}")
        if args.model_path:
            print(f"Loaded student model from {args.model_path}")
        else:
            print("Using randomly initialized student model")
    else:
        raise ValueError(f"Unknown agent: {args.agent}")

    # Run episodes
    all_steps = []
    all_rewards = []

    for episode in range(args.episodes):
        if args.episodes > 1:
            print(f"\n=== Episode {episode + 1}/{args.episodes} ===")

        steps, reward = run_episode(env, agent, render=args.render, verbose=args.verbose, debug_values=args.debug_values)

        all_steps.append(steps)
        all_rewards.append(reward)

        print(f"Episode {episode + 1} finished: {steps} steps, reward: {reward:.2f}")

        # Print usage stats for hybrid agent
        if args.agent == 'hybrid':
            stats = agent.get_usage_stats()
            print(f"  Student: {stats['student_count']} times ({stats['student_percentage']:.1f}%)")
            print(f"  Random: {stats['random_count']} times ({stats['random_percentage']:.1f}%)")
            print(f"  Teacher: {stats['teacher_count']} times ({stats['teacher_percentage']:.1f}%)")

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
