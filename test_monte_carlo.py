"""Test Monte Carlo agent on Tetris."""

import argparse
import torch
from pufferlib.ocean.tetris import tetris
from monte_carlo_agent import MonteCarloAgent
from model import ValueNetwork


def test_monte_carlo(args):
    """Test Monte Carlo agent."""
    print("Testing Monte Carlo Agent")
    print("=" * 50)
    print(f"Model: {args.model_path}")
    print(f"Rollouts: {args.num_rollouts}")
    print(f"Depth: {args.rollout_depth}")
    print(f"Temperature: {args.temperature}")
    print(f"Epsilon: {args.epsilon}")
    print("=" * 50)

    # Load model
    device = torch.device(args.device)
    model = ValueNetwork(n_rows=20, n_cols=10, n_actions=6).to(device)

    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    # Create agent
    agent = MonteCarloAgent(
        device=args.device,
        num_rollouts=args.num_rollouts,
        rollout_depth=args.rollout_depth,
        temperature=args.temperature,
        epsilon=args.epsilon
    )
    agent.model = model

    # Create environment
    env = tetris.Tetris()

    # Run episodes
    for episode in range(args.num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        lines_cleared = 0

        while not done:
            action = agent.choose_action(obs)
            obs, reward, terminated, truncated, _ = env.step([action])
            done = terminated[0] or truncated[0]

            # Count lines cleared (rough estimate from reward)
            if reward[0] > 0:
                lines_cleared += int(reward[0] * 10)

            total_reward += reward[0]
            steps += 1

            if args.render:
                env.render()

        print(f"Episode {episode + 1}: Steps={steps}, Lines={lines_cleared}, Reward={total_reward:.2f}")

    env.close()


def main():
    parser = argparse.ArgumentParser(description="Test Monte Carlo agent")
    parser.add_argument('--model-path', type=str, required=True,
                        help="Path to trained model")
    parser.add_argument('--num-episodes', type=int, default=5,
                        help="Number of test episodes")
    parser.add_argument('--num-rollouts', type=int, default=20,
                        help="Number of Monte Carlo rollouts per action")
    parser.add_argument('--rollout-depth', type=int, default=10,
                        help="Maximum depth of each rollout")
    parser.add_argument('--temperature', type=float, default=0.1,
                        help="Temperature for action sampling in rollouts")
    parser.add_argument('--epsilon', type=float, default=0.1,
                        help="Probability of random action in rollouts")
    parser.add_argument('--device', type=str, default='cpu',
                        help="Device to use (cpu, cuda, or mps)")
    parser.add_argument('--render', action='store_true',
                        help="Render the game")

    args = parser.parse_args()
    test_monte_carlo(args)


if __name__ == '__main__':
    main()
