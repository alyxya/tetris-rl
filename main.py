"""
Main script for running Tetris agents.

Usage examples:
    python main.py --agent heuristic [--episodes 1] [--render]
    python main.py --agent value --model-path models/value.pt [--episodes 1]
    python main.py --agent hybrid --agents heuristic,value --probs 0.5,0.5 [--episodes 1]
"""

import argparse
import time
import numpy as np
from pufferlib.ocean.tetris import tetris
from heuristic_agent import HeuristicAgent
from value_agent import ValueAgent
from hybrid_agent import HybridAgent
from mixed_teacher_agent import MixedTeacherAgent
from reward_utils import compute_lines_cleared, compute_simple_reward, compute_shaped_reward, compute_heuristic_normalized_reward, compute_column_heights, ACTION_NAMES


def run_episode(env, agent, render=False, verbose=True, show_rewards=False, seed=None):
    """
    Run a single episode with the given agent.

    Args:
        env: Tetris environment
        agent: Agent to use
        render: Whether to render the game
        verbose: Whether to print progress
        show_rewards: Whether to show rewards and Q-values for each action
        seed: Random seed for environment reset

    Returns:
        steps: Number of steps taken
    """
    if seed is None:
        seed = int(time.time() * 1e6)
    obs, _ = env.reset(seed=seed)
    agent.reset()

    done = False
    steps = 0

    while not done:
        # Extract observation from batch (env returns batched obs)
        obs_single = obs[0] if len(obs.shape) > 1 else obs

        # Parse observation to get board states
        if show_rewards and hasattr(agent, 'parse_observation'):
            _, locked, active = agent.parse_observation(obs_single)

        # Get Q-values BEFORE taking action (for correct display)
        q_values_before = None
        if show_rewards and hasattr(agent, 'get_q_values'):
            q_values_before = agent.get_q_values(obs_single)

        action = agent.choose_action(obs_single)
        next_obs, _, terminated, truncated, _ = env.step([action])
        done = terminated[0] or truncated[0]

        # Show rewards if requested
        if show_rewards and hasattr(agent, 'parse_observation'):
            # Compute reward
            if done:
                reward = 0.0
            else:
                next_obs_single = next_obs[0] if len(next_obs.shape) > 1 else next_obs
                _, next_locked, next_active = agent.parse_observation(next_obs_single)
                lines_cleared = compute_lines_cleared(locked, active, next_locked)

                # Check if piece locked by comparing locked board changes
                old_filled_count = np.sum(locked > 0)
                new_filled_count = np.sum(next_locked > 0)
                piece_locked = new_filled_count > old_filled_count or lines_cleared > 0

                if piece_locked:
                    # Compute heuristic normalized reward
                    reward = compute_heuristic_normalized_reward(locked, next_locked, active, lines_cleared)
                else:
                    # No piece lock, no reward
                    reward = 0.0

            # Use Q-values computed before action (if available)
            q_values = q_values_before

            # Print action and reward
            if done:
                print(f"\n  Step {steps}: Action={ACTION_NAMES[action]} (Episode ended)")
            else:
                if reward != 0.0:
                    print(f"\n  Step {steps}: Action={ACTION_NAMES[action]} | Reward: {reward:+.3f} (piece locked)")
                else:
                    print(f"\n  Step {steps}: Action={ACTION_NAMES[action]} | Reward: 0.000 (positioning)")

            # Print Q-values
            if q_values is not None:
                print(f"    Q-values:")
                num_actions = len(q_values)
                for act in range(num_actions):
                    marker = " <--" if act == action else ""
                    print(f"      {ACTION_NAMES[act]:>10s}: {q_values[act]:+.6f}{marker}")
                # Note if HOLD is excluded
                if num_actions == 6:
                    print(f"      {ACTION_NAMES[6]:>10s}: (excluded from model)")

        if render:
            env.render()

        steps += 1
        obs = next_obs

        if verbose and not show_rewards and steps % 100 == 0:
            print(f"  Step {steps}")

    return steps


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run Tetris agents')
    parser.add_argument('--agent', type=str, required=True,
                        choices=['heuristic', 'value', 'hybrid', 'mixed'],
                        help='Agent type to use')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to model weights (for value agent)')
    parser.add_argument('--episodes', type=int, default=1,
                        help='Number of episodes to run')
    parser.add_argument('--render', action='store_true',
                        help='Render the game (warning: slow)')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='Print progress during episodes')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device for neural networks (cpu or cuda)')
    parser.add_argument('--show-rewards', action='store_true',
                        help='Show line-clear rewards and Q-values at each step')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')

    # Agent options
    parser.add_argument('--temperature', type=float, default=0.0,
                        help='Sampling temperature for agents (default: 0.0)')
    parser.add_argument('--epsilon', type=float, default=0.0,
                        help='Exploration epsilon for value agent')

    # Hybrid agent options
    parser.add_argument('--agents', type=str, default=None,
                        help='Comma-separated list of agents for hybrid (e.g., heuristic,value,random)')
    parser.add_argument('--probs', type=str, default=None,
                        help='Comma-separated probabilities for hybrid agents (must sum to 1.0)')
    parser.add_argument('--value-model', type=str, default=None,
                        help='Model path for value agent in hybrid')

    args = parser.parse_args()

    # Generate seed if not provided
    if args.seed is None:
        args.seed = int(time.time() * 1e6)

    print(f"Using seed: {args.seed}")

    # Create environment (seed passed to reset())
    env = tetris.Tetris()

    # Create agent
    if args.agent == 'heuristic':
        agent = HeuristicAgent(temperature=args.temperature)
        print(f"Running HeuristicAgent for {args.episodes} episode(s)...")
        if args.temperature > 0:
            print(f"  Temperature: {args.temperature}")

    elif args.agent == 'value':
        if not args.model_path:
            print("Warning: No model path provided, using randomly initialized model")
        agent = ValueAgent(device=args.device, model_path=args.model_path)
        print(f"Running ValueAgent for {args.episodes} episode(s)...")
        if args.model_path:
            print(f"Loaded model from {args.model_path}")

    elif args.agent == 'hybrid':
        if not args.agents or not args.probs:
            raise ValueError("Hybrid agent requires --agents and --probs arguments")

        # Parse agents and probabilities
        agent_names = [a.strip() for a in args.agents.split(',')]
        probs = [float(p.strip()) for p in args.probs.split(',')]

        if len(agent_names) != len(probs):
            raise ValueError("Number of agents must match number of probabilities")

        # Create sub-agents
        sub_agents = []
        for name in agent_names:
            if name == 'random':
                sub_agents.append('random')
            elif name == 'heuristic':
                sub_agents.append(HeuristicAgent(temperature=args.temperature))
            elif name == 'value':
                model_path = args.value_model or args.model_path
                if not model_path:
                    print("Warning: No model path for value agent in hybrid")
                sub_agents.append(ValueAgent(device=args.device, model_path=model_path))
            else:
                raise ValueError(f"Unknown agent type: {name}")

        agent = HybridAgent(sub_agents, probs)
        print(f"Running HybridAgent for {args.episodes} episode(s)...")
        print(f"  Agents: {agent_names}")
        print(f"  Probabilities: {probs}")

    elif args.agent == 'mixed':
        agent = MixedTeacherAgent()
        print(f"Running MixedTeacherAgent for {args.episodes} episode(s)...")
        print(f"  Random prob & temperature: (uniform(0,1))^3 per episode")

    else:
        raise ValueError(f"Unknown agent: {args.agent}")

    # Run episodes
    all_steps = []

    for episode in range(args.episodes):
        if args.episodes > 1:
            print(f"\n=== Episode {episode + 1}/{args.episodes} ===")

        # Calculate seed for this episode
        episode_seed = args.seed + episode

        # Special handling for value agent parameters
        if args.agent == 'value':
            original_choose = agent.choose_action
            def choose_action_wrapper(obs):
                return original_choose(obs, epsilon=args.epsilon, temperature=args.temperature)
            agent.choose_action = choose_action_wrapper

        steps = run_episode(env, agent, render=args.render, verbose=args.verbose, show_rewards=args.show_rewards, seed=episode_seed)

        # Restore original methods
        if args.agent == 'value':
            agent.choose_action = original_choose

        all_steps.append(steps)

        # Show mixed teacher parameters
        if args.agent == 'mixed':
            print(f"Episode {episode + 1} finished: {steps} steps (random_prob={agent.random_prob:.4f}, temp={agent.heuristic_temperature:.4f})")
        else:
            print(f"Episode {episode + 1} finished: {steps} steps")

        # Print usage stats for hybrid agent
        if args.agent == 'hybrid':
            stats = agent.get_usage_stats()
            for name, percentage in stats.items():
                print(f"  {name}: {percentage:.1%}")

    # Print summary
    if args.episodes > 1:
        print(f"\n=== Summary ===")
        print(f"Average steps: {sum(all_steps) / len(all_steps):.2f}")
        print(f"Max steps: {max(all_steps)}")
        print(f"Min steps: {min(all_steps)}")

    env.close()


if __name__ == "__main__":
    main()
