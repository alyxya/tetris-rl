"""
Main script for running Tetris agents.

Usage examples:
    python main.py --agent heuristic [--episodes 1] [--render]
    python main.py --agent policy --model-path models/policy.pt [--episodes 1]
    python main.py --agent value --model-path models/value.pt [--episodes 1]
    python main.py --agent hybrid --agents heuristic,policy --probs 0.5,0.5 [--episodes 1]
"""

import argparse
import time
from pufferlib.ocean.tetris import tetris
from heuristic_agent import HeuristicAgent
from policy_agent import PolicyAgent
from value_agent import ValueAgent
from hybrid_agent import HybridAgent


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
        total_reward: Total reward from environment
    """
    obs, _ = env.reset(seed=int(time.time() * 1e6))
    agent.reset()

    done = False
    total_reward = 0
    steps = 0

    while not done:
        # Extract observation from batch (env returns batched obs)
        obs_single = obs[0] if len(obs.shape) > 1 else obs
        action = agent.choose_action(obs_single)
        next_obs, reward, terminated, truncated, _ = env.step([action])
        done = terminated[0] or truncated[0]

        if render:
            env.render()

        total_reward += reward[0] if hasattr(reward, '__getitem__') else reward
        steps += 1
        obs = next_obs

        if verbose and steps % 100 == 0:
            print(f"  Step {steps}, Total reward: {total_reward:.2f}")

    return steps, total_reward


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run Tetris agents')
    parser.add_argument('--agent', type=str, required=True,
                        choices=['heuristic', 'policy', 'value', 'hybrid'],
                        help='Agent type to use')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to model weights (for policy/value agents)')
    parser.add_argument('--episodes', type=int, default=1,
                        help='Number of episodes to run')
    parser.add_argument('--render', action='store_true',
                        help='Render the game (warning: slow)')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='Print progress during episodes')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device for neural networks (cpu or cuda)')

    # Policy agent options
    parser.add_argument('--deterministic', action='store_true',
                        help='Use deterministic policy (argmax) instead of sampling')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature for policy agent')

    # Value agent options
    parser.add_argument('--epsilon', type=float, default=0.0,
                        help='Exploration epsilon for value agent')

    # Hybrid agent options
    parser.add_argument('--agents', type=str, default=None,
                        help='Comma-separated list of agents for hybrid (e.g., heuristic,policy,random)')
    parser.add_argument('--probs', type=str, default=None,
                        help='Comma-separated probabilities for hybrid agents (must sum to 1.0)')
    parser.add_argument('--policy-model', type=str, default=None,
                        help='Model path for policy agent in hybrid')
    parser.add_argument('--value-model', type=str, default=None,
                        help='Model path for value agent in hybrid')

    args = parser.parse_args()

    # Create environment (seed passed to reset())
    env = tetris.Tetris()

    # Create agent
    if args.agent == 'heuristic':
        agent = HeuristicAgent()
        print(f"Running HeuristicAgent for {args.episodes} episode(s)...")

    elif args.agent == 'policy':
        if not args.model_path:
            print("Warning: No model path provided, using randomly initialized model")
        agent = PolicyAgent(device=args.device, model_path=args.model_path)
        print(f"Running PolicyAgent for {args.episodes} episode(s)...")
        if args.model_path:
            print(f"Loaded model from {args.model_path}")

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
                sub_agents.append(HeuristicAgent())
            elif name == 'policy':
                model_path = args.policy_model or args.model_path
                if not model_path:
                    print("Warning: No model path for policy agent in hybrid")
                sub_agents.append(PolicyAgent(device=args.device, model_path=model_path))
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

    else:
        raise ValueError(f"Unknown agent: {args.agent}")

    # Run episodes
    all_steps = []
    all_rewards = []

    for episode in range(args.episodes):
        if args.episodes > 1:
            print(f"\n=== Episode {episode + 1}/{args.episodes} ===")

        # Special handling for policy agent parameters
        if args.agent == 'policy':
            def choose_action_wrapper(obs):
                return agent.choose_action(obs,
                                         deterministic=args.deterministic,
                                         temperature=args.temperature)
            original_choose = agent.choose_action
            agent.choose_action = choose_action_wrapper

        # Special handling for value agent parameters
        if args.agent == 'value':
            def choose_action_wrapper(obs):
                return agent.choose_action(obs, epsilon=args.epsilon)
            original_choose = agent.choose_action
            agent.choose_action = choose_action_wrapper

        steps, reward = run_episode(env, agent, render=args.render, verbose=args.verbose)

        # Restore original methods
        if args.agent in ['policy', 'value']:
            agent.choose_action = original_choose

        all_steps.append(steps)
        all_rewards.append(reward)

        print(f"Episode {episode + 1} finished: {steps} steps, reward: {reward:.2f}")

        # Print usage stats for hybrid agent
        if args.agent == 'hybrid':
            stats = agent.get_usage_stats()
            for name, percentage in stats.items():
                print(f"  {name}: {percentage:.1%}")

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
