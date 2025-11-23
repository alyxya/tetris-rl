"""
Supervised learning for Tetris agents.

Training mode:
- Value network: Learn Q-values from heuristic-based rewards via rollouts
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time
import os
from pufferlib.ocean.tetris import tetris

from model import ValueNetwork
from heuristic_agent import HeuristicAgent
from value_agent import ValueAgent
import heuristic as heuristic_module


def collect_value_rollouts(agent, env, num_episodes, gamma=0.99):
    """
    Collect rollouts and compute heuristic-based Q-value targets.

    For each state-action pair, compute the heuristic reward and propagate
    future rewards with discounting.

    Returns:
        states_empty: List of board_empty states
        states_filled: List of board_filled states
        actions: List of actions taken
        q_targets: List of Q-value targets
    """
    states_empty = []
    states_filled = []
    actions = []
    rewards = []

    for _ in tqdm(range(num_episodes), desc="Collecting rollouts"):
        obs, _ = env.reset(seed=int(time.time() * 1e6))
        done = False
        episode_data = []

        while not done:
            action = agent.choose_action(obs)

            # Store state and action
            _, locked, active = agent.parse_observation(obs)
            board_empty = locked.copy()
            board_filled = locked.copy()
            board_filled[active > 0] = 1.0

            episode_data.append((board_empty, board_filled, action))

            # Take step
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Compute heuristic reward for this action
            piece_shape = agent.extract_piece_shape(active)
            if piece_shape is not None:
                # Evaluate the action using heuristic
                reward, _ = heuristic_module.evaluate_placement(
                    locked, piece_shape, 0, 0  # Dummy rotation/col, will be overridden
                )
            else:
                reward = 0.0

            rewards.append(reward)

        # Compute discounted returns for this episode
        returns = []
        R = 0
        for r in reversed(rewards[-len(episode_data):]):
            R = r + gamma * R
            returns.insert(0, R)

        # Add to dataset
        for (be, bf, a), ret in zip(episode_data, returns):
            states_empty.append(be)
            states_filled.append(bf)
            actions.append(a)

        # Clear episode rewards
        rewards = rewards[:-len(episode_data)]

    # Compute Q-targets: for each (s,a) pair, the target is the discounted return
    q_targets = rewards  # Actually returns, but we call them q_targets

    return states_empty, states_filled, actions, q_targets


def train_value_network(args):
    """Train value network with heuristic-based rewards."""
    print("Training Value Network")
    print("=" * 50)

    device = torch.device(args.device)

    # Create environment and teacher
    env = tetris.Tetris()
    teacher = HeuristicAgent()

    # Create model
    model = ValueNetwork(n_rows=20, n_cols=10, n_actions=7).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # Collect rollouts
    print(f"\nCollecting {args.num_episodes} episodes...")
    states_empty, states_filled, actions, q_targets = collect_value_rollouts(
        teacher, env, args.num_episodes, args.gamma
    )

    # Convert to tensors
    states_empty = torch.FloatTensor(np.array(states_empty)).unsqueeze(1).to(device)
    states_filled = torch.FloatTensor(np.array(states_filled)).unsqueeze(1).to(device)
    actions = torch.LongTensor(actions).to(device)
    q_targets = torch.FloatTensor(q_targets).to(device)

    print(f"Dataset size: {len(actions)} samples")

    # Training loop
    model.train()
    best_loss = float('inf')

    for epoch in range(args.epochs):
        # Shuffle data
        indices = torch.randperm(len(actions))
        total_loss = 0
        num_batches = 0

        # Mini-batch training
        for i in range(0, len(indices), args.batch_size):
            batch_idx = indices[i:i+args.batch_size]

            batch_empty = states_empty[batch_idx]
            batch_filled = states_filled[batch_idx]
            batch_actions = actions[batch_idx]
            batch_targets = q_targets[batch_idx]

            # Forward pass
            q_values = model(batch_empty, batch_filled)
            q_pred = q_values.gather(1, batch_actions.unsqueeze(1)).squeeze(1)

            # Compute loss
            loss = criterion(q_pred, batch_targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {avg_loss:.6f}")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            output_dir = os.path.dirname(args.output)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            torch.save(model.state_dict(), args.output)
            print(f"Saved best model to {args.output}")

    print("\nTraining complete!")


def main():
    parser = argparse.ArgumentParser(description="Supervised training for Tetris")
    parser.add_argument('--num-episodes', type=int, default=100,
                        help="Number of episodes to collect")
    parser.add_argument('--epochs', type=int, default=20,
                        help="Number of training epochs")
    parser.add_argument('--batch-size', type=int, default=64,
                        help="Batch size for training")
    parser.add_argument('--lr', type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument('--gamma', type=float, default=0.99,
                        help="Discount factor (for value mode)")
    parser.add_argument('--device', type=str, default='cpu',
                        help="Device to use (cpu or cuda)")
    parser.add_argument('--output', type=str, required=True,
                        help="Output path for trained model")

    args = parser.parse_args()

    train_value_network(args)


if __name__ == '__main__':
    main()
