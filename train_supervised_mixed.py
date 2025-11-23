"""
Supervised learning for Tetris Q-value network with mixed teacher.

Uses a combination of random actions and heuristic agent to collect training data.
Trains the value network to predict Q-values based on simple line-clear rewards.
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time
import os
import pickle
from pufferlib.ocean.tetris import tetris

from model import ValueNetwork
from mixed_teacher_agent import MixedTeacherAgent
from reward_utils import compute_lines_cleared, compute_simple_reward


def collect_rollouts(agent, env, num_episodes, gamma=0.99):
    """
    Collect rollouts from mixed teacher and compute Q-value targets.

    For each state-action pair, compute the reward based on line clears
    and propagate future rewards with discounting.

    Returns:
        states_empty: List of board_empty states
        states_filled: List of board_filled states
        actions: List of actions taken
        q_targets: List of Q-value targets (discounted returns)
    """
    all_states_empty = []
    all_states_filled = []
    all_actions = []
    all_q_targets = []

    for _ in tqdm(range(num_episodes), desc="Collecting rollouts"):
        # Reset agent for new episode (samples new random_prob and temperature)
        agent.reset()

        obs, _ = env.reset(seed=int(time.time() * 1e6))
        done = False

        # Episode data
        episode_states_empty = []
        episode_states_filled = []
        episode_actions = []
        episode_rewards = []

        while not done:
            # Extract single observation from batch
            obs_single = obs[0] if len(obs.shape) > 1 else obs

            # Get current state
            _, locked, active = agent.parse_observation(obs_single)
            board_empty = locked.copy()
            board_filled = locked.copy()
            board_filled[active > 0] = 1.0

            # Choose action using mixed teacher
            action = agent.choose_action(obs_single)

            # Store state and action
            episode_states_empty.append(board_empty)
            episode_states_filled.append(board_filled)
            episode_actions.append(action)

            # Take step in environment
            next_obs, _, terminated, truncated, _ = env.step([action])
            done = terminated[0] or truncated[0]

            # Compute reward based on line clears
            if done:
                # No reward on death (board state is invalid)
                reward = 0.0
            else:
                # Get next state to compute reward
                next_obs_single = next_obs[0] if len(next_obs.shape) > 1 else next_obs
                _, next_locked, _ = agent.parse_observation(next_obs_single)
                lines_cleared = compute_lines_cleared(locked, active, next_locked)
                reward = compute_simple_reward(lines_cleared)

            episode_rewards.append(reward)

            # Move to next state
            obs = next_obs

        # Compute discounted returns for this episode
        returns = []
        R = 0.0
        for r in reversed(episode_rewards):
            R = r + gamma * R
            returns.insert(0, R)

        # Add episode data to dataset
        all_states_empty.extend(episode_states_empty)
        all_states_filled.extend(episode_states_filled)
        all_actions.extend(episode_actions)
        all_q_targets.extend(returns)

    return all_states_empty, all_states_filled, all_actions, all_q_targets


def train_value_network(model, states_empty, states_filled, actions, q_targets,
                       args, device):
    """
    Train the value network on collected data.

    Args:
        model: ValueNetwork model
        states_empty: List of board_empty states
        states_filled: List of board_filled states
        actions: List of actions
        q_targets: List of Q-value targets
        args: Training arguments
        device: torch device
    """
    # Convert to tensors
    states_empty = torch.FloatTensor(np.array(states_empty)).unsqueeze(1).to(device)
    states_filled = torch.FloatTensor(np.array(states_filled)).unsqueeze(1).to(device)
    actions = torch.LongTensor(actions).to(device)
    q_targets = torch.FloatTensor(q_targets).to(device)

    print(f"\nDataset size: {len(actions)} samples")
    print(f"Q-target stats: mean={q_targets.mean():.4f}, std={q_targets.std():.4f}, "
          f"min={q_targets.min():.4f}, max={q_targets.max():.4f}")

    # Setup optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # Training loop
    model.train()
    best_loss = float('inf')

    for epoch in range(args.epochs):
        # Shuffle data
        indices = torch.randperm(len(actions))
        total_loss = 0
        num_batches = 0

        # Mini-batch training with progress bar
        num_total_batches = (len(indices) + args.batch_size - 1) // args.batch_size
        pbar = tqdm(range(0, len(indices), args.batch_size),
                    desc=f"Epoch {epoch+1}/{args.epochs}",
                    total=num_total_batches)

        for i in pbar:
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

            # Update progress bar with running loss
            pbar.set_postfix({'loss': f'{total_loss / num_batches:.6f}'})

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {avg_loss:.6f}", flush=True)

        # Save checkpoint every epoch
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Save epoch checkpoint
        base_name = args.output.rsplit('.', 1)[0]  # Remove extension
        ext = args.output.rsplit('.', 1)[1] if '.' in args.output else 'pth'
        epoch_checkpoint = f"{base_name}_epoch{epoch+1}.{ext}"
        torch.save(model.state_dict(), epoch_checkpoint)
        print(f"Saved checkpoint to {epoch_checkpoint}", flush=True)

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), args.output)
            print(f"Saved best model to {args.output}", flush=True)

    print("\nTraining complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Supervised training for Tetris with mixed teacher"
    )
    parser.add_argument('--num-episodes', type=int, default=1000,
                        help="Number of episodes to collect")
    parser.add_argument('--epochs', type=int, default=20,
                        help="Number of training epochs")
    parser.add_argument('--batch-size', type=int, default=64,
                        help="Batch size for training")
    parser.add_argument('--lr', type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument('--gamma', type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument('--device', type=str, default='cpu',
                        help="Device to use (cpu or cuda)")
    parser.add_argument('--output', type=str, required=True,
                        help="Output path for trained model")
    parser.add_argument('--save-data', type=str, default=None,
                        help="Path to save collected dataset (optional)")
    parser.add_argument('--load-data', type=str, default=None,
                        help="Path to load pre-collected dataset (optional)")
    parser.add_argument('--init-model', type=str, default=None,
                        help="Path to pretrained model to continue training from (optional)")

    args = parser.parse_args()

    print("Supervised Training with Mixed Teacher")
    print("=" * 50)

    device = torch.device(args.device)

    # Load or collect data
    if args.load_data:
        print(f"\nLoading dataset from {args.load_data}...")
        with open(args.load_data, 'rb') as f:
            data = pickle.load(f)
            states_empty = data['states_empty']
            states_filled = data['states_filled']
            actions = data['actions']
            q_targets = data['q_targets']
        print(f"Loaded {len(actions)} samples")
    else:
        # Create environment and teacher
        env = tetris.Tetris()
        teacher = MixedTeacherAgent()

        # Collect rollouts
        print(f"\nCollecting {args.num_episodes} episodes with mixed teacher...")
        states_empty, states_filled, actions, q_targets = collect_rollouts(
            teacher, env, args.num_episodes, args.gamma
        )

        # Optionally save dataset
        if args.save_data:
            print(f"\nSaving dataset to {args.save_data}...")
            data = {
                'states_empty': states_empty,
                'states_filled': states_filled,
                'actions': actions,
                'q_targets': q_targets,
            }
            output_dir = os.path.dirname(args.save_data)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            with open(args.save_data, 'wb') as f:
                pickle.dump(data, f)
            print(f"Dataset saved!")

    # Create and train model
    model = ValueNetwork(n_rows=20, n_cols=10, n_actions=7).to(device)

    # Load pretrained weights if provided
    if args.init_model:
        print(f"\nLoading pretrained model from {args.init_model}...")
        checkpoint = torch.load(args.init_model, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("Model loaded successfully!")

    train_value_network(model, states_empty, states_filled, actions, q_targets,
                       args, device)


if __name__ == '__main__':
    main()
