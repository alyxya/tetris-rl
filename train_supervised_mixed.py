"""
Supervised learning for Tetris Q-value network with mixed teacher.

Uses teacher demonstrations to collect transitions (s, a, r, s', done).
Trains the value network iteratively using Bellman equation to converge to optimal Q-values.
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import time
import os
import pickle
from pufferlib.ocean.tetris import tetris

from model import ValueNetwork
from mixed_teacher_agent import MixedTeacherAgent
from reward_utils import compute_lines_cleared, compute_simple_reward, compute_shaped_reward, compute_heuristic_normalized_reward


def collect_transitions(agent, env, num_episodes):
    """
    Collect transitions from teacher demonstrations.

    Returns:
        transitions: List of (state_empty, state_filled, action, reward, next_empty, next_filled, done)
    """
    transitions = []

    for _ in tqdm(range(num_episodes), desc="Collecting transitions"):
        # Reset agent for new episode
        agent.reset()

        obs, _ = env.reset(seed=int(time.time() * 1e6))
        done = False

        while not done:
            # Extract single observation from batch
            obs_single = obs[0] if len(obs.shape) > 1 else obs

            # Get current state
            _, locked, active = agent.parse_observation(obs_single)
            board_empty = locked.copy()
            board_filled = locked.copy()
            board_filled[active > 0] = 1.0

            # Choose action using teacher
            action = agent.choose_action(obs_single)

            # Take step in environment
            next_obs, _, terminated, truncated, _ = env.step([action])
            done = terminated[0] or truncated[0]

            # Compute reward based on board state change
            # Note: Reward computation is deferred to training time based on flags
            # Here we just store placeholder reward and compute actual reward later
            if done:
                reward = 0.0  # Placeholder - will be replaced with death penalty during training
                # Use dummy next state (won't be used due to done=True)
                next_empty = board_empty.copy()
                next_filled = board_filled.copy()
            else:
                # Get next state
                next_obs_single = next_obs[0] if len(next_obs.shape) > 1 else next_obs
                _, next_locked, next_active = agent.parse_observation(next_obs_single)
                reward = 0.0  # Placeholder - will be computed during training

                # Store next state
                next_empty = next_locked.copy()
                next_filled = next_locked.copy()
                next_filled[next_active > 0] = 1.0

            # Store transition with active piece info for reward computation
            transitions.append((board_empty, board_filled, action, reward, next_empty, next_filled, done, active))

            # Move to next state
            obs = next_obs

    return transitions


def train_value_network(model, transitions, args, device):
    """
    Train the value network using iterative Q-learning with Bellman updates.

    Args:
        model: ValueNetwork model
        transitions: List of (state_empty, state_filled, action, reward, next_empty, next_filled, done)
        args: Training arguments
        device: torch device
    """
    print(f"\nDataset size (before filtering): {len(transitions)} transitions")

    # Filter out HOLD actions (action 6) since model only supports 6 actions
    transitions = [t for t in transitions if t[2] != 6]
    print(f"Dataset size (after filtering HOLD): {len(transitions)} transitions")

    # Check if transitions have active piece info (new format) or not (old format)
    has_active_piece = len(transitions[0]) == 8

    # Convert transitions to arrays for batching
    states_empty = np.array([t[0] for t in transitions])
    states_filled = np.array([t[1] for t in transitions])
    actions = np.array([t[2] for t in transitions])
    next_empty = np.array([t[4] for t in transitions])
    next_filled = np.array([t[5] for t in transitions])
    dones = np.array([t[6] for t in transitions])

    if has_active_piece:
        active_pieces = [t[7] for t in transitions]
    else:
        # Old format: extract active piece from state_filled - state_empty
        print("Old dataset format detected - extracting active pieces from states...")
        active_pieces = [transitions[i][1] - transitions[i][0] for i in range(len(transitions))]

    # Compute rewards based on flags (matching RL training)
    print("Computing rewards from board states...")
    rewards = []
    for i in tqdm(range(len(transitions)), desc="Computing rewards"):
        old_board = transitions[i][0]  # state_empty
        new_board = transitions[i][4]  # next_empty
        done = transitions[i][6]
        active_piece = active_pieces[i]

        if done:
            # Apply death penalty (matching RL)
            reward = -args.death_penalty
        else:
            # Compute lines cleared
            lines = compute_lines_cleared(old_board, active_piece, new_board)

            # Check if piece locked (matching RL logic)
            old_filled_count = np.sum(old_board > 0)
            new_filled_count = np.sum(new_board > 0)
            piece_locked = new_filled_count > old_filled_count or lines > 0

            if args.heuristic_rewards and piece_locked:
                # Use heuristic normalized reward (matching RL)
                reward = compute_heuristic_normalized_reward(old_board, new_board, active_piece, lines)
            elif piece_locked:
                # Use shaped or simple rewards
                if args.shaped_rewards:
                    reward = compute_shaped_reward(old_board, new_board, lines)
                else:
                    reward = compute_simple_reward(lines)
            else:
                # Piece didn't lock
                reward = 0.0

        rewards.append(reward)

    rewards = np.array(rewards)
    print(f"Reward stats: mean={np.mean(rewards):.4f}, std={np.std(rewards):.4f}, "
          f"min={np.min(rewards):.4f}, max={np.max(rewards):.4f}")

    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Create target network for stable Q-learning
    target_model = ValueNetwork(n_rows=20, n_cols=10, n_actions=6).to(device)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()

    # Training loop
    model.train()
    best_loss = float('inf')

    for epoch in range(args.epochs):
        # Shuffle data
        indices = np.random.permutation(len(transitions))
        total_loss = 0
        num_batches = 0

        # Mini-batch training with progress bar
        num_total_batches = (len(indices) + args.batch_size - 1) // args.batch_size
        pbar = tqdm(range(0, len(indices), args.batch_size),
                    desc=f"Epoch {epoch+1}/{args.epochs}",
                    total=num_total_batches)

        for i in pbar:
            batch_idx = indices[i:i+args.batch_size]

            # Get batch data
            batch_empty = torch.FloatTensor(states_empty[batch_idx]).unsqueeze(1).to(device)
            batch_filled = torch.FloatTensor(states_filled[batch_idx]).unsqueeze(1).to(device)
            batch_actions = torch.LongTensor(actions[batch_idx]).to(device)
            batch_rewards = torch.FloatTensor(rewards[batch_idx]).to(device)
            batch_next_empty = torch.FloatTensor(next_empty[batch_idx]).unsqueeze(1).to(device)
            batch_next_filled = torch.FloatTensor(next_filled[batch_idx]).unsqueeze(1).to(device)
            batch_dones = torch.FloatTensor(dones[batch_idx]).to(device)

            # Compute Q-values for current state
            q_values = model(batch_empty, batch_filled)
            q_pred = q_values.gather(1, batch_actions.unsqueeze(1)).squeeze(1)

            # Compute target Q-values using Bellman equation: Q(s,a) = r + γ * max_a' Q(s',a')
            with torch.no_grad():
                next_q_values = target_model(batch_next_empty, batch_next_filled)
                next_q_max = next_q_values.max(1)[0]
                q_target = batch_rewards + args.gamma * next_q_max * (1 - batch_dones)

            # Compute loss (Huber loss is more robust to outliers than MSE, matching RL)
            loss = F.smooth_l1_loss(q_pred, q_target)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            # Update progress bar with running loss
            pbar.set_postfix({'loss': f'{total_loss / num_batches:.6f}'})

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {avg_loss:.6f}", flush=True)

        # Update target network every few epochs
        if (epoch + 1) % args.target_update == 0:
            target_model.load_state_dict(model.state_dict())
            print(f"Updated target network at epoch {epoch+1}", flush=True)

        # Save checkpoint every epoch
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Save epoch checkpoint
        base_name = args.output.rsplit('.', 1)[0]  # Remove extension
        ext = args.output.rsplit('.', 1)[1] if '.' in args.output else 'pth'
        epoch_checkpoint = f"{base_name}_epoch_{epoch+1}.{ext}"
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
        description="Supervised training for Tetris with iterative Q-learning"
    )
    parser.add_argument('--num-episodes', type=int, default=1000,
                        help="Number of episodes to collect")
    parser.add_argument('--epochs', type=int, default=20,
                        help="Number of training epochs")
    parser.add_argument('--batch-size', type=int, default=256,
                        help="Batch size for training")
    parser.add_argument('--lr', type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument('--gamma', type=float, default=0.99,
                        help="Discount factor (matching RL default)")
    parser.add_argument('--device', type=str, default='cpu',
                        help="Device to use (cpu, cuda, or mps)")
    parser.add_argument('--output', type=str, required=True,
                        help="Output path for trained model")
    parser.add_argument('--save-data', type=str, default=None,
                        help="Path to save collected transitions (optional)")
    parser.add_argument('--load-data', type=str, default=None,
                        help="Path to load pre-collected transitions (optional)")
    parser.add_argument('--init-model', type=str, default=None,
                        help="Path to pretrained model to continue training from (optional)")
    parser.add_argument('--target-update', type=int, default=5,
                        help="Update target network every N epochs")
    parser.add_argument('--shaped-rewards', action='store_true',
                        help="Use shaped rewards (height, holes, bumpiness) instead of simple line-clear rewards")
    parser.add_argument('--heuristic-rewards', action='store_true',
                        help="Use heuristic normalized rewards (compares placement against all possibilities, matching RL)")
    parser.add_argument('--death-penalty', type=float, default=0.0,
                        help="Penalty applied when agent dies (default: 0.0, matching RL when specified)")

    args = parser.parse_args()

    print("Supervised Training with Iterative Q-Learning")
    print("=" * 50)

    device = torch.device(args.device)

    # Load or collect data
    if args.load_data:
        print(f"\nLoading transitions from {args.load_data}...")
        with open(args.load_data, 'rb') as f:
            transitions = pickle.load(f)
        print(f"Loaded {len(transitions)} transitions")
    else:
        # Create environment and teacher
        env = tetris.Tetris()
        teacher = MixedTeacherAgent()

        # Collect transitions
        print(f"\nCollecting {args.num_episodes} episodes with teacher...")
        transitions = collect_transitions(teacher, env, args.num_episodes)

        # Optionally save dataset
        if args.save_data:
            print(f"\nSaving transitions to {args.save_data}...")
            output_dir = os.path.dirname(args.save_data)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            with open(args.save_data, 'wb') as f:
                pickle.dump(transitions, f)
            print(f"Transitions saved!")

    # Create model (6 actions: NO_OP, LEFT, RIGHT, ROTATE, SOFT_DROP, HARD_DROP - excludes HOLD)
    model = ValueNetwork(n_rows=20, n_cols=10, n_actions=6).to(device)

    # Load pretrained weights if provided
    if args.init_model:
        print(f"\nLoading pretrained model from {args.init_model}...")
        checkpoint = torch.load(args.init_model, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("Model loaded successfully!")

    # Train model
    train_value_network(model, transitions, args, device)


if __name__ == '__main__':
    main()
