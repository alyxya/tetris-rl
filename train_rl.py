"""
Reinforcement learning for Tetris agents.

Two training modes:
1. Value network: Q-learning with heuristic rewards
2. Policy network: REINFORCE with heuristic auxiliary rewards
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from tqdm import tqdm
import time
import os
from pufferlib.ocean.tetris import tetris

from model import PolicyNetwork, ValueNetwork
from value_agent import ValueAgent
from policy_agent import PolicyAgent
import heuristic as heuristic_module
from heuristic import rotate_piece_cw


class ReplayBuffer:
    """Fixed-size replay buffer for experience replay."""

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state_empty, state_filled, action, reward, next_empty, next_filled, done):
        """Add experience to buffer."""
        self.buffer.append((state_empty, state_filled, action, reward, next_empty, next_filled, done))

    def sample(self, batch_size):
        """Sample random batch from buffer."""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]

        states_empty = np.array([x[0] for x in batch])
        states_filled = np.array([x[1] for x in batch])
        actions = np.array([x[2] for x in batch])
        rewards = np.array([x[3] for x in batch])
        next_empty = np.array([x[4] for x in batch])
        next_filled = np.array([x[5] for x in batch])
        dones = np.array([x[6] for x in batch])

        return states_empty, states_filled, actions, rewards, next_empty, next_filled, dones

    def __len__(self):
        return len(self.buffer)


ACTION_NO_OP = 0
ACTION_LEFT = 1
ACTION_RIGHT = 2
ACTION_ROTATE = 3
ACTION_SOFT_DROP = 4
ACTION_HARD_DROP = 5
ACTION_HOLD = 6


def compute_heuristic_reward(locked_board, active_piece, next_locked_board, action):
    """
    Compute heuristic-based reward for the chosen action.

    Reward structure:
    1. Large immediate reward for line clears (if they happened)
    2. Otherwise, normalize action-specific heuristic scores (mean 0, std 0.01)
       based on placements across identity and single-rotation orientations.
    """
    piece_shape = extract_piece_shape_from_board(active_piece)
    if piece_shape is None:
        return 0.0

    # Check if lines were cleared (compare locked block counts)
    prev_locked_count = np.sum(locked_board > 0)
    next_locked_count = np.sum(next_locked_board > 0)

    # Calculate lines cleared (each cleared line removes 10 blocks, piece adds ~4 blocks)
    # Delta = prev_count + piece_blocks - cleared_lines * 10
    # So: cleared_lines = (prev_count + piece_blocks - next_count) / 10
    piece_blocks = np.sum(piece_shape > 0)
    delta = prev_locked_count + piece_blocks - next_locked_count
    lines_cleared = max(0, delta // 10)  # Integer division

    # Line clear rewards (scaled down by 10x from original)
    line_clear_rewards = {0: 0.0, 1: 1.0, 2: 3.0, 3: 6.0, 4: 10.0}
    line_reward = line_clear_rewards.get(lines_cleared, 0.0)

    # If lines were cleared, return the large reward immediately
    if lines_cleared > 0:
        return line_reward

    # Otherwise, compute action-based heuristic scores
    # Get current piece position
    piece_positions = np.argwhere(active_piece > 0)
    current_left_col = piece_positions[:, 1].min()

    # Evaluate placements for identity and single clockwise rotation only
    n_cols = locked_board.shape[1]
    rotations_to_consider = (0, 1)
    placements = []  # List of (rotation, col, score)

    for rotation in rotations_to_consider:
        rotated_shape = piece_shape.copy()
        for _ in range(rotation):
            rotated_shape = rotate_piece_cw(rotated_shape)

        piece_width = rotated_shape.shape[1]

        for col in range(n_cols - piece_width + 1):
            score, _ = heuristic_module.evaluate_placement(
                locked_board, piece_shape, rotation, col
            )

            # Only consider valid placements
            if score > float('-inf'):
                placements.append((rotation, col, score))

    if len(placements) == 0:
        return 0.0

    scores = np.array([p[2] for p in placements])
    cols = np.array([p[1] for p in placements])
    rotations = np.array([p[0] for p in placements])

    # Find score for current column and rotation 0 (identity orientation)
    current_mask = (cols == current_left_col) & (rotations == 0)
    current_score = float(scores[current_mask][0]) if current_mask.any() else 0.0

    left_mask = cols <= current_left_col
    mean_left = float(np.mean(scores[left_mask])) if left_mask.any() else 0.0

    right_mask = cols >= current_left_col
    mean_right = float(np.mean(scores[right_mask])) if right_mask.any() else 0.0

    rotation_mask = rotations > 0
    mean_rotation = float(np.mean(scores[rotation_mask])) if rotation_mask.any() else 0.0

    raw_scores = {
        ACTION_NO_OP: current_score,
        ACTION_LEFT: mean_left,
        ACTION_RIGHT: mean_right,
        ACTION_ROTATE: mean_rotation,
        ACTION_SOFT_DROP: current_score,
    }

    raw_values = np.array(list(raw_scores.values()), dtype=np.float32)
    raw_mean = float(raw_values.mean())
    raw_std = float(raw_values.std())
    if raw_std == 0.0:
        raw_std = 1.0

    normalized = (raw_values - raw_mean) / raw_std
    target_std = 0.01
    normalized *= target_std

    rewards_by_action = np.zeros(7, dtype=np.float32)
    for idx, act in enumerate(raw_scores.keys()):
        rewards_by_action[act] = float(normalized[idx])

    return float(rewards_by_action[action])


def extract_piece_shape_from_board(active_piece):
    """Extract piece shape from active piece board."""
    piece_positions = np.argwhere(active_piece > 0)
    if len(piece_positions) == 0:
        return None

    min_row, min_col = piece_positions.min(axis=0)
    max_row, max_col = piece_positions.max(axis=0)
    piece_shape = active_piece[min_row:max_row+1, min_col:max_col+1]
    return piece_shape


def train_value_rl(args):
    """Train value network with Q-learning."""
    print("Training Value Network with Q-Learning")
    print("=" * 50)

    device = torch.device(args.device)
    env = tetris.Tetris()

    # Create online and target networks
    online_net = ValueNetwork(n_rows=20, n_cols=10, n_actions=7).to(device)
    target_net = ValueNetwork(n_rows=20, n_cols=10, n_actions=7).to(device)

    # Load pretrained weights if provided
    if args.init_model:
        checkpoint = torch.load(args.init_model, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            online_net.load_state_dict(checkpoint['model_state_dict'])
        else:
            online_net.load_state_dict(checkpoint)
        print(f"Loaded initial weights from {args.init_model}")

    target_net.load_state_dict(online_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(online_net.parameters(), lr=args.lr)
    replay_buffer = ReplayBuffer(args.buffer_size)

    # Training loop
    best_reward = float('-inf')
    epsilon = args.epsilon_start
    epsilon_decay = (args.epsilon_start - args.epsilon_end) / args.num_episodes

    agent = ValueAgent(device=args.device)
    agent.model = online_net

    for episode in tqdm(range(args.num_episodes), desc="Training"):
        obs, _ = env.reset(seed=int(time.time() * 1e6))
        done = False
        total_reward = 0
        steps = 0

        while not done:
            # Extract single observation from batch
            obs_single = obs[0] if len(obs.shape) > 1 else obs

            # Parse state
            _, locked, active = agent.parse_observation(obs_single)
            board_empty = locked.copy()
            board_filled = locked.copy()
            board_filled[active > 0] = 1.0

            # Epsilon-greedy action selection
            action = agent.choose_action(obs_single, epsilon=epsilon)

            # Take step
            next_obs, _, terminated, truncated, _ = env.step([action])
            done = terminated[0] or truncated[0]

            # Parse next state
            next_obs_single = next_obs[0] if len(next_obs.shape) > 1 else next_obs
            _, next_locked, next_active = agent.parse_observation(next_obs_single)

            # Compute action-conditioned heuristic reward (line clears + normalized heuristic score)
            # Apply penalty for losing the game
            if done:
                reward = -1.0
            else:
                reward = compute_heuristic_reward(locked, active, next_locked, action)

            next_empty = next_locked.copy()
            next_filled = next_locked.copy()
            next_filled[next_active > 0] = 1.0

            # Store in replay buffer
            replay_buffer.push(board_empty, board_filled, action, reward, next_empty, next_filled, done)

            total_reward += reward
            steps += 1
            obs = next_obs

            # Training step
            if len(replay_buffer) >= args.batch_size:
                # Sample batch
                batch_empty, batch_filled, batch_actions, batch_rewards, batch_next_empty, batch_next_filled, batch_dones = replay_buffer.sample(args.batch_size)

                # Convert to tensors
                batch_empty = torch.FloatTensor(batch_empty).unsqueeze(1).to(device)
                batch_filled = torch.FloatTensor(batch_filled).unsqueeze(1).to(device)
                batch_actions = torch.LongTensor(batch_actions).to(device)
                batch_rewards = torch.FloatTensor(batch_rewards).to(device)
                batch_next_empty = torch.FloatTensor(batch_next_empty).unsqueeze(1).to(device)
                batch_next_filled = torch.FloatTensor(batch_next_filled).unsqueeze(1).to(device)
                batch_dones = torch.FloatTensor(batch_dones).to(device)

                # Compute Q-values
                q_values = online_net(batch_empty, batch_filled)
                q_pred = q_values.gather(1, batch_actions.unsqueeze(1)).squeeze(1)

                # Compute target Q-values
                with torch.no_grad():
                    next_q_values = target_net(batch_next_empty, batch_next_filled)
                    next_q_max = next_q_values.max(1)[0]
                    q_target = batch_rewards + args.gamma * next_q_max * (1 - batch_dones)

                # Compute loss
                loss = F.mse_loss(q_pred, q_target)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(online_net.parameters(), args.grad_clip)
                optimizer.step()

        # Update target network
        if episode % args.target_update == 0:
            target_net.load_state_dict(online_net.state_dict())

        # Decay epsilon
        epsilon = max(args.epsilon_end, epsilon - epsilon_decay)

        # Logging
        if episode % 10 == 0:
            print(f"\nEpisode {episode} - Steps: {steps}, Reward: {total_reward:.2f}, Epsilon: {epsilon:.3f}")

        # Save best model
        if total_reward > best_reward:
            best_reward = total_reward
            output_dir = os.path.dirname(args.output)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            torch.save(online_net.state_dict(), args.output)
            print(f"Saved best model with reward {best_reward:.2f}")

    # Save final model
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    torch.save(online_net.state_dict(), args.output)
    print(f"\nTraining complete! Final model saved to {args.output}")


def train_policy_rl(args):
    """Train policy network with REINFORCE."""
    print("Training Policy Network with REINFORCE")
    print("=" * 50)

    device = torch.device(args.device)
    env = tetris.Tetris()

    # Create model
    model = PolicyNetwork(n_rows=20, n_cols=10, n_actions=7).to(device)

    # Load pretrained weights if provided
    if args.init_model:
        checkpoint = torch.load(args.init_model, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded initial weights from {args.init_model}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    agent = PolicyAgent(device=args.device)
    agent.model = model

    best_reward = float('-inf')

    for episode in tqdm(range(args.num_episodes), desc="Training"):
        obs, _ = env.reset(seed=int(time.time() * 1e6))
        done = False

        states_empty = []
        states_filled = []
        actions = []
        rewards = []

        # Collect episode
        while not done:
            # Extract single observation from batch
            obs_single = obs[0] if len(obs.shape) > 1 else obs

            # Parse state
            _, locked, active = agent.parse_observation(obs_single)
            board_empty = locked.copy()
            board_filled = locked.copy()
            board_filled[active > 0] = 1.0

            states_empty.append(board_empty)
            states_filled.append(board_filled)

            # Sample action
            action = agent.choose_action(obs_single, deterministic=False, temperature=args.temperature)
            actions.append(action)

            # Take step
            next_obs, _, terminated, truncated, _ = env.step([action])
            done = terminated[0] or truncated[0]

            # Parse next state
            next_obs_single = next_obs[0] if len(next_obs.shape) > 1 else next_obs
            _, next_locked, _ = agent.parse_observation(next_obs_single)

            # Compute action-conditioned heuristic reward (line clears + normalized heuristic score)
            # Apply penalty for losing the game
            if done:
                reward = -1.0
            else:
                reward = compute_heuristic_reward(locked, active, next_locked, action)
            rewards.append(reward)

            obs = next_obs

        # Compute returns
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + args.gamma * R
            returns.insert(0, R)

        # Convert to tensors
        states_empty = torch.FloatTensor(np.array(states_empty)).unsqueeze(1).to(device)
        states_filled = torch.FloatTensor(np.array(states_filled)).unsqueeze(1).to(device)
        actions = torch.LongTensor(actions).to(device)
        returns = torch.FloatTensor(returns).to(device)

        # Normalize returns
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Compute policy loss
        logits = model(states_empty, states_filled)
        log_probs = F.log_softmax(logits, dim=1)
        action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        policy_loss = -(action_log_probs * returns).mean()

        # Backward pass
        optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        total_reward = sum(rewards)

        # Logging
        if episode % 10 == 0:
            print(f"\nEpisode {episode} - Steps: {len(rewards)}, Reward: {total_reward:.2f}")

        # Save best model
        if total_reward > best_reward:
            best_reward = total_reward
            output_dir = os.path.dirname(args.output)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            torch.save(model.state_dict(), args.output)
            print(f"Saved best model with reward {best_reward:.2f}")

    # Save final model
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), args.output)
    print(f"\nTraining complete! Final model saved to {args.output}")


def main():
    parser = argparse.ArgumentParser(description="RL training for Tetris")
    parser.add_argument('--mode', type=str, required=True, choices=['value', 'policy'],
                        help="Training mode: 'value' or 'policy'")
    parser.add_argument('--num-episodes', type=int, default=1000,
                        help="Number of episodes to train")
    parser.add_argument('--lr', type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument('--gamma', type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument('--device', type=str, default='cpu',
                        help="Device to use (cpu or cuda)")
    parser.add_argument('--output', type=str, required=True,
                        help="Output path for trained model")
    parser.add_argument('--init-model', type=str, default=None,
                        help="Path to pretrained model to initialize from")
    parser.add_argument('--grad-clip', type=float, default=1.0,
                        help="Gradient clipping threshold")

    # Value-specific args
    parser.add_argument('--buffer-size', type=int, default=10000,
                        help="Replay buffer size (value mode)")
    parser.add_argument('--batch-size', type=int, default=64,
                        help="Batch size (value mode)")
    parser.add_argument('--epsilon-start', type=float, default=0.2,
                        help="Initial epsilon (value mode)")
    parser.add_argument('--epsilon-end', type=float, default=0.01,
                        help="Final epsilon (value mode)")
    parser.add_argument('--target-update', type=int, default=10,
                        help="Target network update frequency (value mode)")

    # Policy-specific args
    parser.add_argument('--temperature', type=float, default=1.0,
                        help="Sampling temperature (policy mode)")

    args = parser.parse_args()

    if args.mode == 'value':
        train_value_rl(args)
    else:
        train_policy_rl(args)


if __name__ == '__main__':
    main()
