"""
Reinforcement learning for Tetris agents.

Training mode:
- Value network: Q-learning with heuristic rewards
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

from model import ValueNetwork
from value_agent import ValueAgent
from reward_utils import compute_lines_cleared, compute_simple_reward, compute_shaped_reward


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


def train_value_rl(args):
    """Train value network with Q-learning using shaped rewards."""
    reward_type = "Shaped" if args.shaped_rewards else "Simple"
    print(f"Training Value Network with Q-Learning ({reward_type} Rewards)")
    print("=" * 50)

    device = torch.device(args.device)
    env = tetris.Tetris()

    # Create online and target networks (6 actions - no HOLD)
    online_net = ValueNetwork(n_rows=20, n_cols=10, n_actions=6).to(device)
    target_net = ValueNetwork(n_rows=20, n_cols=10, n_actions=6).to(device)

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
    epsilon = args.epsilon_start
    epsilon_decay = (args.epsilon_start - args.epsilon_end) / args.num_episodes

    # Temperature decay
    temperature = args.temperature_start if args.temperature_start is not None else 0.0
    temperature_end = args.temperature_end if args.temperature_end is not None else 0.0
    temperature_decay = (temperature - temperature_end) / args.num_episodes

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

            # Epsilon-greedy action selection with temperature
            action = agent.choose_action(obs_single, epsilon=epsilon, temperature=temperature)

            # Take step
            next_obs, _, terminated, truncated, _ = env.step([action])
            done = terminated[0] or truncated[0]

            # Compute reward
            if done:
                # No reward on death (board state invalid)
                reward = 0.0
                # Parse next state for storing in buffer
                next_obs_single = next_obs[0] if len(next_obs.shape) > 1 else next_obs
                _, next_locked, next_active = agent.parse_observation(next_obs_single)
            else:
                # Parse next state
                next_obs_single = next_obs[0] if len(next_obs.shape) > 1 else next_obs
                _, next_locked, next_active = agent.parse_observation(next_obs_single)
                lines_cleared = compute_lines_cleared(locked, active, next_locked)

                if args.shaped_rewards:
                    reward = compute_shaped_reward(board_empty, next_locked, lines_cleared)
                else:
                    reward = compute_simple_reward(lines_cleared)

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

        # Decay epsilon and temperature
        epsilon = max(args.epsilon_end, epsilon - epsilon_decay)
        temperature = max(temperature_end, temperature - temperature_decay)

        # Logging
        if episode % 10 == 0:
            print(f"\nEpisode {episode} - Steps: {steps}, Reward: {total_reward:.2f}, Epsilon: {epsilon:.3f}, Temp: {temperature:.3f}")

        # Save model at regular intervals
        if episode % args.save_interval == 0 and episode > 0:
            output_dir = os.path.dirname(args.output)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            torch.save(online_net.state_dict(), args.output)
            print(f"Saved model at episode {episode}")

    # Save final model
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    torch.save(online_net.state_dict(), args.output)
    print(f"\nTraining complete! Final model saved to {args.output}")


def main():
    parser = argparse.ArgumentParser(description="RL training for Tetris")
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
    parser.add_argument('--save-interval', type=int, default=20,
                        help="Save model every N episodes")

    # Value-specific args
    parser.add_argument('--buffer-size', type=int, default=10000,
                        help="Replay buffer size (value mode)")
    parser.add_argument('--batch-size', type=int, default=256,
                        help="Batch size (value mode)")
    parser.add_argument('--epsilon-start', type=float, default=0.2,
                        help="Initial epsilon (value mode)")
    parser.add_argument('--epsilon-end', type=float, default=0.01,
                        help="Final epsilon (value mode)")
    parser.add_argument('--target-update', type=int, default=2,
                        help="Target network update frequency (value mode)")
    parser.add_argument('--temperature-start', type=float, default=None,
                        help="Initial temperature for Boltzmann exploration (default: 0.0)")
    parser.add_argument('--temperature-end', type=float, default=None,
                        help="Final temperature for Boltzmann exploration (default: 0.0)")
    parser.add_argument('--shaped-rewards', action='store_true',
                        help="Use shaped rewards instead of simple line-clear rewards")

    args = parser.parse_args()

    train_value_rl(args)


if __name__ == '__main__':
    main()
