"""
Q-learning fine-tuning for the unified Tetris Q-value agent.

Workflow:
1. Start from a supervised-pretrained model (optional but recommended)
2. Continue training with TD learning using the same line-clear shaped rewards
"""

import argparse
import os
import random
import time
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

from pufferlib.ocean.tetris import tetris
from agents.q_agent import TetrisQNetwork
from utils.rewards import extract_line_clear_reward


def extract_boards(obs_flat, n_rows=20, n_cols=10):
    """Convert flattened env observation into (board_empty, board_filled)."""
    board = obs_flat[: n_rows * n_cols].reshape(n_rows, n_cols)
    locked = (board == 1).astype(np.float32)
    active = (board == 2)

    board_empty = locked
    board_filled = locked.copy()
    board_filled[active] = 1.0
    return board_empty, board_filled


class ReplayBuffer:
    """Simple FIFO replay buffer for experience replay."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def push(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state_e, state_f, actions, rewards, next_e, next_f, dones = zip(*batch)
        return (
            np.stack(state_e),
            np.stack(state_f),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.stack(next_e),
            np.stack(next_f),
            np.array(dones, dtype=np.float32),
        )


class QLearningTrainer:
    """Encapsulates optimization logic for Q-learning with target network."""

    def __init__(
        self,
        model,
        device,
        lr=1e-4,
        gamma=0.99,
        buffer_size=100000,
        min_buffer=2000,
    ):
        self.model = model
        self.target_model = TetrisQNetwork().to(device)
        self.target_model.load_state_dict(model.state_dict())
        self.device = device
        self.gamma = gamma
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.min_buffer = min_buffer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.global_step = 0

    def select_action(self, board_empty, board_filled, epsilon):
        """Epsilon-greedy action selection using current Q-values."""
        if np.random.random() < epsilon:
            return np.random.randint(0, 7)

        board_empty_t = torch.FloatTensor(board_empty).unsqueeze(0).unsqueeze(0).to(self.device)
        board_filled_t = torch.FloatTensor(board_filled).unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            values = self.model(board_empty_t, board_filled_t)
        return int(values.argmax(dim=1).item())

    def store(self, transition):
        self.replay_buffer.push(transition)

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def optimize(self, batch_size):
        """Run one TD update if enough samples are available."""
        if len(self.replay_buffer) < self.min_buffer:
            return None

        state_e, state_f, actions, rewards, next_e, next_f, dones = self.replay_buffer.sample(batch_size)

        state_e = torch.FloatTensor(state_e).unsqueeze(1).to(self.device)
        state_f = torch.FloatTensor(state_f).unsqueeze(1).to(self.device)
        next_e = torch.FloatTensor(next_e).unsqueeze(1).to(self.device)
        next_f = torch.FloatTensor(next_f).unsqueeze(1).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        q_values = self.model(state_e, state_f)
        state_action_values = q_values.gather(1, actions).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_model(next_e, next_f).max(1)[0]
            targets = rewards + self.gamma * (1.0 - dones) * next_q_values

        loss = F.mse_loss(state_action_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.global_step += 1
        return loss.item()


def evaluate_model(model, device, n_episodes=10):
    """Run greedy episodes to gauge performance."""
    env = tetris.Tetris(seed=int(time.time() * 1e6))
    rewards = []

    for _ in range(n_episodes):
        obs, _ = env.reset(seed=int(time.time() * 1e6))
        done = False
        total_reward = 0.0

        while not done:
            board_empty, board_filled = extract_boards(obs[0])
            board_empty_t = torch.FloatTensor(board_empty).unsqueeze(0).unsqueeze(0).to(device)
            board_filled_t = torch.FloatTensor(board_filled).unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                action = model(board_empty_t, board_filled_t).argmax(dim=1).item()

            obs, reward, terminated, truncated, info = env.step([action])
            done = terminated[0] or truncated[0]
            total_reward += extract_line_clear_reward(reward[0])

        rewards.append(total_reward)

    env.close()
    return float(np.mean(rewards)), float(np.std(rewards))


def save_checkpoint(model, target_model, optimizer, episode, best_reward, checkpoint_dir, filename):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'target_state_dict': target_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'episode': episode,
        'best_reward': best_reward,
    }
    path = os.path.join(checkpoint_dir, filename)
    torch.save(checkpoint, path)
    return path


def load_checkpoint(checkpoint_path, model, target_model, optimizer, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    target_model.load_state_dict(checkpoint['target_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint


def train_rl(
    model_path=None,
    checkpoint=None,
    n_episodes=500,
    device='cpu',
    batch_size=128,
    buffer_size=100000,
    min_buffer=2000,
    gamma=0.99,
    lr=1e-4,
    epsilon_start=0.2,
    epsilon_end=0.01,
    epsilon_decay=0.5,
    target_update=1000,
    eval_frequency=25,
    eval_episodes=5,
    checkpoint_dir='checkpoints_rl',
    save_frequency=50,
):
    """Main Q-learning training loop."""

    device = torch.device(device)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs('models', exist_ok=True)

    model = TetrisQNetwork().to(device)
    trainer = QLearningTrainer(
        model,
        device,
        lr=lr,
        gamma=gamma,
        buffer_size=buffer_size,
        min_buffer=min_buffer,
    )

    start_episode = 0
    best_reward = -float('inf')

    if model_path:
        print(f"Loading supervised weights from {model_path}")
        state_dict = torch.load(model_path, map_location=device, weights_only=False)
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            model.load_state_dict(state_dict['model_state_dict'])
        else:
            model.load_state_dict(state_dict)
        trainer.update_target()

    if checkpoint:
        print(f"Resuming from checkpoint {checkpoint}")
        ckpt = load_checkpoint(checkpoint, model, trainer.target_model, trainer.optimizer, device)
        start_episode = ckpt.get('episode', 0) + 1
        best_reward = ckpt.get('best_reward', -float('inf'))

    env = tetris.Tetris(seed=int(time.time() * 1e6))
    global_step = 0

    for episode in range(start_episode, n_episodes):
        obs, _ = env.reset(seed=int(time.time() * 1e6))
        board_empty, board_filled = extract_boards(obs[0])
        done = False
        episode_reward = 0.0

        frac = episode / max(1, n_episodes - 1)
        epsilon = max(epsilon_end, epsilon_start - frac * (epsilon_start - epsilon_end) * epsilon_decay)

        while not done:
            action = trainer.select_action(board_empty, board_filled, epsilon)
            next_obs, reward, terminated, truncated, info = env.step([action])
            done = terminated[0] or truncated[0]
            next_empty, next_filled = extract_boards(next_obs[0])
            reward_value = extract_line_clear_reward(reward[0])

            trainer.store((board_empty, board_filled, action, reward_value, next_empty, next_filled, float(done)))
            loss = trainer.optimize(batch_size)

            board_empty, board_filled = next_empty, next_filled
            episode_reward += reward_value
            global_step += 1

            if global_step % target_update == 0:
                trainer.update_target()

        if (episode + 1) % eval_frequency == 0:
            mean_reward, std_reward = evaluate_model(trainer.model, device, eval_episodes)
            print(f"Episode {episode + 1}/{n_episodes} | Eval reward: {mean_reward:.2f} ± {std_reward:.2f}")
            if mean_reward > best_reward:
                best_reward = mean_reward
                save_checkpoint(trainer.model, trainer.target_model, trainer.optimizer, episode, best_reward, checkpoint_dir, 'best.pt')

        if (episode + 1) % save_frequency == 0:
            save_checkpoint(trainer.model, trainer.target_model, trainer.optimizer, episode, best_reward, checkpoint_dir, f'episode_{episode+1}.pt')

        print(f"Episode {episode + 1}: reward={episode_reward:.2f}, epsilon={epsilon:.3f}, buffer={len(trainer.replay_buffer)}")

    env.close()

    final_path = 'models/q_value_agent_rl.pt'
    torch.save(trainer.model.state_dict(), final_path)
    print(f"Training complete. Final RL model saved to {final_path}")


def main():
    parser = argparse.ArgumentParser(description='RL fine-tuning for the Q-value agent (TD learning)')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to supervised model weights to initialize from')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Resume RL training from checkpoint')
    parser.add_argument('--episodes', type=int, default=500,
                        help='Number of RL episodes to run')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'],
                        help='Torch device to use')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for TD updates')
    parser.add_argument('--buffer-size', type=int, default=100000,
                        help='Replay buffer capacity')
    parser.add_argument('--min-buffer', type=int, default=2000,
                        help='Samples required before training starts')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--epsilon-start', type=float, default=0.2,
                        help='Starting epsilon for exploration')
    parser.add_argument('--epsilon-end', type=float, default=0.01,
                        help='Final epsilon for exploration')
    parser.add_argument('--epsilon-decay', type=float, default=0.5,
                        help='Fraction of training used to decay epsilon')
    parser.add_argument('--target-update', type=int, default=1000,
                        help='Target network update frequency (steps)')
    parser.add_argument('--eval-frequency', type=int, default=25,
                        help='Evaluate model every N episodes')
    parser.add_argument('--eval-episodes', type=int, default=5,
                        help='Episodes per evaluation run')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints_rl',
                        help='Directory for RL checkpoints')
    parser.add_argument('--save-frequency', type=int, default=50,
                        help='Save checkpoint every N episodes')

    args = parser.parse_args()

    train_rl(
        model_path=args.model_path,
        checkpoint=args.checkpoint,
        n_episodes=args.episodes,
        device=args.device,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        min_buffer=args.min_buffer,
        gamma=args.gamma,
        lr=args.lr,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        target_update=args.target_update,
        eval_frequency=args.eval_frequency,
        eval_episodes=args.eval_episodes,
        checkpoint_dir=args.checkpoint_dir,
        save_frequency=args.save_frequency,
    )


if __name__ == '__main__':
    main()
