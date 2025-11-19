"""
Reinforcement Learning: Train CNN agent with policy gradient (REINFORCE).

Fine-tune supervised pretrained model with RL to potentially surpass teacher performance.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
from tqdm import tqdm
import argparse
import os
from datetime import datetime

from pufferlib.ocean.tetris import tetris
from agents.cnn_agent import TetrisCNN


class RLTrainer:
    """REINFORCE trainer for Tetris CNN agent."""

    def __init__(self, model, device, lr=1e-4, gamma=0.99, entropy_coef=0.01):
        """
        Initialize RL trainer.

        Args:
            model: TetrisCNN model
            device: torch device
            lr: Learning rate
            gamma: Discount factor for returns
            entropy_coef: Entropy regularization coefficient
        """
        self.model = model.to(device)
        self.device = device
        self.gamma = gamma
        self.entropy_coef = entropy_coef

        self.optimizer = optim.Adam(model.parameters(), lr=lr)

        # Episode buffer
        self.reset_episode()

    def reset_episode(self):
        """Reset episode buffer."""
        self.log_probs = []
        self.rewards = []
        self.entropies = []

    def select_action(self, board_empty, board_filled, temperature=1.0):
        """
        Select action and store log probability.

        Args:
            board_empty: (1, 1, H, W) board tensor with piece as empty
            board_filled: (1, 1, H, W) board tensor with piece as filled
            temperature: Sampling temperature

        Returns:
            action: Selected action
        """
        logits = self.model(board_empty, board_filled)
        logits = logits / temperature

        probs = F.softmax(logits, dim=1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()

        # Store for training
        self.log_probs.append(dist.log_prob(action))
        self.entropies.append(dist.entropy())

        return action.item()

    def store_reward(self, reward):
        """Store reward for current timestep."""
        self.rewards.append(reward)

    def compute_returns(self):
        """
        Compute discounted returns for episode.

        Returns:
            returns: List of discounted returns
        """
        returns = []
        R = 0

        # Compute returns backwards
        for reward in reversed(self.rewards):
            R = reward + self.gamma * R
            returns.insert(0, R)

        # Normalize returns for stability
        returns = torch.tensor(returns, device=self.device)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        return returns

    def update(self):
        """
        Update model using REINFORCE.

        Returns:
            loss: Total loss value
        """
        if len(self.rewards) == 0:
            return 0.0

        returns = self.compute_returns()

        policy_loss = []
        for log_prob, R in zip(self.log_probs, returns):
            policy_loss.append(-log_prob * R)

        # Add entropy bonus for exploration
        entropy = torch.stack(self.entropies).mean()

        # Total loss
        loss = torch.stack(policy_loss).sum() - self.entropy_coef * entropy

        # Update
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()


def prepare_board_inputs(obs, device):
    """
    Prepare dual board representation for CNN.

    Args:
        obs: flattened observation array
        device: torch device

    Returns:
        board_empty: (1, 1, H, W) tensor with piece as empty
        board_filled: (1, 1, H, W) tensor with piece as filled
    """
    full_board = obs[0, :200].reshape(20, 10)
    locked = (full_board == 1).astype(np.float32)
    active = (full_board == 2)

    # Board with piece as empty
    board_empty = locked.copy()

    # Board with piece as filled
    board_filled = locked.copy()
    board_filled[active] = 1.0

    # Convert to tensors
    board_empty = torch.FloatTensor(board_empty).unsqueeze(0).unsqueeze(0).to(device)
    board_filled = torch.FloatTensor(board_filled).unsqueeze(0).unsqueeze(0).to(device)

    return board_empty, board_filled


def evaluate_model(model, device, n_episodes=10, deterministic=True):
    """
    Evaluate model performance.

    Args:
        model: TetrisCNN model
        device: torch device
        n_episodes: Number of episodes to evaluate
        deterministic: Use argmax action selection

    Returns:
        metrics: Dict with mean/std reward and length
    """
    env = tetris.Tetris()
    model.eval()

    total_rewards = []
    total_lengths = []

    with torch.no_grad():
        for _ in range(n_episodes):
            obs, _ = env.reset()
            done = False
            episode_reward = 0
            steps = 0

            while not done:
                board_empty, board_filled = prepare_board_inputs(obs, device)
                logits = model(board_empty, board_filled)

                if deterministic:
                    action = logits.argmax(dim=1).item()
                else:
                    probs = F.softmax(logits, dim=1)
                    action = torch.multinomial(probs, num_samples=1).item()

                obs, reward, terminated, truncated, info = env.step([action])
                done = terminated[0] or truncated[0]

                episode_reward += reward[0]
                steps += 1

            total_rewards.append(episode_reward)
            total_lengths.append(steps)

    env.close()

    return {
        'mean_reward': np.mean(total_rewards),
        'std_reward': np.std(total_rewards),
        'mean_length': np.mean(total_lengths),
        'std_length': np.std(total_lengths)
    }


def save_checkpoint(model, optimizer, episode, best_reward, checkpoint_dir, filename):
    """Save RL training checkpoint."""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'episode': episode,
        'best_reward': best_reward,
        'timestamp': datetime.now().isoformat()
    }
    path = os.path.join(checkpoint_dir, filename)
    torch.save(checkpoint, path)
    return path


def load_checkpoint(checkpoint_path, model, optimizer=None, device='cpu'):
    """Load training checkpoint or model weights."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Handle both full checkpoints and model-only weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint
    else:
        # Just model weights
        model.load_state_dict(checkpoint)
        return {'episode': 0, 'best_reward': -float('inf')}


def train_rl(
    model_path=None,
    checkpoint=None,
    n_episodes=1000,
    temperature=1.0,
    lr=1e-4,
    gamma=0.99,
    entropy_coef=0.01,
    device='cpu',
    checkpoint_dir='checkpoints_rl',
    save_frequency=100,
    eval_frequency=50,
    eval_episodes=10,
    line_clear_bonus=50.0,
    credit_window=10,
    credit_decay=0.7
):
    """
    Train model with reinforcement learning.

    Args:
        model_path: Path to pretrained model weights (for starting from supervised)
        checkpoint: Path to RL checkpoint to resume from
        n_episodes: Number of episodes to train
        temperature: Sampling temperature
        lr: Learning rate
        gamma: Discount factor
        entropy_coef: Entropy regularization coefficient
        device: cpu or cuda
        checkpoint_dir: Directory to save checkpoints
        save_frequency: Save checkpoint every N episodes
        eval_frequency: Evaluate model every N episodes
        eval_episodes: Number of episodes for evaluation
        line_clear_bonus: Bonus reward per line cleared (default: 50.0)
        credit_window: Number of past actions to credit (default: 10)
        credit_decay: Exponential decay for backward credit (default: 0.7)
    """
    device = torch.device(device)

    # Create directories
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs('models', exist_ok=True)

    # Initialize model
    model = TetrisCNN(n_rows=20, n_cols=10, n_actions=7).to(device)
    trainer = RLTrainer(model, device, lr=lr, gamma=gamma, entropy_coef=entropy_coef)

    # Training state
    start_episode = 0
    best_reward = -float('inf')

    # Load checkpoint or pretrained model
    if checkpoint:
        print(f"\nLoading RL checkpoint: {checkpoint}")
        ckpt = load_checkpoint(checkpoint, model, trainer.optimizer, device)
        start_episode = ckpt.get('episode', 0) + 1
        best_reward = ckpt.get('best_reward', -float('inf'))
        print(f"Resuming from episode {start_episode}")
        print(f"Best reward so far: {best_reward:.2f}")
    elif model_path:
        print(f"\nLoading pretrained model: {model_path}")
        load_checkpoint(model_path, model, None, device)
        print("Starting RL training from supervised model")
    else:
        print("\nWarning: Training from scratch (no pretrained model)")

    # Training info
    print(f"\n{'='*70}")
    print(f"RL Training Configuration")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Episodes: {n_episodes}")
    print(f"Learning rate: {lr}")
    print(f"Gamma: {gamma}")
    print(f"Entropy coefficient: {entropy_coef}")
    print(f"Temperature: {temperature}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"{'='*70}\n")

    # Track metrics
    episode_rewards = deque(maxlen=100)
    episode_lengths = deque(maxlen=100)
    episode_losses = deque(maxlen=100)

    env = tetris.Tetris()

    for episode in range(start_episode, n_episodes):
        obs, _ = env.reset()
        trainer.reset_episode()

        done = False
        episode_reward = 0
        steps = 0
        prev_lines_cleared = 0

        # Play episode
        model.train()
        while not done:
            # Select action
            board_empty, board_filled = prepare_board_inputs(obs, device)
            action = trainer.select_action(board_empty, board_filled, temperature=temperature)

            # Step environment
            obs, reward, terminated, truncated, info = env.step([action])
            done = terminated[0] or truncated[0]

            # Check if lines were cleared
            current_lines = 0
            if isinstance(info, list) and len(info) > 0 and isinstance(info[0], dict):
                current_lines = info[0].get('lines_cleared', 0)

            lines_just_cleared = max(0, current_lines - prev_lines_cleared)
            prev_lines_cleared = current_lines

            # Apply reward shaping: bonus for line clears, propagated backwards
            shaped_reward = reward[0]
            if lines_just_cleared > 0:
                # Give bonus to this action
                shaped_reward += line_clear_bonus * lines_just_cleared

                # Propagate credit backwards to recent actions with decay
                num_recent = min(credit_window, len(trainer.rewards))
                for i in range(1, num_recent + 1):
                    bonus = line_clear_bonus * lines_just_cleared * (credit_decay ** i)
                    trainer.rewards[-i] += bonus

            # Store reward
            trainer.store_reward(shaped_reward)
            episode_reward += reward[0]  # Track original reward for logging
            steps += 1

        # Update policy
        loss = trainer.update()

        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        episode_losses.append(loss)

        # Logging
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards)
            avg_length = np.mean(episode_lengths)
            avg_loss = np.mean(episode_losses)
            print(f"Episode {episode+1}/{n_episodes}: "
                  f"Reward: {avg_reward:.2f}, Length: {avg_length:.0f}, Loss: {avg_loss:.4f}")

        # Evaluation
        if (episode + 1) % eval_frequency == 0:
            print(f"\nEvaluating at episode {episode+1}...")
            eval_metrics = evaluate_model(model, device, n_episodes=eval_episodes)
            print(f"  Reward: {eval_metrics['mean_reward']:.2f} ± {eval_metrics['std_reward']:.2f}")
            print(f"  Length: {eval_metrics['mean_length']:.0f} ± {eval_metrics['std_length']:.0f}")

            # Save best model
            if eval_metrics['mean_reward'] > best_reward:
                best_reward = eval_metrics['mean_reward']
                save_checkpoint(
                    model, trainer.optimizer, episode, best_reward,
                    checkpoint_dir, 'best_rl.pt'
                )
                print(f"  -> Saved best RL model (reward: {best_reward:.2f})")

        # Save periodic checkpoint
        if (episode + 1) % save_frequency == 0:
            filename = f'checkpoint_rl_ep{episode+1:04d}.pt'
            save_checkpoint(
                model, trainer.optimizer, episode, best_reward,
                checkpoint_dir, filename
            )
            print(f"Checkpoint saved: {filename}")

    env.close()

    # Save final checkpoint
    save_checkpoint(
        model, trainer.optimizer, n_episodes - 1, best_reward,
        checkpoint_dir, 'final_rl.pt'
    )

    # Save final model weights only (for easy loading)
    torch.save(model.state_dict(), 'models/cnn_agent_rl.pt')

    print(f"\n{'='*70}")
    print("RL Training Complete!")
    print(f"{'='*70}")
    print(f"Best reward: {best_reward:.2f}")
    print(f"Checkpoints saved to: {checkpoint_dir}/")
    print(f"Final model saved to: models/cnn_agent_rl.pt")


def main():
    parser = argparse.ArgumentParser(
        description='Train CNN agent with reinforcement learning (REINFORCE)'
    )

    # Model loading
    parser.add_argument('--model', type=str, default=None,
                        help='Path to pretrained model weights (e.g., from supervised training)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to RL checkpoint to resume from')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'],
                        help='Device to use for training')

    # Training schedule
    parser.add_argument('--episodes', type=int, default=1000,
                        help='Number of training episodes')
    parser.add_argument('--eval-frequency', type=int, default=50,
                        help='Evaluate every N episodes')
    parser.add_argument('--eval-episodes', type=int, default=10,
                        help='Number of episodes for evaluation')

    # RL hyperparameters
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor')
    parser.add_argument('--entropy-coef', type=float, default=0.01,
                        help='Entropy regularization coefficient')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature')

    # Checkpointing
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints_rl',
                        help='Directory to save checkpoints')
    parser.add_argument('--save-frequency', type=int, default=100,
                        help='Save checkpoint every N episodes')

    # Reward shaping
    parser.add_argument('--line-clear-bonus', type=float, default=50.0,
                        help='Bonus reward per line cleared')
    parser.add_argument('--credit-window', type=int, default=10,
                        help='Number of past actions to credit for line clears')
    parser.add_argument('--credit-decay', type=float, default=0.7,
                        help='Decay factor for backward credit assignment')

    args = parser.parse_args()

    train_rl(
        model_path=args.model,
        checkpoint=args.checkpoint,
        n_episodes=args.episodes,
        temperature=args.temperature,
        lr=args.lr,
        gamma=args.gamma,
        entropy_coef=args.entropy_coef,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        save_frequency=args.save_frequency,
        eval_frequency=args.eval_frequency,
        eval_episodes=args.eval_episodes,
        line_clear_bonus=args.line_clear_bonus,
        credit_window=args.credit_window,
        credit_decay=args.credit_decay
    )


if __name__ == "__main__":
    main()
