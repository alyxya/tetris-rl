"""
Reinforcement Learning: Train CNN agent with policy gradient (REINFORCE).

Starts from supervised pretrained model and fine-tunes with RL.
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


def train_rl(model, device, n_episodes=1000, temperature=1.0,
             save_interval=100, eval_interval=50):
    """
    Train model with reinforcement learning.

    Args:
        model: TetrisCNN model
        device: torch device
        n_episodes: Number of episodes to train
        temperature: Sampling temperature
        save_interval: Save model every N episodes
        eval_interval: Evaluate model every N episodes
    """
    env = tetris.Tetris()
    trainer = RLTrainer(model, device, lr=1e-4, gamma=0.99, entropy_coef=0.01)

    # Track metrics
    episode_rewards = deque(maxlen=100)
    episode_lengths = deque(maxlen=100)

    for episode in tqdm(range(n_episodes), desc="Training RL"):
        obs, _ = env.reset()
        trainer.reset_episode()

        done = False
        episode_reward = 0
        steps = 0

        while not done:
            # Select action
            board_empty, board_filled = prepare_board_inputs(obs, device)
            action = trainer.select_action(board_empty, board_filled, temperature=temperature)

            # Step environment
            obs, reward, terminated, truncated, info = env.step([action])
            done = terminated[0] or truncated[0]

            # Store reward
            trainer.store_reward(reward[0])
            episode_reward += reward[0]
            steps += 1

        # Update policy
        loss = trainer.update()

        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)

        # Logging
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards)
            avg_length = np.mean(episode_lengths)
            print(f"\nEpisode {episode+1}: Avg Reward: {avg_reward:.2f}, "
                  f"Avg Length: {avg_length:.0f}, Loss: {loss:.4f}")

        # Save checkpoint
        if (episode + 1) % save_interval == 0:
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), f'models/cnn_agent_rl_ep{episode+1}.pt')
            print(f"Saved checkpoint at episode {episode+1}")

        # Evaluation
        if (episode + 1) % eval_interval == 0:
            eval_reward, eval_length = evaluate_model(model, device, n_episodes=5)
            print(f"Evaluation: Reward: {eval_reward:.2f}, Length: {eval_length:.0f}")
            model.train()

    env.close()

    # Save final model
    torch.save(model.state_dict(), 'models/cnn_agent_rl_final.pt')
    print("\nRL training complete!")


def evaluate_model(model, device, n_episodes=10, deterministic=True):
    """
    Evaluate model performance.

    Args:
        model: TetrisCNN model
        device: torch device
        n_episodes: Number of episodes to evaluate
        deterministic: Use argmax action selection

    Returns:
        avg_reward: Average episode reward
        avg_length: Average episode length
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

    return np.mean(total_rewards), np.mean(total_lengths)


def main():
    parser = argparse.ArgumentParser(description='Train CNN agent with RL')
    parser.add_argument('--model', type=str, default='models/cnn_agent_best.pt',
                        help='Path to pretrained model')
    parser.add_argument('--episodes', type=int, default=1000,
                        help='Number of training episodes')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'],
                        help='Device to use')
    parser.add_argument('--save-interval', type=int, default=100,
                        help='Save model every N episodes')
    parser.add_argument('--eval-interval', type=int, default=50,
                        help='Evaluate every N episodes')

    args = parser.parse_args()

    # Load pretrained model
    device = torch.device(args.device)
    model = TetrisCNN(n_rows=20, n_cols=10, n_actions=7).to(device)

    if os.path.exists(args.model):
        model.load_state_dict(torch.load(args.model, map_location=device))
        print(f"Loaded pretrained model from {args.model}")
    else:
        print(f"Warning: Model {args.model} not found. Training from scratch.")

    # Train with RL
    print(f"\nStarting RL training on {device}...")
    train_rl(
        model,
        device,
        n_episodes=args.episodes,
        temperature=args.temperature,
        save_interval=args.save_interval,
        eval_interval=args.eval_interval
    )


if __name__ == "__main__":
    main()
