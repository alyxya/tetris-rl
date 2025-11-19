"""
Proximal Policy Optimization (PPO): Train CNN agent with modern RL.

PPO is more sample-efficient than REINFORCE with better credit assignment.
Uses clipped surrogate objective and value function for advantage estimation.
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


class ActorCritic(nn.Module):
    """
    Actor-Critic network for PPO.

    Shares CNN backbone between actor (policy) and critic (value function).
    """

    def __init__(self, n_rows=20, n_cols=10, n_actions=7):
        super().__init__()

        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_actions = n_actions

        # Shared CNN backbone
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.conv_output_size = 64 * n_rows * n_cols

        # Shared feature layer
        self.fc_shared = nn.Linear(self.conv_output_size * 2, 256)

        # Actor head (policy)
        self.fc_actor = nn.Linear(256, 128)
        self.actor_out = nn.Linear(128, n_actions)

        # Critic head (value function)
        self.fc_critic = nn.Linear(256, 128)
        self.critic_out = nn.Linear(128, 1)

        self.dropout = nn.Dropout(0.3)

    def forward_cnn(self, x):
        """Process single board through CNN."""
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        return x

    def forward(self, board_empty, board_filled):
        """
        Forward pass returning both policy logits and value estimate.

        Args:
            board_empty: (batch, 1, H, W) board with piece as empty
            board_filled: (batch, 1, H, W) board with piece as filled

        Returns:
            logits: (batch, n_actions) action logits
            value: (batch, 1) state value estimate
        """
        # Shared CNN features
        features_empty = self.forward_cnn(board_empty)
        features_filled = self.forward_cnn(board_filled)
        features = torch.cat([features_empty, features_filled], dim=1)

        # Shared layer
        shared = F.relu(self.fc_shared(features))
        shared = self.dropout(shared)

        # Actor (policy)
        actor = F.relu(self.fc_actor(shared))
        actor = self.dropout(actor)
        logits = self.actor_out(actor)

        # Critic (value)
        critic = F.relu(self.fc_critic(shared))
        critic = self.dropout(critic)
        value = self.critic_out(critic)

        return logits, value

    def get_action_and_value(self, board_empty, board_filled, action=None):
        """
        Get action, log_prob, entropy, and value.

        Args:
            board_empty: (batch, 1, H, W)
            board_filled: (batch, 1, H, W)
            action: (batch,) optional actions to evaluate

        Returns:
            action: (batch,) sampled or provided actions
            log_prob: (batch,) log probability of actions
            entropy: (batch,) policy entropy
            value: (batch, 1) state value
        """
        logits, value = self.forward(board_empty, board_filled)
        probs = F.softmax(logits, dim=1)
        dist = torch.distributions.Categorical(probs)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, entropy, value


def prepare_board_inputs(obs, device):
    """Prepare dual board representation."""
    full_board = obs[0, :200].reshape(20, 10)
    locked = (full_board == 1).astype(np.float32)
    active = (full_board == 2)

    board_empty = locked.copy()
    board_filled = locked.copy()
    board_filled[active] = 1.0

    board_empty = torch.FloatTensor(board_empty).unsqueeze(0).unsqueeze(0).to(device)
    board_filled = torch.FloatTensor(board_filled).unsqueeze(0).unsqueeze(0).to(device)

    return board_empty, board_filled


class PPOTrainer:
    """PPO trainer with rollout buffer and advantage estimation."""

    def __init__(self, model, device, lr=3e-4, gamma=0.99, gae_lambda=0.95,
                 clip_epsilon=0.2, vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5):
        """
        Initialize PPO trainer.

        Args:
            model: ActorCritic model
            device: torch device
            lr: Learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda for advantage estimation
            clip_epsilon: PPO clipping parameter
            vf_coef: Value function loss coefficient
            ent_coef: Entropy bonus coefficient
            max_grad_norm: Gradient clipping norm
        """
        self.model = model.to(device)
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm

        self.optimizer = optim.Adam(model.parameters(), lr=lr)

        # Rollout buffer
        self.reset_buffer()

    def reset_buffer(self):
        """Reset rollout buffer."""
        self.states_empty = []
        self.states_filled = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []

    def store_transition(self, board_empty, board_filled, action, log_prob, value, reward, done):
        """Store transition in buffer."""
        # Remove batch dimension before storing (boards are (1, 1, H, W), store as (1, H, W))
        self.states_empty.append(board_empty.squeeze(0).cpu().numpy())
        self.states_filled.append(board_filled.squeeze(0).cpu().numpy())
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)

    def compute_gae(self, last_value):
        """
        Compute Generalized Advantage Estimation.

        Args:
            last_value: Value estimate for final state (0 if terminal)

        Returns:
            advantages: GAE advantages
            returns: Discounted returns (targets for value function)
        """
        advantages = []
        gae = 0

        values = self.values + [last_value]

        # Compute advantages backwards
        for t in reversed(range(len(self.rewards))):
            delta = self.rewards[t] + self.gamma * values[t + 1] * (1 - self.dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - self.dones[t]) * gae
            advantages.insert(0, gae)

        advantages = torch.tensor(advantages, device=self.device, dtype=torch.float32)
        returns = advantages + torch.tensor(self.values, device=self.device, dtype=torch.float32)

        return advantages, returns

    def update(self, last_value, n_epochs=4, batch_size=64):
        """
        Update policy using PPO.

        Args:
            last_value: Value estimate for final state
            n_epochs: Number of optimization epochs
            batch_size: Mini-batch size

        Returns:
            metrics: Dict of training metrics
        """
        if len(self.rewards) == 0:
            return {'policy_loss': 0, 'value_loss': 0, 'entropy': 0, 'total_loss': 0}

        # Compute advantages
        advantages, returns = self.compute_gae(last_value)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Convert to tensors
        states_empty = torch.FloatTensor(np.array(self.states_empty)).to(self.device)
        states_filled = torch.FloatTensor(np.array(self.states_filled)).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)

        # Training metrics
        policy_losses = []
        value_losses = []
        entropies = []

        # Multiple epochs of optimization
        for _ in range(n_epochs):
            # Generate random mini-batches
            indices = np.random.permutation(len(self.rewards))

            for start in range(0, len(self.rewards), batch_size):
                end = start + batch_size
                batch_idx = indices[start:end]

                # Get batch
                b_states_empty = states_empty[batch_idx]
                b_states_filled = states_filled[batch_idx]
                b_actions = actions[batch_idx]
                b_old_log_probs = old_log_probs[batch_idx]
                b_advantages = advantages[batch_idx]
                b_returns = returns[batch_idx]

                # Forward pass
                _, new_log_probs, entropy, values = self.model.get_action_and_value(
                    b_states_empty, b_states_filled, b_actions
                )

                # Policy loss (clipped surrogate objective)
                ratio = torch.exp(new_log_probs - b_old_log_probs)
                surr1 = ratio * b_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * b_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(values.squeeze(), b_returns)

                # Entropy bonus
                entropy_loss = entropy.mean()

                # Total loss
                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy_loss

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Track metrics
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy_loss.item())

        return {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy': np.mean(entropies),
            'total_loss': np.mean(policy_losses) + self.vf_coef * np.mean(value_losses)
        }


def evaluate_model(model, device, n_episodes=10):
    """Evaluate model performance."""
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
                logits, _ = model(board_empty, board_filled)
                action = logits.argmax(dim=1).item()

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
    """Save PPO training checkpoint."""
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
    """Load training checkpoint or pretrained weights."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']

        # Check if this is from supervised training (TetrisCNN) or PPO (ActorCritic)
        if 'fc1.weight' in state_dict:
            # This is TetrisCNN from supervised training
            # Only load the CNN backbone (conv layers)
            print("Loading CNN backbone from supervised checkpoint (TetrisCNN)")
            cnn_state = {k: v for k, v in state_dict.items()
                        if k.startswith('conv')}
            model.load_state_dict(cnn_state, strict=False)
            print(f"Loaded {len(cnn_state)} CNN layers")
            return {'episode': 0, 'best_reward': -float('inf')}
        else:
            # This is ActorCritic from PPO training - load everything
            model.load_state_dict(state_dict)
            if optimizer and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            return checkpoint
    else:
        # Raw model weights (no checkpoint wrapper)
        # Check if TetrisCNN or ActorCritic
        if 'fc1.weight' in checkpoint:
            # TetrisCNN weights - only load CNN
            print("Loading CNN backbone from model weights (TetrisCNN)")
            cnn_state = {k: v for k, v in checkpoint.items()
                        if k.startswith('conv')}
            model.load_state_dict(cnn_state, strict=False)
            print(f"Loaded {len(cnn_state)} CNN layers")
        else:
            # ActorCritic weights - load everything
            model.load_state_dict(checkpoint)
        return {'episode': 0, 'best_reward': -float('inf')}


def train_ppo(
    model_path=None,
    checkpoint=None,
    n_episodes=1000,
    steps_per_update=2048,
    n_epochs=4,
    batch_size=64,
    lr=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_epsilon=0.2,
    vf_coef=0.5,
    ent_coef=0.01,
    device='cpu',
    checkpoint_dir='checkpoints_ppo',
    save_frequency=100,
    eval_frequency=50,
    eval_episodes=10
):
    """
    Train model with PPO.

    Args:
        model_path: Path to pretrained model (from supervised training)
        checkpoint: Path to PPO checkpoint to resume from
        n_episodes: Number of episodes to train
        steps_per_update: Number of environment steps before PPO update
        n_epochs: Number of PPO epochs per update
        batch_size: Mini-batch size for PPO updates
        lr: Learning rate
        gamma: Discount factor
        gae_lambda: GAE lambda
        clip_epsilon: PPO clipping parameter
        vf_coef: Value function coefficient
        ent_coef: Entropy coefficient
        device: cpu or cuda
        checkpoint_dir: Directory to save checkpoints
        save_frequency: Save checkpoint every N episodes
        eval_frequency: Evaluate every N episodes
        eval_episodes: Number of episodes for evaluation
    """
    device = torch.device(device)

    # Create directories
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs('models', exist_ok=True)

    # Initialize model
    model = ActorCritic(n_rows=20, n_cols=10, n_actions=7).to(device)
    trainer = PPOTrainer(
        model, device, lr=lr, gamma=gamma, gae_lambda=gae_lambda,
        clip_epsilon=clip_epsilon, vf_coef=vf_coef, ent_coef=ent_coef
    )

    # Training state
    start_episode = 0
    best_reward = -float('inf')

    # Load checkpoint or pretrained model
    if checkpoint:
        print(f"\nLoading PPO checkpoint: {checkpoint}")
        ckpt = load_checkpoint(checkpoint, model, trainer.optimizer, device)
        start_episode = ckpt.get('episode', 0) + 1
        best_reward = ckpt.get('best_reward', -float('inf'))
        print(f"Resuming from episode {start_episode}")
        print(f"Best reward so far: {best_reward:.2f}")
    elif model_path:
        print(f"\nLoading pretrained model: {model_path}")
        load_checkpoint(model_path, model, None, device)
        print("Starting PPO training from pretrained model")
    else:
        print("\nTraining from scratch")

    # Training info
    print(f"\n{'='*70}")
    print(f"PPO Training Configuration")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Episodes: {n_episodes}")
    print(f"Steps per update: {steps_per_update}")
    print(f"PPO epochs: {n_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    print(f"Gamma: {gamma}")
    print(f"GAE lambda: {gae_lambda}")
    print(f"Clip epsilon: {clip_epsilon}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"{'='*70}\n")

    # Track metrics
    episode_rewards = deque(maxlen=100)
    episode_lengths = deque(maxlen=100)

    env = tetris.Tetris()
    global_step = 0

    for episode in range(start_episode, n_episodes):
        obs, _ = env.reset()
        trainer.reset_buffer()

        done = False
        episode_reward = 0
        steps = 0

        # Collect rollout
        model.eval()
        while not done:
            board_empty, board_filled = prepare_board_inputs(obs, device)

            with torch.no_grad():
                action, log_prob, _, value = model.get_action_and_value(board_empty, board_filled)

            obs, reward, terminated, truncated, info = env.step([action.item()])
            done = terminated[0] or truncated[0]

            trainer.store_transition(
                board_empty, board_filled, action.item(),
                log_prob.item(), value.item(), reward[0], done
            )

            episode_reward += reward[0]
            steps += 1
            global_step += 1

            # PPO update every N steps
            if global_step % steps_per_update == 0 and len(trainer.rewards) > 0:
                model.train()
                # Get final value estimate
                if done:
                    last_value = 0
                else:
                    with torch.no_grad():
                        _, _, _, last_value = model.get_action_and_value(board_empty, board_filled)
                        last_value = last_value.item()

                metrics = trainer.update(last_value, n_epochs=n_epochs, batch_size=batch_size)
                trainer.reset_buffer()
                model.eval()

        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)

        # Logging
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards)
            avg_length = np.mean(episode_lengths)
            print(f"Episode {episode+1}/{n_episodes}: "
                  f"Reward: {avg_reward:.2f}, Length: {avg_length:.0f}")

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
                    checkpoint_dir, 'best_ppo.pt'
                )
                print(f"  -> Saved best PPO model (reward: {best_reward:.2f})")

        # Save periodic checkpoint
        if (episode + 1) % save_frequency == 0:
            filename = f'checkpoint_ppo_ep{episode+1:04d}.pt'
            save_checkpoint(
                model, trainer.optimizer, episode, best_reward,
                checkpoint_dir, filename
            )
            print(f"Checkpoint saved: {filename}")

    env.close()

    # Save final checkpoint
    save_checkpoint(
        model, trainer.optimizer, n_episodes - 1, best_reward,
        checkpoint_dir, 'final_ppo.pt'
    )

    # Save final model weights
    torch.save(model.state_dict(), 'models/cnn_agent_ppo.pt')

    print(f"\n{'='*70}")
    print("PPO Training Complete!")
    print(f"{'='*70}")
    print(f"Best reward: {best_reward:.2f}")
    print(f"Checkpoints saved to: {checkpoint_dir}/")
    print(f"Final model saved to: models/cnn_agent_ppo.pt")


def main():
    parser = argparse.ArgumentParser(
        description='Train CNN agent with PPO (Proximal Policy Optimization)'
    )

    # Model loading
    parser.add_argument('--model', type=str, default=None,
                        help='Path to pretrained model (from supervised training)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to PPO checkpoint to resume from')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'],
                        help='Device to use for training')

    # Training schedule
    parser.add_argument('--episodes', type=int, default=1000,
                        help='Number of training episodes')
    parser.add_argument('--steps-per-update', type=int, default=2048,
                        help='Environment steps before PPO update')
    parser.add_argument('--eval-frequency', type=int, default=50,
                        help='Evaluate every N episodes')
    parser.add_argument('--eval-episodes', type=int, default=10,
                        help='Number of episodes for evaluation')

    # PPO hyperparameters
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor')
    parser.add_argument('--gae-lambda', type=float, default=0.95,
                        help='GAE lambda')
    parser.add_argument('--clip-epsilon', type=float, default=0.2,
                        help='PPO clipping parameter')
    parser.add_argument('--vf-coef', type=float, default=0.5,
                        help='Value function coefficient')
    parser.add_argument('--ent-coef', type=float, default=0.01,
                        help='Entropy coefficient')
    parser.add_argument('--n-epochs', type=int, default=4,
                        help='Number of PPO epochs per update')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Mini-batch size for PPO')

    # Checkpointing
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints_ppo',
                        help='Directory to save checkpoints')
    parser.add_argument('--save-frequency', type=int, default=100,
                        help='Save checkpoint every N episodes')

    args = parser.parse_args()

    train_ppo(
        model_path=args.model,
        checkpoint=args.checkpoint,
        n_episodes=args.episodes,
        steps_per_update=args.steps_per_update,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_epsilon=args.clip_epsilon,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        save_frequency=args.save_frequency,
        eval_frequency=args.eval_frequency,
        eval_episodes=args.eval_episodes
    )


if __name__ == "__main__":
    main()
