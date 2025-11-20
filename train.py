"""
Supervised pretraining for the unified Q-value agent using teacher supervision.

The teacher drives gameplay (with optional random perturbations) and labels each
state, producing a large imitation dataset across diverse board configurations.
Targets are discounted (gamma≈0.99) line-clear returns so that the network
directly regresses future discounted reward instead of a categorical action
distribution.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import argparse
import os
import time
from datetime import datetime

from pufferlib.ocean.tetris import tetris
from agents.heuristic_agent import HeuristicAgent
from agents.q_agent import QValueAgent, TetrisQNetwork
from utils.rewards import (
    extract_line_clear_reward,
    compute_discounted_returns,
    count_lines_cleared,
)


class TetrisDataset(Dataset):
    """Dataset of (state, action, q-value) tuples."""

    def __init__(self, states_empty, states_filled, actions, q_values):
        self.states_empty = states_empty
        self.states_filled = states_filled
        self.actions = actions
        self.q_values = q_values

    def __len__(self):
        return len(self.q_values)

    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.states_empty[idx]),
            torch.FloatTensor(self.states_filled[idx]),
            torch.LongTensor([self.actions[idx]]),
            torch.FloatTensor([self.q_values[idx]])
        )


def collect_data(
    teacher_agent,
    n_episodes=10,
    random_prob=0.1,
    discount=0.99,
    height_penalty_weight=0.001,
    verbose=True,
    debug_qvalues=False,
):
    """
    Collect training data by letting the teacher (with optional random noise) play.

    Args:
        teacher_agent: Teacher agent (e.g., HeuristicAgent) that provides labels
        n_episodes: Number of episodes to collect
        random_prob: Probability of forcing a random action for extra coverage
        discount: Discount factor for returns (default: 0.99)
        height_penalty_weight: Penalty weight per unit height for movements (default: 0.001)
        verbose: Print progress
        debug_qvalues: Print detailed Q-value computation for first episode

    Returns:
        states_empty: list of boards with piece as empty
        states_filled: list of boards with piece as filled
        actions: list of teacher action labels
        q_values: list of discounted returns (rewards - height penalties)
        episode_rewards: list of per-episode net rewards
    """
    env = tetris.Tetris(seed=int(time.time() * 1e6))
    n_rows = env.n_rows
    n_cols = env.n_cols
    board_size = n_rows * n_cols

    states_empty = []
    states_filled = []
    actions = []
    q_targets = []
    episode_rewards = []

    if verbose:
        teacher_share = 1.0 - random_prob
        print(
            f"Collecting data (teacher~{teacher_share:.2f}, random={random_prob:.2f}) "
            f"from {n_episodes} episodes..."
        )

    # Movement actions that incur height-based penalties (left, right, rotate)
    penalty_actions = {1, 2, 3}

    for episode in tqdm(range(n_episodes), disable=not verbose):
        obs, _ = env.reset(seed=int(time.time() * 1e6))
        teacher_agent.reset()
        done = False
        episode_reward = 0
        episode_states_empty = []
        episode_states_filled = []
        episode_actions = []
        episode_net_rewards = []  # Rewards minus height penalties

        while not done:
            # Parse observation
            full_board = obs[0, :board_size].reshape(n_rows, n_cols)
            locked = (full_board == 1).astype(np.float32)
            active = (full_board == 2)

            board_empty = locked.copy()
            board_filled = locked.copy()
            board_filled[active] = 1.0
            prev_board = full_board.copy()

            # Get teacher action (always - this is the label)
            teacher_action = teacher_agent.choose_action(obs[0])

            # Store state (labels decided after knowing executed action)
            episode_states_empty.append(board_empty)
            episode_states_filled.append(board_filled)

            # Decide which action to take in environment
            if np.random.random() < random_prob:
                action_to_take = np.random.randint(0, 7)
            else:
                action_to_take = teacher_action

            # Step environment
            next_obs, reward, terminated, truncated, info = env.step([action_to_take])
            done = terminated[0] or truncated[0]

            # Extract line clear rewards
            next_board = next_obs[0, :board_size].reshape(n_rows, n_cols)
            step_reward = extract_line_clear_reward(prev_board, next_board)

            # Compute height-based penalty for movement actions
            # Height = row index of topmost active cell (0 = top row)
            # Add 1 so even top row (row 0) has a small penalty
            step_penalty = 0.0
            if action_to_take in penalty_actions and np.any(active):
                active_rows = np.where(active)[0]
                height_from_top = np.min(active_rows)  # Smaller row = higher up
                step_penalty = height_penalty_weight * (height_from_top + 1)

            # Net reward for this step
            net_reward = step_reward - step_penalty
            episode_reward += net_reward
            episode_net_rewards.append(net_reward)
            episode_actions.append(action_to_take)

            obs = next_obs

        episode_rewards.append(episode_reward)

        # Compute discounted returns from net rewards (reward - penalty already combined)
        q_values_episode = compute_discounted_returns(episode_net_rewards, gamma=discount)

        # Print detailed Q-value info for first episode (if debug flag is set)
        if episode == 0 and debug_qvalues:
            print(f"\n{'='*80}")
            print(f"Q-value Computation Details (Episode 1, first 30 steps)")
            print(f"{'='*80}")
            print(f"Total steps: {len(q_values_episode)}")
            print(f"Height penalty weight: {height_penalty_weight}")
            print()
            print(f"{'Step':>4} | {'Action':>6} | {'Net Reward':>10} | {'Q-Value':>8}")
            print("-" * 50)
            action_names = ["noop", "left", "right", "rotate", "soft", "hard", "hold"]
            for i in range(min(30, len(q_values_episode))):
                action_name = action_names[episode_actions[i]]
                print(f"{i:4d} | {action_name:>6} | {episode_net_rewards[i]:10.4f} | {q_values_episode[i]:8.4f}")
            print()

        states_empty.extend(episode_states_empty)
        states_filled.extend(episode_states_filled)
        actions.extend(episode_actions)
        q_targets.extend(q_values_episode)

    env.close()

    if verbose:
        print(f"Collected {len(actions)} samples, avg reward: {np.mean(episode_rewards):.2f}")

    return states_empty, states_filled, actions, q_targets, episode_rewards


def evaluate_agent(agent, n_episodes=10, height_penalty_weight=0.001):
    """Evaluate agent performance with height-based penalty system."""
    env = tetris.Tetris(seed=int(time.time() * 1e6))
    n_rows = env.n_rows
    n_cols = env.n_cols
    board_size = n_rows * n_cols
    total_rewards = []
    total_lines = []
    penalty_actions = {1, 2, 3}

    for _ in range(n_episodes):
        obs, _ = env.reset(seed=int(time.time() * 1e6))
        done = False
        episode_reward = 0
        episode_lines = 0

        while not done:
            full_board = obs[0, :board_size].reshape(n_rows, n_cols)
            prev_board = full_board.copy()
            active = (full_board == 2)

            action = agent.choose_action(obs[0], deterministic=True)
            next_obs, reward, terminated, truncated, info = env.step([action])
            done = terminated[0] or truncated[0]

            next_board = next_obs[0, :board_size].reshape(n_rows, n_cols)
            step_reward = extract_line_clear_reward(prev_board, next_board)

            # Height-based penalty
            step_penalty = 0.0
            if action in penalty_actions and np.any(active):
                active_rows = np.where(active)[0]
                height_from_top = np.min(active_rows)
                step_penalty = height_penalty_weight * (height_from_top + 1)

            lines_cleared = count_lines_cleared(prev_board, next_board)

            episode_reward += step_reward - step_penalty
            episode_lines += lines_cleared
            obs = next_obs

        total_rewards.append(episode_reward)
        total_lines.append(episode_lines)

    env.close()
    return {
        'mean_reward': np.mean(total_rewards),
        'std_reward': np.std(total_rewards),
        'mean_lines': np.mean(total_lines),
        'std_lines': np.std(total_lines)
    }


def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch of Q-value regression with teacher actions."""
    model.train()
    total_loss = 0.0
    total_samples = 0

    for board_empty, board_filled, actions, q_targets in train_loader:
        board_empty = board_empty.unsqueeze(1).to(device)
        board_filled = board_filled.unsqueeze(1).to(device)
        actions = actions.squeeze(1).to(device)
        q_targets = q_targets.squeeze(1).to(device)

        optimizer.zero_grad()
        q_values = model(board_empty, board_filled)
        selected_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = criterion(selected_q, q_targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * actions.size(0)
        total_samples += actions.size(0)

    return total_loss / total_samples


def validate(model, val_loader, criterion, device):
    """Validate Q-value regression performance."""
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for board_empty, board_filled, actions, q_targets in val_loader:
            board_empty = board_empty.unsqueeze(1).to(device)
            board_filled = board_filled.unsqueeze(1).to(device)
            actions = actions.squeeze(1).to(device)
            q_targets = q_targets.squeeze(1).to(device)

            q_values = model(board_empty, board_filled)
            selected_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
            loss = criterion(selected_q, q_targets)

            total_loss += loss.item() * actions.size(0)
            total_samples += actions.size(0)

    return total_loss / total_samples


def save_checkpoint(model, optimizer, scheduler, iteration, epoch, best_metrics,
                     dataset_sizes, checkpoint_dir, filename):
    """Save training checkpoint."""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'iteration': iteration,
        'epoch': epoch,
        'best_metrics': best_metrics,
        'dataset_sizes': dataset_sizes,
        'timestamp': datetime.now().isoformat()
    }
    path = os.path.join(checkpoint_dir, filename)
    torch.save(checkpoint, path)
    return path


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, device='cpu'):
    """Load training checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return checkpoint


def train(
    teacher_type='heuristic',
    checkpoint=None,
    n_iterations=10,
    episodes_per_iter=20,
    epochs_per_iter=10,
    batch_size=128,
    lr=1e-3,
    device='cpu',
    val_split=0.2,
    random_action_prob=0.1,
    discount=0.99,
    height_penalty_weight=0.001,
    checkpoint_dir='checkpoints',
    save_frequency=1,
    debug_qvalues=False
):
    """
    Main training loop.

    Args:
        teacher_type: Type of teacher agent ('heuristic' is default)
        checkpoint: Path to checkpoint to resume from (None = start from scratch)
        n_iterations: Number of data collection iterations
        episodes_per_iter: Episodes to collect per iteration
        epochs_per_iter: Training epochs per iteration
        batch_size: Batch size for training
        lr: Learning rate
        device: 'cpu' or 'cuda'
        val_split: Validation split ratio
        random_action_prob: Probability of forcing a random environment action
        discount: Discount factor for returns (default: 0.99)
        height_penalty_weight: Penalty weight per unit height for movements (default: 0.001)
        checkpoint_dir: Directory to save checkpoints
        save_frequency: Save checkpoint every N iterations
        debug_qvalues: Print detailed Q-value computation table (default: False)
    """
    device = torch.device(device)

    # Create directories
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs('models', exist_ok=True)

    # Initialize model
    model = TetrisQNetwork(n_rows=20, n_cols=10, n_actions=7).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    criterion = nn.MSELoss()

    # Initialize agents
    eval_agent = QValueAgent(device=str(device))
    eval_agent.model = model

    if teacher_type == 'heuristic':
        teacher_agent = HeuristicAgent()
    else:
        raise ValueError(f"Unknown teacher type: {teacher_type}")

    # Aggregate dataset
    all_states_empty = []
    all_states_filled = []
    all_actions = []
    all_q_values = []

    # Training state
    start_iteration = 0
    best_metrics = {
        'val_loss': float('inf'),
        'eval_reward': -float('inf')
    }

    # Load checkpoint if provided
    if checkpoint:
        print(f"\nLoading checkpoint: {checkpoint}")
        ckpt = load_checkpoint(checkpoint, model, optimizer, scheduler, device)
        start_iteration = ckpt['iteration'] + 1
        best_metrics = ckpt.get('best_metrics', best_metrics)
        if 'val_loss' not in best_metrics:
            best_metrics['val_loss'] = float('inf')
        print(f"Resuming from iteration {start_iteration}")
        print(f"Best metrics so far: {best_metrics}")

    # Training info
    print(f"\n{'='*70}")
    print(f"Training Configuration")
    print(f"{'='*70}")
    print(f"Teacher: {teacher_type}")
    print(f"Device: {device}")
    print(f"Iterations: {n_iterations}")
    print(f"Episodes per iteration: {episodes_per_iter}")
    print(f"Epochs per iteration: {epochs_per_iter}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    print(f"Random action prob: {random_action_prob:.2f}")
    print(f"Discount factor: {discount:.2f}")
    print(f"Height penalty weight: {height_penalty_weight:.4f}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"{'='*70}\n")

    # Main training loop
    for iteration in range(start_iteration, n_iterations):
        print(f"\n{'='*70}")
        print(f"Iteration {iteration + 1}/{n_iterations}")
        print(f"{'='*70}")

        # Collect data using teacher policy + random noise
        states_empty, states_filled, actions, q_values, _ = collect_data(
            teacher_agent,
            n_episodes=episodes_per_iter,
            random_prob=random_action_prob,
            discount=discount,
            height_penalty_weight=height_penalty_weight,
            verbose=True,
            debug_qvalues=debug_qvalues
        )

        # Add to aggregate dataset
        all_states_empty.extend(states_empty)
        all_states_filled.extend(states_filled)
        all_actions.extend(actions)
        all_q_values.extend(q_values)

        print(f"Total dataset size: {len(all_actions):,} samples")

        # Create train/val split
        n_samples = len(all_actions)
        n_val = int(n_samples * val_split)
        indices = np.random.permutation(n_samples)

        train_idx = indices[n_val:]
        val_idx = indices[:n_val]

        train_dataset = TetrisDataset(
            [all_states_empty[i] for i in train_idx],
            [all_states_filled[i] for i in train_idx],
            [all_actions[i] for i in train_idx],
            [all_q_values[i] for i in train_idx]
        )

        val_dataset = TetrisDataset(
            [all_states_empty[i] for i in val_idx],
            [all_states_filled[i] for i in val_idx],
            [all_actions[i] for i in val_idx],
            [all_q_values[i] for i in val_idx]
        )

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )

        # Train for several epochs
        print(f"\nTraining for {epochs_per_iter} epochs...")
        for epoch in range(epochs_per_iter):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
            val_loss = validate(model, val_loader, criterion, device)

            print(f"  Epoch {epoch+1}/{epochs_per_iter}: "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            # Save best validation model
            if val_loss < best_metrics['val_loss']:
                best_metrics['val_loss'] = val_loss
                save_checkpoint(
                    model, optimizer, scheduler, iteration, epoch,
                    best_metrics, len(all_actions), checkpoint_dir, 'best_val.pt'
                )
                print(f"    -> Saved best validation model (val_loss: {val_loss:.4f})")

        # Step scheduler based on validation loss
        scheduler.step(val_loss)

        # Evaluate agent performance
        print("\nEvaluating agent performance...")
        eval_metrics = evaluate_agent(eval_agent, n_episodes=10)
        print(f"  Reward: {eval_metrics['mean_reward']:.2f} ± {eval_metrics['std_reward']:.2f}")
        print(f"  Lines: {eval_metrics['mean_lines']:.2f} ± {eval_metrics['std_lines']:.2f}")

        # Save best performance model
        if eval_metrics['mean_reward'] > best_metrics['eval_reward']:
            best_metrics['eval_reward'] = eval_metrics['mean_reward']
            save_checkpoint(
                model, optimizer, scheduler, iteration, epochs_per_iter - 1,
                best_metrics, len(all_actions), checkpoint_dir, 'best_performance.pt'
            )
            print(f"  -> Saved best performance model (reward: {eval_metrics['mean_reward']:.2f})")

        # Save periodic checkpoint
        if (iteration + 1) % save_frequency == 0:
            filename = f'checkpoint_iter{iteration+1:03d}.pt'
            save_checkpoint(
                model, optimizer, scheduler, iteration, epochs_per_iter - 1,
                best_metrics, len(all_actions), checkpoint_dir, filename
            )
            print(f"\nCheckpoint saved: {filename}")

    # Save final checkpoint
    save_checkpoint(
        model, optimizer, scheduler, n_iterations - 1, epochs_per_iter - 1,
        best_metrics, len(all_actions), checkpoint_dir, 'final.pt'
    )

    # Save final model weights only (for easy loading)
    final_model_path = 'models/q_value_agent.pt'
    torch.save(model.state_dict(), final_model_path)

    print(f"\n{'='*70}")
    print("Training Complete!")
    print(f"{'='*70}")
    best_val_loss = best_metrics.get('val_loss', float('inf'))
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best evaluation reward: {best_metrics['eval_reward']:.2f}")
    print(f"Total dataset size: {len(all_actions):,} samples")
    print(f"Checkpoints saved to: {checkpoint_dir}/")
    print(f"Final model saved to: {final_model_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Supervised pretraining for the unified Q-value agent'
    )

    # Model and training
    parser.add_argument('--teacher', type=str, default='heuristic',
                        help='Teacher agent type (default: heuristic)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda', 'mps'],
                        help='Device to use for training')

    # Training schedule
    parser.add_argument('--iterations', type=int, default=10,
                        help='Number of data collection iterations')
    parser.add_argument('--episodes', type=int, default=20,
                        help='Episodes to collect per iteration')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Training epochs per iteration')

    # Hyperparameters
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--val-split', type=float, default=0.2,
                        help='Validation split ratio')
    parser.add_argument('--discount', type=float, default=0.99,
                        help='Discount factor for returns')
    parser.add_argument('--height-penalty-weight', type=float, default=0.001,
                        help='Penalty weight per unit height for movement actions')

    # Data diversity
    parser.add_argument('--random-action-prob', type=float, default=0.1,
                        help='Probability of forcing a random action during data collection')

    # Checkpointing
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--save-frequency', type=int, default=1,
                        help='Save checkpoint every N iterations')

    # Debug options
    parser.add_argument('--debug-qvalues', action='store_true',
                        help='Print detailed Q-value computation table for first episode')

    args = parser.parse_args()

    train(
        teacher_type=args.teacher,
        checkpoint=args.checkpoint,
        n_iterations=args.iterations,
        episodes_per_iter=args.episodes,
        epochs_per_iter=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        val_split=args.val_split,
        random_action_prob=args.random_action_prob,
        discount=args.discount,
        height_penalty_weight=args.height_penalty_weight,
        checkpoint_dir=args.checkpoint_dir,
        save_frequency=args.save_frequency,
        debug_qvalues=args.debug_qvalues
    )


if __name__ == "__main__":
    main()
