"""
Train CNN agent using on-policy data collection with teacher supervision.

The student (CNN agent) collects data by playing, and the teacher provides
the correct action labels. This addresses distribution shift by training on
states the student actually encounters.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import argparse
import os
import json
from datetime import datetime

from pufferlib.ocean.tetris import tetris
from agents.heuristic_agent import HeuristicAgent
from agents.cnn_agent import CNNAgent, TetrisCNN


class TetrisDataset(Dataset):
    """Dataset of (state, action) pairs."""

    def __init__(self, states_empty, states_filled, actions):
        self.states_empty = states_empty
        self.states_filled = states_filled
        self.actions = actions

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.states_empty[idx]),
            torch.FloatTensor(self.states_filled[idx]),
            torch.LongTensor([self.actions[idx]])
        )


def collect_data(student_agent, teacher_agent, n_episodes=10, exploration_prob=0.5, verbose=True):
    """
    Collect training data: student acts, teacher labels.

    Args:
        student_agent: CNNAgent that acts in the environment
        teacher_agent: Teacher agent (e.g., HeuristicAgent) that provides labels
        n_episodes: Number of episodes to collect
        exploration_prob: Probability of taking teacher action instead of student action
        verbose: Print progress

    Returns:
        states_empty: list of boards with piece as empty
        states_filled: list of boards with piece as filled
        actions: list of teacher action labels
        episode_rewards: list of episode rewards
    """
    env = tetris.Tetris()

    states_empty = []
    states_filled = []
    actions = []
    episode_rewards = []

    student_agent.model.eval()

    if verbose:
        print(f"Collecting data (exploration={exploration_prob:.2f}) from {n_episodes} episodes...")

    for episode in tqdm(range(n_episodes), disable=not verbose):
        obs, _ = env.reset()
        teacher_agent.reset()
        done = False
        episode_reward = 0

        while not done:
            # Parse observation
            full_board = obs[0, :200].reshape(20, 10)
            locked = (full_board == 1).astype(np.float32)
            active = (full_board == 2)

            board_empty = locked.copy()
            board_filled = locked.copy()
            board_filled[active] = 1.0

            # Get teacher action (always - this is the label)
            teacher_action = teacher_agent.choose_action(obs[0])

            # Store state with teacher label
            states_empty.append(board_empty)
            states_filled.append(board_filled)
            actions.append(teacher_action)

            # Decide which action to take in environment
            if np.random.random() < exploration_prob:
                # Take teacher action (exploration)
                action_to_take = teacher_action
            else:
                # Take student action (on-policy)
                action_to_take = student_agent.choose_action(obs[0], deterministic=False)

            # Step environment
            obs, reward, terminated, truncated, info = env.step([action_to_take])
            done = terminated[0] or truncated[0]
            episode_reward += reward[0]

        episode_rewards.append(episode_reward)

    env.close()

    if verbose:
        print(f"Collected {len(actions)} samples, avg reward: {np.mean(episode_rewards):.2f}")

    return states_empty, states_filled, actions, episode_rewards


def evaluate_agent(agent, n_episodes=10):
    """Evaluate agent performance."""
    env = tetris.Tetris()
    total_rewards = []
    total_lines = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        episode_lines = 0

        while not done:
            action = agent.choose_action(obs[0], deterministic=True)
            obs, reward, terminated, truncated, info = env.step([action])
            done = terminated[0] or truncated[0]
            episode_reward += reward[0]

            # Track lines cleared (info is a list, one per env)
            if isinstance(info, list) and len(info) > 0:
                episode_lines = info[0].get('lines_cleared', 0) if isinstance(info[0], dict) else 0

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
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for board_empty, board_filled, actions in train_loader:
        board_empty = board_empty.unsqueeze(1).to(device)
        board_filled = board_filled.unsqueeze(1).to(device)
        actions = actions.squeeze(1).to(device)

        optimizer.zero_grad()
        logits = model(board_empty, board_filled)
        loss = criterion(logits, actions)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * actions.size(0)
        _, predicted = logits.max(1)
        total_correct += predicted.eq(actions).sum().item()
        total_samples += actions.size(0)

    return total_loss / total_samples, 100.0 * total_correct / total_samples


def validate(model, val_loader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for board_empty, board_filled, actions in val_loader:
            board_empty = board_empty.unsqueeze(1).to(device)
            board_filled = board_filled.unsqueeze(1).to(device)
            actions = actions.squeeze(1).to(device)

            logits = model(board_empty, board_filled)
            loss = criterion(logits, actions)

            total_loss += loss.item() * actions.size(0)
            _, predicted = logits.max(1)
            total_correct += predicted.eq(actions).sum().item()
            total_samples += actions.size(0)

    return total_loss / total_samples, 100.0 * total_correct / total_samples


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
    initial_exploration=0.9,
    final_exploration=0.1,
    checkpoint_dir='checkpoints',
    save_frequency=1
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
        initial_exploration: Starting exploration probability (teacher action rate)
        final_exploration: Final exploration probability
        checkpoint_dir: Directory to save checkpoints
        save_frequency: Save checkpoint every N iterations
    """
    device = torch.device(device)

    # Create directories
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs('models', exist_ok=True)

    # Initialize model
    model = TetrisCNN(n_rows=20, n_cols=10, n_actions=7).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )
    criterion = nn.CrossEntropyLoss()

    # Initialize agents
    student_agent = CNNAgent(device=str(device))
    student_agent.model = model

    if teacher_type == 'heuristic':
        teacher_agent = HeuristicAgent()
    else:
        raise ValueError(f"Unknown teacher type: {teacher_type}")

    # Aggregate dataset
    all_states_empty = []
    all_states_filled = []
    all_actions = []

    # Training state
    start_iteration = 0
    best_metrics = {
        'val_acc': 0.0,
        'eval_reward': -float('inf')
    }

    # Load checkpoint if provided
    if checkpoint:
        print(f"\nLoading checkpoint: {checkpoint}")
        ckpt = load_checkpoint(checkpoint, model, optimizer, scheduler, device)
        start_iteration = ckpt['iteration'] + 1
        best_metrics = ckpt.get('best_metrics', best_metrics)
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
    print(f"Exploration: {initial_exploration:.2f} -> {final_exploration:.2f}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"{'='*70}\n")

    # Main training loop
    for iteration in range(start_iteration, n_iterations):
        # Calculate exploration probability (linear decay)
        exploration = initial_exploration + (final_exploration - initial_exploration) * (
            iteration / max(n_iterations - 1, 1)
        )

        print(f"\n{'='*70}")
        print(f"Iteration {iteration + 1}/{n_iterations} (exploration={exploration:.2f})")
        print(f"{'='*70}")

        # Collect data using current student policy
        states_empty, states_filled, actions, episode_rewards = collect_data(
            student_agent, teacher_agent,
            n_episodes=episodes_per_iter,
            exploration_prob=exploration,
            verbose=True
        )

        # Add to aggregate dataset
        all_states_empty.extend(states_empty)
        all_states_filled.extend(states_filled)
        all_actions.extend(actions)

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
            [all_actions[i] for i in train_idx]
        )

        val_dataset = TetrisDataset(
            [all_states_empty[i] for i in val_idx],
            [all_states_filled[i] for i in val_idx],
            [all_actions[i] for i in val_idx]
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
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_acc = validate(model, val_loader, criterion, device)

            print(f"  Epoch {epoch+1}/{epochs_per_iter}: "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

            # Save best validation model
            if val_acc > best_metrics['val_acc']:
                best_metrics['val_acc'] = val_acc
                save_checkpoint(
                    model, optimizer, scheduler, iteration, epoch,
                    best_metrics, len(all_actions), checkpoint_dir, 'best_val.pt'
                )
                print(f"    -> Saved best validation model (val_acc: {val_acc:.2f}%)")

        # Step scheduler based on validation accuracy
        scheduler.step(val_acc)

        # Evaluate agent performance
        print("\nEvaluating agent performance...")
        eval_metrics = evaluate_agent(student_agent, n_episodes=10)
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
    torch.save(model.state_dict(), 'models/cnn_agent.pt')

    print(f"\n{'='*70}")
    print("Training Complete!")
    print(f"{'='*70}")
    print(f"Best validation accuracy: {best_metrics['val_acc']:.2f}%")
    print(f"Best evaluation reward: {best_metrics['eval_reward']:.2f}")
    print(f"Total dataset size: {len(all_actions):,} samples")
    print(f"Checkpoints saved to: {checkpoint_dir}/")
    print(f"Final model saved to: models/cnn_agent.pt")


def main():
    parser = argparse.ArgumentParser(
        description='Train CNN agent with on-policy data collection'
    )

    # Model and training
    parser.add_argument('--teacher', type=str, default='heuristic',
                        help='Teacher agent type (default: heuristic)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'],
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

    # Exploration schedule
    parser.add_argument('--initial-exploration', type=float, default=0.9,
                        help='Initial exploration probability (teacher action rate)')
    parser.add_argument('--final-exploration', type=float, default=0.1,
                        help='Final exploration probability')

    # Checkpointing
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--save-frequency', type=int, default=1,
                        help='Save checkpoint every N iterations')

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
        initial_exploration=args.initial_exploration,
        final_exploration=args.final_exploration,
        checkpoint_dir=args.checkpoint_dir,
        save_frequency=args.save_frequency
    )


if __name__ == "__main__":
    main()
