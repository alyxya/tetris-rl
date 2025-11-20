"""
Train value-based CNN agent using supervised learning with teacher demonstrations.

The agent learns to predict discounted return-to-go for (state, action) pairs by
observing a teacher agent. For each (state, action) pair, the target is the
discounted sum of future rewards from that timestep onwards.

Target for timestep t: V(s_t, a_t) = Σ(γ^i * r_{t+i}) for i=0 to episode end

Key differences from policy-based training:
- Dataset: (state, action, discounted_return) tuples instead of (state, action) pairs
- Loss: MSE on predicted vs actual discounted returns instead of cross-entropy on actions
- Inference: evaluate all actions and pick highest predicted value
- Discount factor (gamma): controls how much we value future vs immediate rewards
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
from agents.value_agent import ValueAgent, TetrisValueCNN


class TetrisValueDataset(Dataset):
    """Dataset of (state, action, reward) tuples."""

    def __init__(self, states_empty, states_filled, actions, rewards):
        self.states_empty = states_empty
        self.states_filled = states_filled
        self.actions = actions
        self.rewards = rewards

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        # Create one-hot encoding for action
        action_onehot = np.zeros(7, dtype=np.float32)
        action_onehot[self.actions[idx]] = 1.0

        return (
            torch.FloatTensor(self.states_empty[idx]),
            torch.FloatTensor(self.states_filled[idx]),
            torch.FloatTensor(action_onehot),
            torch.FloatTensor([self.rewards[idx]])
        )


def collect_data(teacher_agent, student_agent=None, n_episodes=10, student_mix=0.0, gamma=0.95, verbose=True):
    """
    Collect training data using mix of teacher and student agents.

    Args:
        teacher_agent: Teacher agent (e.g., HeuristicAgent) that provides action labels
        student_agent: Student agent (ValueAgent) that can also act (optional)
        n_episodes: Number of episodes to collect
        student_mix: Probability of using student's action vs teacher's action (0.0 = pure teacher, 1.0 = pure student)
        gamma: Discount factor for computing discounted return-to-go (default: 0.95)
        verbose: Print progress

    Returns:
        states_empty: list of boards with piece as empty
        states_filled: list of boards with piece as filled
        actions: list of teacher actions (labels)
        rewards: list of discounted return-to-go values for (state, action) pairs
        episode_rewards: list of episode total rewards

    Note:
        When student_mix > 0, student acts in environment with probability student_mix,
        but teacher's action is still used as the label for learning. This allows the
        student to explore states it would visit while still learning from teacher's policy.
    """
    env = tetris.Tetris()

    states_empty = []
    states_filled = []
    actions = []
    step_rewards = []
    episode_rewards = []

    if verbose:
        if student_mix > 0 and student_agent is not None:
            print(f"Collecting data from {n_episodes} episodes (gamma={gamma}, student_mix={student_mix:.2f})...")
        else:
            print(f"Collecting data from {n_episodes} episodes (gamma={gamma})...")

    for episode in tqdm(range(n_episodes), disable=not verbose):
        obs, _ = env.reset()
        teacher_agent.reset()
        if student_agent is not None:
            student_agent.reset()
        done = False
        episode_reward = 0

        episode_states_empty = []
        episode_states_filled = []
        episode_actions = []
        episode_step_rewards = []

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

            # Decide which action to actually execute in environment
            if student_agent is not None and np.random.random() < student_mix:
                # Student acts (for exploration/on-policy data)
                action_to_execute = student_agent.choose_action(obs[0], deterministic=False)
            else:
                # Teacher acts
                action_to_execute = teacher_action

            # Store state and teacher's action (the label)
            episode_states_empty.append(board_empty)
            episode_states_filled.append(board_filled)
            episode_actions.append(teacher_action)

            # Step environment with chosen action
            obs, reward, terminated, truncated, info = env.step([action_to_execute])
            done = terminated[0] or truncated[0]
            step_reward = reward[0]
            episode_reward += step_reward

            episode_step_rewards.append(step_reward)

        episode_rewards.append(episode_reward)

        # Compute discounted return-to-go for each timestep
        episode_reward_values = []
        for t in range(len(episode_step_rewards)):
            discounted_return = 0.0
            for i, r in enumerate(episode_step_rewards[t:]):
                discounted_return += (gamma ** i) * r
            episode_reward_values.append(discounted_return)

        # Add episode data to dataset
        states_empty.extend(episode_states_empty)
        states_filled.extend(episode_states_filled)
        actions.extend(episode_actions)
        step_rewards.extend(episode_reward_values)

    env.close()

    if verbose:
        print(f"Collected {len(actions)} samples, avg episode reward: {np.mean(episode_rewards):.2f}")
        print(f"Reward stats: mean={np.mean(step_rewards):.2f}, std={np.std(step_rewards):.2f}, "
              f"min={np.min(step_rewards):.2f}, max={np.max(step_rewards):.2f}")

    return states_empty, states_filled, actions, step_rewards, episode_rewards


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

            # Track lines cleared
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
    total_samples = 0

    for board_empty, board_filled, action_onehot, target_values in train_loader:
        board_empty = board_empty.unsqueeze(1).to(device)
        board_filled = board_filled.unsqueeze(1).to(device)
        action_onehot = action_onehot.to(device)
        target_values = target_values.to(device)

        optimizer.zero_grad()
        predicted_values = model(board_empty, board_filled, action_onehot)
        loss = criterion(predicted_values, target_values)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * board_empty.size(0)
        total_samples += board_empty.size(0)

    return total_loss / total_samples


def validate(model, val_loader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for board_empty, board_filled, action_onehot, target_values in val_loader:
            board_empty = board_empty.unsqueeze(1).to(device)
            board_filled = board_filled.unsqueeze(1).to(device)
            action_onehot = action_onehot.to(device)
            target_values = target_values.to(device)

            predicted_values = model(board_empty, board_filled, action_onehot)
            loss = criterion(predicted_values, target_values)

            total_loss += loss.item() * board_empty.size(0)
            total_samples += board_empty.size(0)

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
    gamma=0.95,
    initial_student_mix=0.0,
    final_student_mix=0.5,
    checkpoint=None,
    n_iterations=10,
    episodes_per_iter=20,
    epochs_per_iter=10,
    batch_size=128,
    lr=1e-3,
    device='cpu',
    val_split=0.2,
    checkpoint_dir='checkpoints_value',
    save_frequency=1
):
    """
    Main training loop with optional student exploration.

    Args:
        teacher_type: Type of teacher agent ('heuristic' is default)
        gamma: Discount factor for computing discounted return-to-go (default: 0.95)
        initial_student_mix: Initial probability of student acting (default: 0.0 = pure teacher)
        final_student_mix: Final probability of student acting (default: 0.5 = 50/50 mix)
        checkpoint: Path to checkpoint to resume from (None = start from scratch)
        n_iterations: Number of data collection iterations
        episodes_per_iter: Episodes to collect per iteration
        epochs_per_iter: Training epochs per iteration
        batch_size: Batch size for training
        lr: Learning rate
        device: 'cpu' or 'cuda'
        val_split: Validation split ratio
        checkpoint_dir: Directory to save checkpoints
        save_frequency: Save checkpoint every N iterations

    Note:
        student_mix controls how much the student (value agent) explores vs follows teacher.
        It linearly increases from initial_student_mix to final_student_mix over iterations.
        - 0.0: Pure teacher demonstrations (no exploration)
        - 0.5: 50/50 mix of student and teacher actions
        - 1.0: Pure student exploration (teacher only labels)
    """
    device = torch.device(device)

    # Create directories
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs('models', exist_ok=True)

    # Initialize model
    model = TetrisValueCNN(n_rows=20, n_cols=10, n_actions=7).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    criterion = nn.MSELoss()

    # Initialize agents
    value_agent = ValueAgent(device=str(device))
    value_agent.model = model

    if teacher_type == 'heuristic':
        teacher_agent = HeuristicAgent()
    else:
        raise ValueError(f"Unknown teacher type: {teacher_type}")

    # Aggregate dataset
    all_states_empty = []
    all_states_filled = []
    all_actions = []
    all_rewards = []

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
        print(f"Resuming from iteration {start_iteration}")
        print(f"Best metrics so far: {best_metrics}")

    # Training info
    print(f"\n{'='*70}")
    print(f"Value-Based Training Configuration")
    print(f"{'='*70}")
    print(f"Teacher: {teacher_type}")
    print(f"Gamma (discount factor): {gamma}")
    print(f"Student mix: {initial_student_mix:.2f} -> {final_student_mix:.2f}")
    print(f"Device: {device}")
    print(f"Iterations: {n_iterations}")
    print(f"Episodes per iteration: {episodes_per_iter}")
    print(f"Epochs per iteration: {epochs_per_iter}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"{'='*70}\n")

    # Main training loop
    for iteration in range(start_iteration, n_iterations):
        # Calculate student_mix for this iteration (linear schedule)
        student_mix = initial_student_mix + (final_student_mix - initial_student_mix) * (
            iteration / max(n_iterations - 1, 1)
        )

        print(f"\n{'='*70}")
        print(f"Iteration {iteration + 1}/{n_iterations} (student_mix={student_mix:.2f})")
        print(f"{'='*70}")

        # Collect data with student/teacher mix
        states_empty, states_filled, actions, rewards, episode_rewards = collect_data(
            teacher_agent,
            student_agent=value_agent,
            n_episodes=episodes_per_iter,
            student_mix=student_mix,
            gamma=gamma,
            verbose=True
        )

        # Add to aggregate dataset
        all_states_empty.extend(states_empty)
        all_states_filled.extend(states_filled)
        all_actions.extend(actions)
        all_rewards.extend(rewards)

        print(f"Total dataset size: {len(all_actions):,} samples")

        # Create train/val split
        n_samples = len(all_actions)
        n_val = int(n_samples * val_split)
        indices = np.random.permutation(n_samples)

        train_idx = indices[n_val:]
        val_idx = indices[:n_val]

        train_dataset = TetrisValueDataset(
            [all_states_empty[i] for i in train_idx],
            [all_states_filled[i] for i in train_idx],
            [all_actions[i] for i in train_idx],
            [all_rewards[i] for i in train_idx]
        )

        val_dataset = TetrisValueDataset(
            [all_states_empty[i] for i in val_idx],
            [all_states_filled[i] for i in val_idx],
            [all_actions[i] for i in val_idx],
            [all_rewards[i] for i in val_idx]
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
        eval_metrics = evaluate_agent(value_agent, n_episodes=10)
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
    torch.save(model.state_dict(), 'models/value_agent.pt')

    print(f"\n{'='*70}")
    print("Training Complete!")
    print(f"{'='*70}")
    print(f"Best validation loss: {best_metrics['val_loss']:.4f}")
    print(f"Best evaluation reward: {best_metrics['eval_reward']:.2f}")
    print(f"Total dataset size: {len(all_actions):,} samples")
    print(f"Checkpoints saved to: {checkpoint_dir}/")
    print(f"Final model saved to: models/value_agent.pt")


def main():
    parser = argparse.ArgumentParser(
        description='Train value-based CNN agent with teacher demonstrations'
    )

    # Model and training
    parser.add_argument('--teacher', type=str, default='heuristic',
                        help='Teacher agent type (default: heuristic)')
    parser.add_argument('--gamma', type=float, default=0.95,
                        help='Discount factor for return-to-go (default: 0.95)')
    parser.add_argument('--initial-student-mix', type=float, default=0.0,
                        help='Initial probability of student acting vs teacher (default: 0.0)')
    parser.add_argument('--final-student-mix', type=float, default=0.5,
                        help='Final probability of student acting vs teacher (default: 0.5)')
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

    # Checkpointing
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints_value',
                        help='Directory to save checkpoints')
    parser.add_argument('--save-frequency', type=int, default=1,
                        help='Save checkpoint every N iterations')

    args = parser.parse_args()

    train(
        teacher_type=args.teacher,
        gamma=args.gamma,
        initial_student_mix=args.initial_student_mix,
        final_student_mix=args.final_student_mix,
        checkpoint=args.checkpoint,
        n_iterations=args.iterations,
        episodes_per_iter=args.episodes,
        epochs_per_iter=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        val_split=args.val_split,
        checkpoint_dir=args.checkpoint_dir,
        save_frequency=args.save_frequency
    )


if __name__ == "__main__":
    main()
