"""
Iterative (Dataset Aggregation): Train CNN agent with interactive expert feedback.

Iterative addresses distribution shift by:
1. CNN agent acts in environment (on-policy data collection)
2. Expert (heuristic agent) provides labels for these states
3. CNN learns from its own trajectory distribution

This helps the model learn to recover from its own mistakes.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import argparse
import os

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


def collect_iterative_data(cnn_agent, expert_agent, n_episodes=10, beta=0.5, verbose=True):
    """
    Collect data using Iterative: CNN acts, expert labels.

    Args:
        cnn_agent: CNNAgent that will act in environment
        expert_agent: HeuristicAgent that provides expert labels
        n_episodes: Number of episodes to collect
        beta: Probability of taking expert action (for exploration)
              1.0 = pure expert, 0.0 = pure CNN
        verbose: Print progress

    Returns:
        states_empty: list of boards with piece as empty
        states_filled: list of boards with piece as filled
        actions: list of expert actions for these states
    """
    env = tetris.Tetris()

    states_empty = []
    states_filled = []
    actions = []

    cnn_agent.model.eval()  # Eval mode for collection

    if verbose:
        print(f"Collecting Iterative data (beta={beta}) from {n_episodes} episodes...")

    for episode in tqdm(range(n_episodes), disable=not verbose):
        obs, _ = env.reset()
        expert_agent.reset()
        done = False
        steps = 0

        while not done:
            # Parse observation
            full_board = obs[0, :200].reshape(20, 10)
            locked = (full_board == 1).astype(np.float32)
            active = (full_board == 2)

            board_empty = locked.copy()
            board_filled = locked.copy()
            board_filled[active] = 1.0

            # Get expert action for this state (always)
            expert_action = expert_agent.choose_action(obs[0])

            # Store state with expert label
            states_empty.append(board_empty)
            states_filled.append(board_filled)
            actions.append(expert_action)

            # Decide which action to take in environment
            if np.random.random() < beta:
                # Take expert action (exploration)
                action_to_take = expert_action
            else:
                # Take CNN action (on-policy)
                action_to_take = cnn_agent.choose_action(obs[0], deterministic=False)

            # Step environment
            obs, reward, terminated, truncated, info = env.step([action_to_take])
            done = terminated[0] or truncated[0]
            steps += 1

    env.close()

    if verbose:
        print(f"Collected {len(actions)} state-action pairs")

    return states_empty, states_filled, actions


def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for board_empty, board_filled, actions in train_loader:
        board_empty = board_empty.unsqueeze(1).to(device)
        board_filled = board_filled.unsqueeze(1).to(device)
        actions = actions.squeeze(1).to(device)

        optimizer.zero_grad()
        logits = model(board_empty, board_filled)
        loss = criterion(logits, actions)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = logits.max(1)
        train_correct += predicted.eq(actions).sum().item()
        train_total += actions.size(0)

    return train_loss / len(train_loader), 100.0 * train_correct / train_total


def validate(model, val_loader, criterion, device):
    """Validate model."""
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for board_empty, board_filled, actions in val_loader:
            board_empty = board_empty.unsqueeze(1).to(device)
            board_filled = board_filled.unsqueeze(1).to(device)
            actions = actions.squeeze(1).to(device)

            logits = model(board_empty, board_filled)
            loss = criterion(logits, actions)

            val_loss += loss.item()
            _, predicted = logits.max(1)
            val_correct += predicted.eq(actions).sum().item()
            val_total += actions.size(0)

    return val_loss / len(val_loader), 100.0 * val_correct / val_total


def evaluate_agent_performance(cnn_agent, n_episodes=5):
    """Evaluate agent performance in actual gameplay."""
    env = tetris.Tetris()
    total_rewards = []
    total_steps = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        steps = 0

        while not done:
            action = cnn_agent.choose_action(obs[0], deterministic=True)
            obs, reward, terminated, truncated, info = env.step([action])
            done = terminated[0] or truncated[0]
            episode_reward += reward[0]
            steps += 1

        total_rewards.append(episode_reward)
        total_steps.append(steps)

    env.close()
    return np.mean(total_rewards), np.mean(total_steps)


def train_iterative(model_path=None, n_iterations=5, episodes_per_iter=20,
                epochs_per_iter=5, batch_size=128, lr=1e-3, device='cpu',
                val_split=0.2, initial_beta=1.0, final_beta=0.1):
    """
    Train with Iterative algorithm.

    Args:
        model_path: Path to pretrained model (if None, start from scratch)
        n_iterations: Number of Iterative iterations
        episodes_per_iter: Episodes to collect per iteration
        epochs_per_iter: Training epochs per iteration
        batch_size: Batch size
        lr: Learning rate
        device: cpu or cuda
        val_split: Validation split ratio
        initial_beta: Starting beta (expert action probability)
        final_beta: Final beta (expert action probability)
    """
    device = torch.device(device)

    # Create models directory
    os.makedirs('models', exist_ok=True)

    # Initialize models
    cnn_model = TetrisCNN(n_rows=20, n_cols=10, n_actions=7).to(device)
    if model_path and os.path.exists(model_path):
        cnn_model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded pretrained model from {model_path}")
    else:
        print("Training from scratch")

    cnn_agent = CNNAgent(device=str(device))
    cnn_agent.model = cnn_model

    expert_agent = HeuristicAgent()

    # Optimizer and criterion
    optimizer = optim.Adam(cnn_model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Aggregate dataset (grows over iterations)
    all_states_empty = []
    all_states_filled = []
    all_actions = []

    best_val_acc = 0.0
    best_game_performance = 0.0

    print(f"\nStarting Iterative training for {n_iterations} iterations")
    print(f"Model parameters: {sum(p.numel() for p in cnn_model.parameters()):,}")

    for iteration in range(n_iterations):
        # Decay beta linearly
        beta = initial_beta + (final_beta - initial_beta) * (iteration / max(n_iterations - 1, 1))

        print(f"\n{'='*60}")
        print(f"Iterative Iteration {iteration + 1}/{n_iterations} (beta={beta:.2f})")
        print(f"{'='*60}")

        # Collect data using current policy
        states_empty, states_filled, actions = collect_iterative_data(
            cnn_agent, expert_agent,
            n_episodes=episodes_per_iter,
            beta=beta,
            verbose=True
        )

        # Add to aggregate dataset
        all_states_empty.extend(states_empty)
        all_states_filled.extend(states_filled)
        all_actions.extend(actions)

        print(f"Total dataset size: {len(all_actions)} samples")

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

        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size,
                               shuffle=False, num_workers=0)

        # Train for several epochs on aggregate dataset
        print(f"\nTraining for {epochs_per_iter} epochs...")
        for epoch in range(epochs_per_iter):
            train_loss, train_acc = train_epoch(cnn_model, train_loader, optimizer, criterion, device)
            val_loss, val_acc = validate(cnn_model, val_loader, criterion, device)

            print(f"  Epoch {epoch+1}/{epochs_per_iter}: "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

            # Save best validation model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(cnn_model.state_dict(), 'models/cnn_agent_iterative_best.pt')
                print(f"    -> Saved best validation model (val_acc: {val_acc:.2f}%)")

        # Evaluate agent performance in actual games
        print("\nEvaluating agent performance...")
        avg_reward, avg_steps = evaluate_agent_performance(cnn_agent, n_episodes=5)
        print(f"  Average reward: {avg_reward:.2f}, Average steps: {avg_steps:.0f}")

        # Save checkpoint
        torch.save(cnn_model.state_dict(), f'models/cnn_agent_iterative_iter{iteration+1}.pt')

        # Save best game performance model
        if avg_reward > best_game_performance:
            best_game_performance = avg_reward
            torch.save(cnn_model.state_dict(), 'models/cnn_agent_iterative_best_performance.pt')
            print(f"  -> Saved best performance model (reward: {avg_reward:.2f})")

    # Save final model
    torch.save(cnn_model.state_dict(), 'models/cnn_agent_iterative_final.pt')
    print(f"\n{'='*60}")
    print("Iterative training complete!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Best game performance: {best_game_performance:.2f} reward")
    print("Models saved to models/")


def main():
    parser = argparse.ArgumentParser(description='Train CNN agent with Iterative')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to pretrained model (optional)')
    parser.add_argument('--iterations', type=int, default=5,
                        help='Number of Iterative iterations')
    parser.add_argument('--episodes', type=int, default=20,
                        help='Episodes to collect per iteration')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Training epochs per iteration')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'],
                        help='Device to use')
    parser.add_argument('--initial-beta', type=float, default=1.0,
                        help='Initial beta (expert probability)')
    parser.add_argument('--final-beta', type=float, default=0.1,
                        help='Final beta (expert probability)')
    parser.add_argument('--val-split', type=float, default=0.2,
                        help='Validation split ratio')

    args = parser.parse_args()

    train_iterative(
        model_path=args.model,
        n_iterations=args.iterations,
        episodes_per_iter=args.episodes,
        epochs_per_iter=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        val_split=args.val_split,
        initial_beta=args.initial_beta,
        final_beta=args.final_beta
    )


if __name__ == "__main__":
    main()
