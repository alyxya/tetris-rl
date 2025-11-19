"""
Supervised learning: Train CNN agent to imitate heuristic agent.

This creates a dataset by running the heuristic agent and collecting
(state, action) pairs, then trains the CNN to predict the heuristic actions.
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
    """Dataset of (state, action) pairs from expert demonstrations."""

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


def collect_expert_data(n_episodes=50, verbose=True):
    """
    Collect demonstration data from heuristic agent.

    Args:
        n_episodes: Number of episodes to collect
        verbose: Print progress

    Returns:
        states_empty: list of boards with piece as empty
        states_filled: list of boards with piece as filled
        actions: list of actions taken
    """
    env = tetris.Tetris()
    expert = HeuristicAgent()

    states_empty = []
    states_filled = []
    actions = []

    if verbose:
        print(f"Collecting expert demonstrations from {n_episodes} episodes...")

    for episode in tqdm(range(n_episodes), disable=not verbose):
        obs, _ = env.reset()
        expert.reset()
        done = False

        while not done:
            # Get expert action
            action = expert.choose_action(obs[0])

            # Parse observation into dual representation
            full_board = obs[0, :200].reshape(20, 10)
            locked = (full_board == 1).astype(np.float32)
            active = (full_board == 2)

            # Board with piece as empty
            board_empty = locked.copy()

            # Board with piece as filled
            board_filled = locked.copy()
            board_filled[active] = 1.0

            # Store experience
            states_empty.append(board_empty)
            states_filled.append(board_filled)
            actions.append(action)

            # Step environment
            obs, reward, terminated, truncated, info = env.step([action])
            done = terminated[0] or truncated[0]

    env.close()

    if verbose:
        print(f"Collected {len(actions)} state-action pairs")

    return states_empty, states_filled, actions


def train_supervised(model, train_loader, val_loader, device, n_epochs=20, lr=1e-3):
    """
    Train model with supervised learning.

    Args:
        model: TetrisCNN model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        device: torch device
        n_epochs: Number of training epochs
        lr: Learning rate

    Returns:
        model: Trained model
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                      factor=0.5, patience=3)

    best_val_loss = float('inf')

    for epoch in range(n_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for board_empty, board_filled, actions in tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}"):
            board_empty = board_empty.unsqueeze(1).to(device)  # Add channel dim
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

        train_loss /= len(train_loader)
        train_acc = 100.0 * train_correct / train_total

        # Validation
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

        val_loss /= len(val_loader)
        val_acc = 100.0 * val_correct / val_total

        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        scheduler.step(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'models/cnn_agent_best.pt')
            print(f"  -> Saved best model (val_loss: {val_loss:.4f})")

    return model


def main():
    parser = argparse.ArgumentParser(description='Train CNN agent via supervised learning')
    parser.add_argument('--episodes', type=int, default=100,
                        help='Number of episodes to collect from expert')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'],
                        help='Device to use for training')
    parser.add_argument('--val-split', type=float, default=0.2,
                        help='Validation split ratio')

    args = parser.parse_args()

    # Create models directory
    os.makedirs('models', exist_ok=True)

    # Collect expert data
    states_empty, states_filled, actions = collect_expert_data(
        n_episodes=args.episodes,
        verbose=True
    )

    # Split into train/val
    n_samples = len(actions)
    n_val = int(n_samples * args.val_split)
    indices = np.random.permutation(n_samples)

    train_idx = indices[n_val:]
    val_idx = indices[:n_val]

    train_dataset = TetrisDataset(
        [states_empty[i] for i in train_idx],
        [states_filled[i] for i in train_idx],
        [actions[i] for i in train_idx]
    )

    val_dataset = TetrisDataset(
        [states_empty[i] for i in val_idx],
        [states_filled[i] for i in val_idx],
        [actions[i] for i in val_idx]
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                           shuffle=False, num_workers=0)

    print(f"\nTrain samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Create and train model
    device = torch.device(args.device)
    model = TetrisCNN(n_rows=20, n_cols=10, n_actions=7)

    print(f"\nTraining on {device}...")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    model = train_supervised(
        model,
        train_loader,
        val_loader,
        device,
        n_epochs=args.epochs,
        lr=args.lr
    )

    # Save final model
    torch.save(model.state_dict(), 'models/cnn_agent_final.pt')
    print("\nTraining complete! Models saved to models/")


if __name__ == "__main__":
    main()
