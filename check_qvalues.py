"""
Quick sanity check: verify the model learns large Q-value deltas for piece-locking transitions.
"""

import torch
import numpy as np
import pickle
from model import ValueNetwork
from reward_utils import compute_shaped_reward, compute_lines_cleared

def check_piece_locking_qvalues(model_path, data_path, device='mps', num_samples=20):
    """Check Q-values on piece-locking transitions with significant rewards."""

    device = torch.device(device)
    model = ValueNetwork(n_rows=20, n_cols=10, n_actions=6).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Load dataset
    print(f"Loading data from {data_path}...")
    with open(data_path, 'rb') as f:
        transitions = pickle.load(f)

    print(f"Loaded {len(transitions)} transitions\n")
    print("=" * 60)
    print("Checking Q-values on piece-locking transitions...")
    print("=" * 60)

    # Find piece-locking transitions with significant rewards
    significant_transitions = []
    for t in transitions:
        old_board, old_filled, action, _, new_board, _, done = t

        if done:
            continue

        active = old_filled - old_board
        lines = compute_lines_cleared(old_board, active, new_board)
        reward = compute_shaped_reward(old_board, new_board, lines)

        if abs(reward) > 0.5:  # Significant reward (piece was locked)
            significant_transitions.append((t, reward))

    print(f"Found {len(significant_transitions)} transitions with |reward| > 0.5\n")

    if len(significant_transitions) == 0:
        print("No significant reward transitions found!")
        return

    # Sample random transitions
    sample_size = min(num_samples, len(significant_transitions))
    sampled = np.random.choice(len(significant_transitions), sample_size, replace=False)

    q_values_list = []
    rewards_list = []

    for idx in sampled:
        (old_board, old_filled, action, _, new_board, _, done), reward = significant_transitions[idx]

        with torch.no_grad():
            empty_t = torch.FloatTensor(old_board).unsqueeze(0).unsqueeze(0).to(device)
            filled_t = torch.FloatTensor(old_filled).unsqueeze(0).unsqueeze(0).to(device)
            q_vals = model(empty_t, filled_t)[0].cpu().numpy()

        q_chosen = q_vals[action]
        q_values_list.append(q_chosen)
        rewards_list.append(reward)

        print(f"Reward: {reward:+7.2f} | Chosen Q: {q_chosen:+7.2f} | Action: {action}")
        print(f"  All Q-values: [{', '.join([f'{q:+6.2f}' for q in q_vals])}]")
        print()

    print("=" * 60)
    print("Summary Statistics:")
    print("=" * 60)
    print(f"Reward range:  [{min(rewards_list):+.2f}, {max(rewards_list):+.2f}]")
    print(f"Q-value range: [{min(q_values_list):+.2f}, {max(q_values_list):+.2f}]")
    print(f"Mean reward:   {np.mean(rewards_list):+.2f}")
    print(f"Mean Q-value:  {np.mean(q_values_list):+.2f}")
    print()

    # Check if Q-values are learning
    if max(abs(q) for q in q_values_list) < 5.0:
        print("⚠️  WARNING: Q-values are small (< 5.0). Model may need more training.")
    elif max(abs(q) for q in q_values_list) < 10.0:
        print("⚠️  Q-values are moderate (< 10.0). Training is progressing but could use more epochs.")
    else:
        print("✓ Q-values are developing well (> 10.0).")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Check Q-values on piece-locking transitions')
    parser.add_argument('--model', type=str, default='models/supervised_value.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--data', type=str, default='data/supervised_dataset_v4.pkl',
                        help='Path to dataset')
    parser.add_argument('--device', type=str, default='mps',
                        help='Device (cpu, cuda, mps)')
    parser.add_argument('--samples', type=int, default=20,
                        help='Number of samples to check')

    args = parser.parse_args()

    check_piece_locking_qvalues(args.model, args.data, args.device, args.samples)
