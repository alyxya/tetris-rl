# Tetris RL Training Guide

## Overview

This guide explains the two-phase training pipeline for the Tetris value network:

1. **Supervised Learning Phase**: Collect data using a mixed teacher (random + heuristic actions) and train the Q-value network to learn from simple line-clear rewards.
2. **Reinforcement Learning Phase**: Continue training with Q-learning using the same simple reward structure.

## Reward Structure

The reward function has been simplified to focus **only on line clears**:

- **1 line clear**: 0.1
- **2 line clears**: 0.3
- **3 line clears**: 0.6
- **4 line clears**: 1.0
- **No lines cleared**: 0.0
- **Game over**: 0.0 (no death penalty)

All other factors (height, holes, bumpiness) have been removed.

## Mixed Teacher Agent

The supervised learning phase uses a mixed teacher that combines:
- **Random actions**: Selected with probability `p = (uniform(0,1))^2`
- **Heuristic agent**: Used otherwise, with temperature `t = (uniform(0,1))^2`

Both `p` and `t` are sampled **once per episode** and remain fixed for all actions in that episode.

## Training Parameters

Key parameters (already set in the code):
- **Gamma (discount factor)**: 0.99
- **Supervised learning episodes**: 1000 (default)
- **RL episodes**: 1000 (default)

## Phase 1: Supervised Learning

Collect data using the mixed teacher and train the value network to predict Q-values.

### Basic Usage

```bash
python train_supervised_mixed.py \
    --num-episodes 1000 \
    --output models/supervised_value.pth \
    --save-data data/supervised_dataset.pkl
```

### Arguments

- `--num-episodes`: Number of episodes to collect (default: 1000)
- `--epochs`: Number of training epochs (default: 20)
- `--batch-size`: Training batch size (default: 64)
- `--lr`: Learning rate (default: 1e-3)
- `--gamma`: Discount factor (default: 0.99)
- `--output`: Path to save the trained model (required)
- `--save-data`: Path to save collected dataset for reuse (optional)
- `--load-data`: Path to load pre-collected dataset (optional)
- `--device`: Device to use: 'cpu' or 'cuda' (default: 'cpu')

### Example with GPU

```bash
python train_supervised_mixed.py \
    --num-episodes 1000 \
    --epochs 20 \
    --device cuda \
    --output models/supervised_value.pth \
    --save-data data/supervised_dataset.pkl
```

### Reusing Collected Data

Once you've collected a dataset, you can reuse it to experiment with different training hyperparameters:

```bash
# Collect data once
python train_supervised_mixed.py \
    --num-episodes 1000 \
    --save-data data/supervised_dataset.pkl \
    --output models/supervised_value_v1.pth

# Reuse data with different learning rate
python train_supervised_mixed.py \
    --load-data data/supervised_dataset.pkl \
    --lr 5e-4 \
    --output models/supervised_value_v2.pth
```

## Phase 2: Reinforcement Learning

Continue training with Q-learning using experience replay and epsilon-greedy exploration.

### Basic Usage

```bash
python train_rl.py \
    --num-episodes 1000 \
    --init-model models/supervised_value.pth \
    --output models/rl_value.pth
```

### Arguments

- `--num-episodes`: Number of episodes to train (default: 1000)
- `--lr`: Learning rate (default: 1e-4)
- `--gamma`: Discount factor (default: 0.99)
- `--buffer-size`: Replay buffer capacity (default: 10000)
- `--batch-size`: Training batch size (default: 256)
- `--epsilon-start`: Initial exploration rate (default: 0.2)
- `--epsilon-end`: Final exploration rate (default: 0.01)
- `--target-update`: Target network update frequency in episodes (default: 2)
- `--temperature`: Boltzmann exploration temperature (default: None)
- `--init-model`: Path to pretrained model (optional but recommended)
- `--output`: Path to save the trained model (required)
- `--save-interval`: Save checkpoint every N episodes (default: 20)
- `--device`: Device to use: 'cpu' or 'cuda' (default: 'cpu')
- `--grad-clip`: Gradient clipping threshold (default: 1.0)

### Example with GPU

```bash
python train_rl.py \
    --num-episodes 1000 \
    --init-model models/supervised_value.pth \
    --device cuda \
    --output models/rl_value.pth \
    --save-interval 50
```

### Training from Scratch (Not Recommended)

While you can train the RL agent from scratch without supervised pretraining, it's not recommended:

```bash
python train_rl.py \
    --num-episodes 5000 \
    --output models/rl_value_scratch.pth
```

## Complete Training Pipeline

Here's the recommended end-to-end training workflow:

```bash
# Step 1: Create directories
mkdir -p models data

# Step 2: Supervised learning (collect data + train)
python train_supervised_mixed.py \
    --num-episodes 1000 \
    --epochs 20 \
    --output models/supervised_value.pth \
    --save-data data/supervised_dataset.pkl \
    --device cuda

# Step 3: Reinforcement learning (fine-tune with Q-learning)
python train_rl.py \
    --num-episodes 1000 \
    --init-model models/supervised_value.pth \
    --output models/rl_value.pth \
    --device cuda \
    --save-interval 50

# Step 4: Evaluate the trained agent
python eval_agent.py \
    --model models/rl_value.pth \
    --num-episodes 100
```

## Testing the Pipeline

Before running full training, you can verify everything works correctly:

```bash
python test_pipeline.py
```

This will run integration tests for:
- Mixed teacher agent with environment interaction
- Value network forward pass
- Value agent action selection
- Reward computation edge cases

## Key Changes from Previous Version

1. **Simplified Rewards**: Removed all heuristic-based rewards, normalized scores, and death penalty. Only line clears matter now.

2. **Mixed Teacher**: New `MixedTeacherAgent` combines random actions and heuristic agent with per-episode sampling of:
   - Random action probability: `(uniform(0,1))^2`
   - Heuristic temperature: `(uniform(0,1))^2`

3. **New Supervised Script**: `train_supervised_mixed.py` replaces the old `train_supervised.py` with:
   - Support for saving/loading datasets
   - Simple line-clear-based Q-value targets
   - Mixed teacher data collection

4. **Refactored RL Training**: `train_rl.py` now uses simple rewards consistently with supervised learning.

## File Structure

```
tetris-rl/
├── reward_utils.py              # Simple reward computation
├── mixed_teacher_agent.py       # Mixed random/heuristic teacher
├── train_supervised_mixed.py    # Supervised learning phase
├── train_rl.py                  # RL fine-tuning phase
├── test_pipeline.py             # Integration tests
├── model.py                     # Value network architecture
├── value_agent.py               # Value-based agent for inference
├── heuristic_agent.py           # Heuristic-based agent
├── base_agent.py                # Base agent class
└── TRAINING_GUIDE.md            # This file
```

## Troubleshooting

### Out of Memory (GPU)

Reduce batch size:
```bash
python train_rl.py --batch-size 128 ...
```

### Training Too Slow

- Reduce number of episodes
- Use GPU if available
- For supervised learning, reuse saved dataset

### Poor Performance

- Ensure you're using supervised pretraining
- Increase number of training episodes
- Check that rewards are being computed correctly using `test_pipeline.py`

## Monitoring Training

Both training scripts print progress every 10 episodes. Watch for:
- **Supervised learning**: Decreasing MSE loss
- **RL training**: Increasing episode length and total reward
