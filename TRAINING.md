# Training Guide

## Overview

`train.py` performs supervised pretraining for the unified Q-value agent. The student collects its own experience in the PufferLib Tetris environment while a heuristic teacher labels every state with the desired action. Random actions are occasionally injected so the dataset captures a diverse set of boards. The resulting model already ranks actions reasonably well, which makes it an ideal starting point for RL fine-tuning.

## Quick Start

### Train from scratch
```bash
python train.py
```

### Resume from checkpoint
```bash
python train.py --checkpoint checkpoints/checkpoint_iter005.pt
```

## Training Process

1. **Data collection** – Student agent acts in the environment. Each step is labeled by the teacher and optionally perturbed by random actions.
2. **Supervised update** – The Q-network trains with cross-entropy on aggregated `(state, action)` pairs, treating the Q-values as logits.
3. **Evaluation & checkpointing** – After each iteration we evaluate the greedy policy, log metrics, and save checkpoints.

Teacher usage decays from `--initial-exploration` to `--final-exploration`, while `--random-action-prob` controls the amount of purely random actions during data collection.

## Arguments

### Model and Training
- `--teacher`: Teacher agent type (default `heuristic`)
- `--checkpoint`: Resume from checkpoint file
- `--device`: `cpu` or `cuda`

### Training Schedule
- `--iterations`: Data collection iterations (default `10`)
- `--episodes`: Episodes per iteration (default `20`)
- `--epochs`: Training epochs per iteration (default `10`)

### Hyperparameters
- `--batch-size`: Batch size (default `128`)
- `--lr`: Learning rate (default `1e-3`)
- `--val-split`: Validation ratio (default `0.2`)

### Exploration Mix
- `--initial-exploration`: Initial probability of executing teacher action (default `0.9`)
- `--final-exploration`: Final probability of executing teacher action (default `0.1`)
- `--random-action-prob`: Probability of forcing a random action (default `0.1`)

### Checkpointing
- `--checkpoint-dir`: Directory for supervised checkpoints (default `checkpoints`)
- `--save-frequency`: Save checkpoint every N iterations (default `1`)

## Outputs

- `checkpoints/best_val.pt`: Best validation accuracy
- `checkpoints/best_performance.pt`: Best evaluation reward
- `checkpoints/checkpoint_iterXXX.pt`: Iteration snapshots
- `models/q_value_agent.pt`: Final supervised weights

## Loading a Trained Model

```python
import torch
from agents.q_agent import QValueAgent

agent = QValueAgent(device='cpu', model_path='models/q_value_agent.pt')

# Or load from a checkpoint manually
agent = QValueAgent(device='cpu')
checkpoint = torch.load('checkpoints/best_performance.pt', weights_only=False)
agent.model.load_state_dict(checkpoint['model_state_dict'])
```

---

# Reinforcement Learning Fine-tuning

`train_rl.py` continues training with Q-learning (TD updates, replay buffer, and target network). It learns directly from the default environment reward and refines the Q-values learned during supervision.

## Quick Start

### Fine-tune from supervised weights
```bash
python train_rl.py --model-path models/q_value_agent.pt --episodes 500
```

### Resume RL from checkpoint
```bash
python train_rl.py --checkpoint checkpoints_rl/episode_0500.pt --episodes 1000
```

## RL Process

1. **Act with epsilon-greedy policy** – Start near-greedy and decay epsilon to 0.01.
2. **Store transitions** – Push `(s, a, r, s', done)` tuples into the replay buffer.
3. **TD updates** – Sample mini-batches, minimize MSE between predicted Q-values and TD targets.
4. **Target network** – Periodically sync a target network for stability.

## RL Arguments

- `--model-path`: Optional supervised weights to bootstrap from
- `--checkpoint`: Resume RL training
- `--episodes`: Number of RL episodes (default `500`)
- `--device`: `cpu` or `cuda`
- `--batch-size`: TD batch size (default `128`)
- `--buffer-size`: Replay buffer capacity (default `100000`)
- `--min-buffer`: Samples required before updates (default `2000`)
- `--gamma`: Discount factor (default `0.99`)
- `--lr`: Learning rate (default `1e-4`)
- `--epsilon-start` / `--epsilon-end` / `--epsilon-decay`: Exploration schedule
- `--target-update`: Frequency (in steps) to sync the target network
- `--eval-frequency`: Evaluate greedy policy every N episodes
- `--eval-episodes`: Episodes per evaluation run
- `--checkpoint-dir`: Directory for RL checkpoints (default `checkpoints_rl`)
- `--save-frequency`: Save checkpoint every N episodes

## Outputs

- `checkpoints_rl/best.pt`: Best evaluation reward during RL
- `checkpoints_rl/episode_XXXX.pt`: Periodic RL checkpoints
- `models/q_value_agent_rl.pt`: Final RL fine-tuned weights

## Why RL After Supervised?

- Supervised training teaches relative action preferences but not true long-horizon returns.
- Q-learning refines those estimates using actual rewards, allowing the agent to surpass the heuristic and learn strategies unavailable in the demonstrations.
