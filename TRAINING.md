# Training Guide

## Overview

The `train.py` script trains a CNN agent using on-policy data collection with teacher supervision. The student (CNN agent) collects data by playing Tetris, and the teacher (heuristic agent by default) provides the correct action labels. This addresses distribution shift by training on states the student actually encounters.

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

1. **Data Collection**: Student agent plays episodes, teacher labels the states
2. **Training**: CNN trains on aggregate dataset for several epochs
3. **Evaluation**: Agent performance measured in actual gameplay
4. **Checkpointing**: Regular checkpoints saved for resuming

The exploration probability (how often the teacher action is taken during data collection) decays linearly from 0.9 to 0.1 over iterations, allowing the student to gradually take control.

## Arguments

### Model and Training
- `--teacher`: Teacher agent type (default: `heuristic`)
- `--checkpoint`: Path to checkpoint to resume from
- `--device`: Device to use (`cpu` or `cuda`, default: `cpu`)

### Training Schedule
- `--iterations`: Number of data collection iterations (default: `10`)
- `--episodes`: Episodes to collect per iteration (default: `20`)
- `--epochs`: Training epochs per iteration (default: `10`)

### Hyperparameters
- `--batch-size`: Batch size for training (default: `128`)
- `--lr`: Learning rate (default: `1e-3`)
- `--val-split`: Validation split ratio (default: `0.2`)

### Exploration Schedule
- `--initial-exploration`: Initial exploration probability (default: `0.9`)
- `--final-exploration`: Final exploration probability (default: `0.1`)

### Checkpointing
- `--checkpoint-dir`: Directory to save checkpoints (default: `checkpoints`)
- `--save-frequency`: Save checkpoint every N iterations (default: `1`)

## Examples

### Quick training run (for testing)
```bash
python train.py --iterations 3 --episodes 10 --epochs 5
```

### Long training run with CUDA
```bash
python train.py --iterations 20 --episodes 50 --epochs 15 --device cuda
```

### Resume training with different hyperparameters
```bash
python train.py --checkpoint checkpoints/checkpoint_iter010.pt --iterations 20 --lr 5e-4
```

### Adjust exploration schedule
```bash
python train.py --initial-exploration 1.0 --final-exploration 0.0
```

## Checkpoints

Checkpoints are saved to the `checkpoints/` directory:

- `best_val.pt`: Best validation accuracy model
- `best_performance.pt`: Best evaluation reward model
- `checkpoint_iterXXX.pt`: Periodic iteration checkpoints
- `final.pt`: Final training checkpoint

Each checkpoint contains:
- Model weights
- Optimizer state
- Scheduler state
- Training iteration and epoch
- Best metrics achieved
- Dataset size
- Timestamp

## Output

The final trained model is saved to `models/cnn_agent.pt` (weights only, for easy loading with `CNNAgent`).

## Loading a Trained Model

```python
from agents.cnn_agent import CNNAgent

# Load final model
agent = CNNAgent(device='cpu', model_path='models/cnn_agent.pt')

# Or load from checkpoint
agent = CNNAgent(device='cpu')
checkpoint = torch.load('checkpoints/best_performance.pt', weights_only=False)
agent.model.load_state_dict(checkpoint['model_state_dict'])
```

---

# Reinforcement Learning Fine-tuning

After supervised training, you can fine-tune the model with pure RL to potentially surpass the teacher's performance.

## PPO (Recommended)

**Proximal Policy Optimization** - Modern, sample-efficient RL algorithm with better credit assignment than REINFORCE.

### Quick Start

```bash
# Train PPO from supervised model (recommended)
python train_ppo.py --model models/cnn_agent.pt --episodes 500

# Quick test (100 episodes, ~10-15 minutes)
python train_ppo.py --model checkpoints/best_performance.pt --episodes 100

# Resume from PPO checkpoint
python train_ppo.py --checkpoint checkpoints_ppo/checkpoint_ppo_ep0200.pt --episodes 1000
```

### How PPO Works

1. **Rollout collection**: Agent plays and collects experience (states, actions, rewards, values)
2. **Advantage estimation**: Uses GAE (Generalized Advantage Estimation) to compute how much better each action was than expected
3. **Policy update**: Updates policy using clipped surrogate objective (more stable than vanilla policy gradient)
4. **Value function**: Learns to predict future rewards for better credit assignment
5. **No teacher**: Learns purely from game rewards, can discover strategies beyond the teacher

### PPO Arguments

**Model Loading:**
- `--model`: Path to pretrained model (from supervised training)
- `--checkpoint`: Path to PPO checkpoint to resume from
- `--device`: `cpu` or `cuda` (default: `cpu`)

**Training Schedule:**
- `--episodes`: Number of training episodes (default: `1000`)
- `--steps-per-update`: Environment steps before PPO update (default: `2048`)
- `--eval-frequency`: Evaluate every N episodes (default: `50`)
- `--eval-episodes`: Number of episodes for evaluation (default: `10`)

**PPO Hyperparameters:**
- `--lr`: Learning rate (default: `3e-4`)
- `--gamma`: Discount factor (default: `0.99`)
- `--gae-lambda`: GAE lambda for advantage estimation (default: `0.95`)
- `--clip-epsilon`: PPO clipping parameter (default: `0.2`)
- `--vf-coef`: Value function loss coefficient (default: `0.5`)
- `--ent-coef`: Entropy coefficient (default: `0.01`)
- `--n-epochs`: PPO epochs per update (default: `4`)
- `--batch-size`: Mini-batch size (default: `64`)

**Checkpointing:**
- `--checkpoint-dir`: Directory to save checkpoints (default: `checkpoints_ppo`)
- `--save-frequency`: Save checkpoint every N episodes (default: `100`)

### PPO Examples

```bash
# Quick test (100 episodes, ~10-15 min)
python train_ppo.py --model models/cnn_agent.pt --episodes 100 --save-frequency 50

# Standard training
python train_ppo.py --model checkpoints/best_performance.pt --episodes 500

# Long training with CUDA
python train_ppo.py --model models/cnn_agent.pt --episodes 2000 --device cuda

# Resume PPO training
python train_ppo.py --checkpoint checkpoints_ppo/checkpoint_ppo_ep0500.pt --episodes 1000

# More frequent updates (faster learning, less stable)
python train_ppo.py --model models/cnn_agent.pt --episodes 500 --steps-per-update 1024

# Less exploration (lower entropy)
python train_ppo.py --model models/cnn_agent.pt --episodes 500 --ent-coef 0.005
```

### PPO Checkpoints

Checkpoints saved to `checkpoints_ppo/`:
- `best_ppo.pt`: Best evaluation reward
- `checkpoint_ppo_epXXXX.pt`: Periodic checkpoints
- `final_ppo.pt`: Final checkpoint

Final model: `models/cnn_agent_ppo.pt`

### Why PPO is Better than REINFORCE

**PPO advantages:**
- ✅ **Better credit assignment**: Value function + GAE understand which actions matter
- ✅ **Sample efficient**: Reuses data with multiple epochs of updates
- ✅ **More stable**: Clipped objective prevents catastrophic policy changes
- ✅ **Faster learning**: Learns from batches rather than full episodes

**REINFORCE issues:**
- ❌ High variance (noisy gradients)
- ❌ Poor credit assignment (treats all actions in episode equally)
- ❌ Sample inefficient (one gradient step per episode)
- ❌ Can be very slow to improve

---

## REINFORCE (Legacy)

For comparison, the older REINFORCE implementation is still available in `train_rl.py`, but **PPO is recommended** for better results.

### REINFORCE Quick Start

```bash
# Train from supervised model
python train_rl.py --model models/cnn_agent.pt --episodes 500

# Resume from checkpoint
python train_rl.py --checkpoint checkpoints_rl/checkpoint_rl_ep0200.pt --episodes 1000
```

Checkpoints saved to `checkpoints_rl/`, final model: `models/cnn_agent_rl.pt`

---

## Why RL After Supervised?

**Supervised learning** is limited by:
- Teacher's skill ceiling
- Distribution shift issues

**RL fine-tuning** (especially PPO) can:
- ✅ Discover strategies the teacher doesn't use
- ✅ Optimize directly for game rewards
- ✅ Potentially surpass teacher performance
- ✅ Learn from its own experience

**Best practice:** Supervised pretraining → PPO fine-tuning
