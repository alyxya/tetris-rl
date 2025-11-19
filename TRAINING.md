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

After supervised training, you can fine-tune the model with pure RL (REINFORCE) to potentially surpass the teacher's performance.

## Quick Start

### Train from supervised model
```bash
python train_rl.py --model models/cnn_agent.pt --episodes 500
```

### Resume from RL checkpoint
```bash
python train_rl.py --checkpoint checkpoints_rl/checkpoint_rl_ep0200.pt --episodes 1000
```

## RL Training Process

1. **Agent plays episode**: Uses current policy to play Tetris
2. **Collects rewards**: Environment provides reward signal
3. **Policy update**: REINFORCE updates policy to maximize expected rewards
4. **No teacher**: Learns purely from game rewards, independent of heuristic

## Arguments

### Model Loading
- `--model`: Path to pretrained model (from supervised training)
- `--checkpoint`: Path to RL checkpoint to resume from
- `--device`: Device to use (`cpu` or `cuda`, default: `cpu`)

### Training Schedule
- `--episodes`: Number of training episodes (default: `1000`)
- `--eval-frequency`: Evaluate every N episodes (default: `50`)
- `--eval-episodes`: Number of episodes for evaluation (default: `10`)

### RL Hyperparameters
- `--lr`: Learning rate (default: `1e-4`)
- `--gamma`: Discount factor for returns (default: `0.99`)
- `--entropy-coef`: Entropy regularization coefficient (default: `0.01`)
- `--temperature`: Sampling temperature (default: `1.0`)

### Checkpointing
- `--checkpoint-dir`: Directory to save checkpoints (default: `checkpoints_rl`)
- `--save-frequency`: Save checkpoint every N episodes (default: `100`)

## Examples

### Quick RL test (100 episodes)
```bash
python train_rl.py --model models/cnn_agent.pt --episodes 100 --save-frequency 50
```

### Long RL training with CUDA
```bash
python train_rl.py --model checkpoints/best_performance.pt --episodes 2000 --device cuda
```

### Resume RL training
```bash
python train_rl.py --checkpoint checkpoints_rl/checkpoint_rl_ep0500.pt --episodes 1000
```

### Adjust exploration (lower entropy = less random)
```bash
python train_rl.py --model models/cnn_agent.pt --episodes 500 --entropy-coef 0.005
```

## RL Checkpoints

Checkpoints are saved to `checkpoints_rl/` directory:

- `best_rl.pt`: Best evaluation reward model
- `checkpoint_rl_epXXXX.pt`: Periodic episode checkpoints
- `final_rl.pt`: Final training checkpoint

Each checkpoint contains:
- Model weights
- Optimizer state
- Episode number
- Best reward achieved
- Timestamp

## Output

The final RL-trained model is saved to `models/cnn_agent_rl.pt`.

## Why RL After Supervised?

**Supervised learning** teaches the agent to imitate the teacher, but it's limited by:
- Teacher's skill ceiling
- Distribution shift issues

**RL fine-tuning** can:
- Discover strategies the teacher doesn't use
- Optimize directly for game rewards
- Potentially surpass teacher performance

However, RL is:
- More sample-inefficient (needs many episodes)
- Can be unstable (use pretrained model as starting point)
- Benefits greatly from good initialization (supervised pretraining)
