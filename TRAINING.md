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
checkpoint = torch.load('checkpoints/best_performance.pt')
agent.model.load_state_dict(checkpoint['model_state_dict'])
```
