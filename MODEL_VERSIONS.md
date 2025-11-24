# Model and Dataset Versions

This document tracks the different versions of trained models and datasets, including their training configurations.

## Version History

### v1
**Configuration:**
- Mixed teacher: `random_prob = (uniform(0,1))^3`, `temperature = (uniform(0,1))^3`
- Episodes: 1000
- Learning rate: 1e-4
- Gamma: 0.99
- **Notes:** First training run with power 3 for both random and temperature

### v2
**Configuration:**
- Mixed teacher: `random_prob = (uniform(0,1))^5`, `temperature = (uniform(0,1))^5`
- Episodes: 1000
- Learning rate: 1e-4
- Gamma: 0.99
- **Notes:** Increased power to 5 for less randomness

### v3
**Configuration:**
- Mixed teacher: `random_prob = (uniform(0,1))^5`, `temperature = 0.0`
- Episodes: 1000
- Learning rate: 1e-4
- Gamma: 0.99
- **Notes:** Set temperature to 0 for greedy heuristic selection

### v4
**Configuration:**
- Mixed teacher: Same as v3 (reusing v3 data)
- Initialized from: v3 model
- Training data: v3 dataset
- Learning rate: 1e-4
- Gamma: 0.99
- **Notes:** Continued training on v3 model with v3 data

### v5
**Configuration:**
- Mixed teacher: Same as v2 (`random_prob = (uniform(0,1))^5`, `temperature = (uniform(0,1))^5`)
- Initialized from: v4 model
- Training data: v2 dataset
- Learning rate: 1e-4
- Gamma: 0.99
- **Notes:** Continued training on v4 model but with v2 data

### v6
**Configuration:**
- Mixed teacher: Same as v3 (`random_prob = (uniform(0,1))^5`, `temperature = 0.0`)
- Initialized from: v5 model
- Training data: v3 dataset
- Learning rate: 1e-4
- Gamma: 0.99
- **Issues:** Model didn't learn to clear lines effectively, teacher still had ~16% random actions
- **Notes:** Continued training on v5 model with v3 data

### v7 (Current)
**Configuration:**
- Mixed teacher: `random_prob = 0.0`, `temperature = 0.0` (pure greedy heuristic, no randomness)
- Initialized from: v6 model
- Training data: New v7 dataset (pure heuristic)
- Episodes: 1000
- Epochs: 10
- Learning rate: 1e-4
- Gamma: 0.99
- **Goal:** Learn from high-quality pure heuristic demonstrations without any random noise
- **Notes:** Episodes are long (~500-1000+ steps), so 1000 episodes provides substantial data

## Training Pipeline

### Supervised Training
```bash
python train_supervised_mixed.py \
    --init-model models/supervised_value_v6.pth \
    --num-episodes 1000 \
    --output models/supervised_value_v7.pth \
    --save-data data/supervised_dataset_v7.pkl \
    --epochs 10 \
    --device mps
```

### RL Fine-tuning
```bash
python train_rl.py \
    --init-model models/supervised_value_v7.pth \
    --output models/rl_value_v7.pth \
    --num-episodes 300 \
    --epsilon-start 0.5 \
    --epsilon-end 0.05 \
    --device mps \
    --save-interval 50
```

## Reward Structure
All versions (v5+) use simple line-clear rewards:
- 0 lines: 0.0
- 1 line: 0.1
- 2 lines: 0.3
- 3 lines: 0.6
- 4 lines: 1.0

Gamma: 0.99 (for discounted returns)

## Files
- `models/supervised_value_v*.pth` - Supervised trained models
- `models/supervised_value_v*_epoch_*.pth` - Per-epoch checkpoints
- `data/supervised_dataset_v*.pkl` - Collected rollout datasets
- `models/rl_value_v*.pth` - RL fine-tuned models (none saved yet)

## Notes
- v1: First attempt with moderate randomness (power 3)
- v2: Less randomness (power 5) for both random and temperature
- v3: Greedy heuristic (temp=0) with some random actions (power 5)
- v4-v6: Continued training experiments mixing different datasets and models
- v7: Pure greedy heuristic (no randomness at all) for best quality demonstrations
- No RL-trained models have been saved yet
