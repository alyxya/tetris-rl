# Training Log

This document tracks all training runs and model versions for the Tetris RL project.

## Training Approach

**Supervised Q-Learning:**
- Collect transitions (s, a, r, s', done) from teacher demonstrations
- Train using iterative Q-learning with Bellman equation: `Q(s,a) = r + γ * max Q(s',a')`
- Use target network (updated every 5 epochs) for stability
- Q-values converge over epochs to optimal value function

**Teacher Agent (Mixed):**
- Per-episode sampling of `random_prob` and `temperature`
- `random_prob`: Probability of taking random action vs heuristic action
- `temperature`: Softmax temperature for heuristic agent (0 = greedy)
- Distribution: `(uniform(0,1))^power` where power controls exploration

**Reward Structure:**
- Line clears only: {0: 0.0, 1: 0.1, 2: 0.3, 3: 0.6, 4: 1.0}
- Gamma: 0.99

## Version History

### v1 (Current)
**Date:** 2025-01-23

**Teacher Configuration:**
- `random_prob = (uniform(0,1))^3` (E[random_prob] = 0.25)
- `temperature = (uniform(0,1))^3` (E[temperature] = 0.25)

**Training Configuration:**
- Episodes: 1000
- Epochs: 5
- Batch size: 256
- Learning rate: 1e-4
- Gamma: 0.99
- Target network update: Every 5 epochs
- Device: mps

**Command:**
```bash
python train_supervised_mixed.py \
    --num-episodes 1000 \
    --output models/supervised_value_v1.pth \
    --save-data data/supervised_dataset_v1.pkl \
    --epochs 5 \
    --device mps
```

**Status:** Completed

**Results:**
- [Add results after testing]

**Notes:**
- First model trained with correct iterative Q-learning approach
- Power of 3 provides good balance: ~25% exploration, ~75% heuristic guidance

---

### v2 (Current)
**Date:** 2025-01-23

**Teacher Configuration:**
- `random_prob = (uniform(0,1))^4` (E[random_prob] = 0.20)
- `temperature = (uniform(0,1))^4` (E[temperature] = 0.20)

**Training Configuration:**
- Episodes: 1000 (new dataset)
- Epochs: 5
- Batch size: 256
- Learning rate: 1e-4
- Gamma: 0.99
- Target network update: Every 5 epochs
- Device: mps
- **Initialized from:** models/supervised_value_v1.pth

**Command:**
```bash
python train_supervised_mixed.py \
    --init-model models/supervised_value_v1.pth \
    --num-episodes 1000 \
    --output models/supervised_value_v2.pth \
    --save-data data/supervised_dataset_v2.pkl \
    --epochs 5 \
    --device mps
```

**Status:** Completed

**Results:**
- [Add results after testing]

**Notes:**
- Continue training v1 with less exploration (power 4 = 20% vs 25%)
- Collect fresh dataset with more greedy heuristic guidance

---

### v3 (Current)
**Date:** 2025-01-23

**Teacher Configuration:**
- `random_prob = (uniform(0,1))^5` (E[random_prob] = 0.167)
- `temperature = (uniform(0,1))^5` (E[temperature] = 0.167)

**Training Configuration:**
- Episodes: 1000 (new dataset)
- Epochs: 5
- Batch size: 256
- Learning rate: 1e-4
- Gamma: 0.99
- Target network update: Every 5 epochs
- Device: mps
- **Initialized from:** models/supervised_value_v2.pth

**Command:**
```bash
python train_supervised_mixed.py \
    --init-model models/supervised_value_v2.pth \
    --num-episodes 1000 \
    --output models/supervised_value_v3.pth \
    --save-data data/supervised_dataset_v3.pkl \
    --epochs 5 \
    --device mps
```

**Status:** Completed

**Results:**
- [Add results after testing]

**Notes:**
- Continue training v2 with even less exploration (power 5 = ~17%)
- More deterministic heuristic guidance

---

### v4 (Current)
**Date:** 2025-01-23

**Teacher Configuration:**
- `random_prob = (uniform(0,1))^4` (E[random_prob] = 0.20)
- `temperature = 0.0` (greedy heuristic)

**Training Configuration:**
- Episodes: 1000 (new dataset)
- Epochs: 5
- Batch size: 256
- Learning rate: 1e-4
- Gamma: 0.99
- Target network update: Every 5 epochs
- Device: mps
- **Initialized from:** models/supervised_value_v3.pth

**Command:**
```bash
python train_supervised_mixed.py \
    --init-model models/supervised_value_v3.pth \
    --num-episodes 1000 \
    --output models/supervised_value_v4.pth \
    --save-data data/supervised_dataset_v4.pkl \
    --epochs 5 \
    --device mps
```

**Status:** Completed

**Results:**
- [Add results after testing]

**Notes:**
- Continue training v3 with greedy heuristic (temp=0)
- Only source of randomness is random action probability (~20%)

---

### v5 (Current)
**Date:** 2025-01-23

**Reward Structure: SHAPED REWARDS** (Major Change)
- **Aggregate height penalty**: -0.51 × sum(column_heights)
- **Holes penalty**: -0.36 × total_holes
- **Bumpiness penalty**: -0.18 × sum(|height_diff|)
- **Line clear bonuses**: {1: 1.0, 2: 3.0, 3: 5.0, 4: 10.0}
- **Total reward**: f(new_board) - f(old_board) + line_bonus

**Training Configuration:**
- **Reuses existing v1-v4 datasets** (computes shaped rewards on-the-fly)
- Train on v1 data (5 epochs) → v2 data (5 epochs) → v3 data (5 epochs) → v4 data (5 epochs)
- Batch size: 256
- Learning rate: 1e-4
- Gamma: 0.99
- Target network update: Every 5 epochs
- Device: mps
- **Initialized from:** Random weights (fresh start, not from v4)

**Commands:**
```bash
# Train on v1 dataset with shaped rewards (from scratch)
python train_supervised_mixed.py \
    --load-data data/supervised_dataset_v1.pkl \
    --output models/supervised_value_v5_from_v1.pth \
    --epochs 5 \
    --shaped-rewards \
    --device mps

# Continue with v2 dataset
python train_supervised_mixed.py \
    --init-model models/supervised_value_v5_from_v1.pth \
    --load-data data/supervised_dataset_v2.pkl \
    --output models/supervised_value_v5_from_v2.pth \
    --epochs 5 \
    --shaped-rewards \
    --device mps

# Continue with v3 dataset
python train_supervised_mixed.py \
    --init-model models/supervised_value_v5_from_v2.pth \
    --load-data data/supervised_dataset_v3.pkl \
    --output models/supervised_value_v5_from_v3.pth \
    --epochs 5 \
    --shaped-rewards \
    --device mps

# Continue with v4 dataset (final)
python train_supervised_mixed.py \
    --init-model models/supervised_value_v5_from_v3.pth \
    --load-data data/supervised_dataset_v4.pkl \
    --output models/supervised_value_v5.pth \
    --epochs 5 \
    --shaped-rewards \
    --device mps
```

**Status:** Ready to train

**Notes:**
- **Solves sparse reward problem**: 92.65% of actions had reward=0 with simple rewards
- **Dense feedback**: Every piece placement gets shaped reward based on board quality
- **Research-backed coefficients**: Based on successful Tetris DRL implementations
- **No new data needed**: Computes shaped rewards from existing board states
- **Fresh start**: Training from scratch (not from v4) to learn shaped reward value function cleanly
- **Expected improvement**: Q-values should be more differentiated, better policy learning

---

### v6 (Current)
**Date:** 2025-01-24

**Major Change: 6-Action Model (Excludes HOLD)**
- Model outputs only 6 Q-values: NO_OP, LEFT, RIGHT, ROTATE, SOFT_DROP, HARD_DROP
- HOLD action (action 6) completely removed from model architecture
- Training automatically filters out all action=6 transitions from datasets

**Reward Structure: SHAPED REWARDS + DEATH PENALTY**
- Aggregate height penalty: -0.51 × sum(column_heights)
- Holes penalty: -0.36 × total_holes
- Bumpiness penalty: -0.18 × sum(|height_diff|)
- Line clear bonuses: {1: 1.0, 2: 3.0, 3: 5.0, 4: 10.0}
- **Death penalty: -1.0**
- Total reward: f(new_board) - f(old_board) + line_bonus (or -1.0 if episode ends)

**Training Configuration:**
- Reuses existing v1-v4 datasets (computes shaped rewards on-the-fly, filters HOLD actions)
- Train on v1 data (5 epochs) → v2 data (5 epochs) → v3 data (5 epochs) → v4 data (5 epochs)
- Batch size: 256
- Learning rate: 1e-4
- Gamma: 0.99
- Target network update: Every 5 epochs
- Device: mps
- **Initialized from:** Random weights (fresh start)

**Commands:**
```bash
# Train on v1 dataset with shaped rewards (from scratch)
python train_supervised_mixed.py \
    --load-data data/supervised_dataset_v1.pkl \
    --output models/supervised_value_v6_from_v1.pth \
    --epochs 5 \
    --shaped-rewards \
    --device mps

# Continue with v2 dataset
python train_supervised_mixed.py \
    --init-model models/supervised_value_v6_from_v1.pth \
    --load-data data/supervised_dataset_v2.pkl \
    --output models/supervised_value_v6_from_v2.pth \
    --epochs 5 \
    --shaped-rewards \
    --device mps

# Continue with v3 dataset
python train_supervised_mixed.py \
    --init-model models/supervised_value_v6_from_v2.pth \
    --load-data data/supervised_dataset_v3.pkl \
    --output models/supervised_value_v6_from_v3.pth \
    --epochs 5 \
    --shaped-rewards \
    --device mps

# Continue with v4 dataset (final)
python train_supervised_mixed.py \
    --init-model models/supervised_value_v6_from_v3.pth \
    --load-data data/supervised_dataset_v4.pkl \
    --output models/supervised_value_v6.pth \
    --epochs 5 \
    --shaped-rewards \
    --device mps
```

**Status:** Ready to train

**Notes:**
- **Rationale for removing HOLD**: Simplifies action space, HOLD action rarely used effectively by heuristic teacher
- **Rationale for death penalty**: Discourage risky play that leads to game over, help agent learn to avoid terminal states
- **Dataset filtering**: Automatically removes action=6 transitions during training (expect ~5-10% data loss)
- **Model architecture change**: Output layer now 128→6 instead of 128→7
- **Incompatible with v1-v5 weights**: Cannot initialize from previous models due to different output size
- **Expected benefits**: Simpler policy, faster training, focus on core Tetris mechanics, better survival

---

## Historical Notes

**Pre-v1 (Deprecated):**
All models trained before 2025-01-23 used incorrect approach (pre-computed trajectory returns instead of iterative Bellman updates). These models and datasets have been discarded.

---

## Testing Commands

**Test supervised model:**
```bash
python main.py --agent value --model-path models/supervised_value_v1.pth --episodes 5 --device mps
```

**Test with visualization:**
```bash
python main.py --agent value --model-path models/supervised_value_v1.pth --episodes 1 --render --device mps
```

**Test mixed teacher:**
```bash
python main.py --agent mixed --episodes 5
```

## RL Fine-tuning

Once supervised training produces a competent baseline, use RL for fine-tuning:

```bash
python train_rl.py \
    --init-model models/supervised_value_v1.pth \
    --output models/rl_value_v1.pth \
    --num-episodes 300 \
    --epsilon-start 0.3 \
    --epsilon-end 0.05 \
    --device mps \
    --save-interval 50
```
