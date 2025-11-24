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

**Status:** Training in progress

**Notes:**
- Continue training v3 with greedy heuristic (temp=0)
- Only source of randomness is random action probability (~20%)

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
