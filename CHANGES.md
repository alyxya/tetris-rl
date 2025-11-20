# Recent Changes

## Environment Reward Removed (2025-11-20)

### What Changed
Removed environment reward entirely from RL training. Now uses **heuristic reward only**.

### Files Modified
- `train_rl.py`:
  - Removed `--heuristic-weight` and `--env-weight` flags
  - Changed `reward = args.heuristic_weight * heuristic_reward + args.env_weight * env_reward` to `reward = compute_heuristic_reward(locked, active)`
  - Applies to both value and policy training modes

- `ARCHITECTURE.md`:
  - Updated to reflect heuristic-only training
  - Removed references to reward mixing

- `REWARD_ANALYSIS.md`:
  - Added rationale for environment reward removal
  - Updated training recommendations

### Rationale
Environment rewards from PufferLib Tetris are:
- **Too sparse**: 76.6% of steps return zero reward
- **Insufficient signal**: Only 23.4% of actions get feedback
- **Poor for learning**: RL agents need dense rewards

Heuristic rewards are:
- **Dense**: Every step gets a reward
- **Consistent**: Guides toward good board states
- **Sufficient**: No need to mix with environment rewards

### Training Commands (Updated)

**Before:**
```bash
python train_rl.py --mode value --num-episodes 1000 \
    --heuristic-weight 1.0 --env-weight 0.0 \
    --output models/value_rl.pt
```

**After:**
```bash
python train_rl.py --mode value --num-episodes 1000 \
    --output models/value_rl.pt
```

Much simpler! 🎉

### Benefits
- ✅ Simpler training interface
- ✅ No need to tune reward weights
- ✅ Clearer design decision
- ✅ Focus on what works (heuristic rewards)
