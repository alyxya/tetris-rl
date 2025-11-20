# Reward Analysis for RL Training

## Key Findings from Hybrid (50% Heuristic + 50% Random) Rollouts

### Environment Rewards (PufferLib Tetris)
- **Mean**: 0.017 per step
- **Std**: 0.060
- **Range**: [0.0, 0.38]
- **Sparsity**: 76.6% of steps have zero reward
- **Non-zero mean**: 0.074 (when reward > 0)

**Distribution:**
- `0.01`: Line clear rewards (most common non-zero)
- `0.30`, `0.16`, `0.18`, `0.24`: Multi-line clears (rare but high value)
- Most steps (76.6%) give no reward at all

### Heuristic Rewards (From heuristic evaluation function)
- **Mean**: -29.4 per step
- **Std**: 20.3
- **Range**: [-93.8, -2.4]
- **Always negative** (penalties for height, holes, bumpiness)

The heuristic evaluates the *best possible placement* and returns a score where:
- Higher (less negative) = better board state
- Lower (more negative) = worse board state

### Episode Statistics (Hybrid Agent)
- **Mean episode length**: 175.8 steps
- **Range**: 114 - 229 steps
- **Cumulative env reward per episode**: 2.27 - 4.11
- **Cumulative heuristic reward per episode**: -2167 to -8744

## Implications for RL Training

### 1. **Environment Rewards Are Very Sparse**
- Only 23.4% of actions receive non-zero rewards
- This makes learning difficult without additional signals
- **Recommendation**: Use heuristic rewards as primary training signal

### 2. **Heuristic Rewards Are Dense**
- Every step has a heuristic evaluation
- Provides consistent learning signal
- Negative values indicate "distance" from good play

### 3. **Environment Rewards Not Used**
**Decision**: Use only heuristic rewards for training.

**Rationale**:
- Environment rewards are too sparse (76.6% zeros)
- Insufficient learning signal for RL
- Heuristic provides dense, consistent feedback
- Scale mismatch would require careful tuning

Training uses: `reward = heuristic_reward` only

### 4. **Hybrid vs Pure Heuristic Performance**
- **Hybrid (50/50)**: ~176 steps per episode, ~3.0 env reward
- **Pure Heuristic**: 10,000+ steps, ~92 env reward
- Random actions significantly hurt performance (as expected)

This validates that:
- ✅ Heuristic agent is strong
- ✅ Random exploration is detrimental
- ✅ RL needs to learn to avoid random-like behavior

## Recommendations for RL Training

### For Value Network (Q-learning)

**Train with heuristic rewards only:**
```bash
python train_rl.py --mode value \\
    --num-episodes 1000 \\
    --output models/value_rl.pt
```

**Why:**
- Heuristic rewards are dense (every step)
- They guide toward good board states
- Environment rewards too sparse for learning
- No need to tune reward weighting

### Expected Learning Trajectory

1. **Early training**: Agent learns to avoid terrible moves (huge negative heuristic)
2. **Mid training**: Agent learns to prefer better board states (less negative)
3. **Late training**: Agent learns to maximize line clears (positive env rewards)

### Monitoring Training

Watch for:
- **Decreasing negative heuristic rewards** (e.g., -50 → -30 → -20)
- **Increasing episode length** (random = ~176, good = 5000+)
- **Increasing environment rewards** (more line clears)

## Sanity Checks Passed ✅

1. ✅ Environment rewards are sparse but meaningful (line clears)
2. ✅ Heuristic rewards are dense and consistent
3. ✅ Hybrid agent shows mix of good/bad behavior (validates exploration)
4. ✅ Pure heuristic significantly outperforms hybrid (validates teacher quality)
5. ✅ Reward scales are reasonable for RL (not too extreme)

You're ready to start RL training with confidence that the reward signals make sense!
