# Tetris RL Architecture

This document describes the clean, minimal architecture for training and running Tetris agents.

## Overview

The codebase supports four agent types with a shared heuristic reward function:
1. **HeuristicAgent** - Rule-based agent using heuristic evaluation
2. **PolicyAgent** - Neural network that samples from action distribution
3. **ValueAgent** - Neural network that selects actions based on Q-values
4. **HybridAgent** - Mixes any combination of agents (including random)

## Core Components

### Model Architecture (`model.py`)

**SharedCNN**: Common CNN backbone for processing dual-board representation
- Custom wall padding (left, right, bottom with value 1.0)
- 3 convolutional layers: 1→32→64→64 channels
- Processes both `board_empty` (locked pieces) and `board_filled` (locked + active)

**PolicyNetwork**: Outputs action logits (7 actions)
- Dual CNN features → concat → MLP → logits
- Used for sampling actions via softmax

**ValueNetwork**: Outputs Q-values (7 actions)
- Dual CNN features → concat → MLP → Q-values
- Used for greedy/epsilon-greedy action selection

### Heuristic System (`heuristic.py`)

Central heuristic function that evaluates state-action pairs:
- **Primary focus**: Line clears (weight: 10.0)
- **Secondary factors**: Height (-0.51), Holes (-0.36), Bumpiness (-0.18)

Key functions:
- `evaluate_placement()`: Score a specific (rotation, column) placement
- `find_best_placement()`: Search all rotations and columns for best move
- `simulate_drop()`: Simulate piece placement and count line clears

This heuristic is used by:
1. HeuristicAgent for action selection
2. ValueAgent training as reward signal
3. PolicyAgent RL training as auxiliary reward

### Agent Types

#### `base_agent.py`
Abstract base class with common utilities:
- `parse_observation()`: Extract board, locked pieces, active piece
- `prepare_board_inputs()`: Create dual board representation
- `extract_piece_shape()`: Get minimal bounding box of active piece
- `choose_action()`: Abstract method (implemented by subclasses)

#### `heuristic_agent.py`
- Uses `find_best_placement()` from heuristic module
- Multi-step planning: rotate → move → drop
- Deterministic behavior based on current state

#### `policy_agent.py`
- Samples actions from PolicyNetwork logits
- Supports deterministic mode (argmax) or stochastic (softmax)
- Temperature parameter controls exploration

#### `value_agent.py`
- Selects actions with highest Q-value from ValueNetwork
- Supports epsilon-greedy exploration

#### `hybrid_agent.py`
- Mixes any agents with configurable probabilities
- Supports 'random' as a special agent type
- Tracks usage statistics across agents

## Training Scripts

### Supervised Learning (`train_supervised.py`)

Two modes:

**Value Network (`--mode value`)**:
1. Collect rollouts from HeuristicAgent
2. Compute heuristic rewards for each action
3. Calculate discounted returns with γ=0.99
4. Train ValueNetwork to predict Q-values via MSE loss

**Policy Network (`--mode policy`)**:
1. Collect rollouts from HeuristicAgent (teacher)
2. Record (state, action) pairs
3. Train PolicyNetwork to imitate via cross-entropy loss

Usage:
```bash
# Train value network
python train_supervised.py --mode value --num-episodes 100 --output models/value.pt

# Train policy network
python train_supervised.py --mode policy --num-episodes 100 --output models/policy.pt
```

### Reinforcement Learning (`train_rl.py`)

Two modes:

**Value Network (`--mode value`)**:
- Q-learning with experience replay
- Online and target networks (update every 10 episodes)
- Epsilon-greedy exploration (0.2 → 0.01)
- Uses heuristic reward only

**Policy Network (`--mode policy`)**:
- REINFORCE algorithm
- Uses heuristic reward only

Usage:
```bash
# Train value network with RL
python train_rl.py --mode value --num-episodes 1000 --output models/value_rl.pt

# Train policy network with RL
python train_rl.py --mode policy --num-episodes 1000 --output models/policy_rl.pt

# Optional: Start from supervised pretrained model
python train_rl.py --mode value --num-episodes 1000 \\
    --init-model models/value_supervised.pt --output models/value_rl.pt
```

## Running Agents (`main.py`)

### Single Agent
```bash
# Heuristic agent
python main.py --agent heuristic --episodes 5

# Policy agent
python main.py --agent policy --model-path models/policy.pt --episodes 5

# Value agent
python main.py --agent value --model-path models/value.pt --episodes 5
```

### Hybrid Agent
```bash
# Mix heuristic + policy (50/50)
python main.py --agent hybrid --agents heuristic,policy --probs 0.5,0.5 \\
    --policy-model models/policy.pt --episodes 5

# Mix heuristic + value + random (40/40/20)
python main.py --agent hybrid --agents heuristic,value,random --probs 0.4,0.4,0.2 \\
    --value-model models/value.pt --episodes 5
```

## Training Pipeline Recommendations

### Stage 1: Supervised Pretraining
1. Train value network on heuristic rollouts
2. Train policy network to imitate heuristic agent
3. Provides strong initialization before RL

### Stage 2: RL Fine-tuning
1. Fine-tune value network with Q-learning
2. Fine-tune policy network with REINFORCE
3. Use heuristic reward as guiding signal

### Stage 3: Evaluation
1. Test individual agents (heuristic, policy, value)
2. Test hybrid combinations
3. Compare performance across different mixing ratios

## Key Design Decisions

### Why Dual Board Representation?
- **board_empty**: Shows where pieces can be placed (collision detection)
- **board_filled**: Shows final placement (value estimation)
- Separates "current state" from "potential state" explicitly

### Reward Structure

**Two-component reward:**
1. **Line clear reward** (immediate, large): 1.0/3.0/6.0/10.0 for 1/2/3/4 lines
2. **Distance nudge** (small): `0.01 / (rotations + moves + 1)` toward optimal placement

**Properties:**
- Line clear rewards dominate (100-1000x larger than nudges)
- Nudges guide exploration when no lines are cleared
- Max nudge = 0.01 (when 1 action away from optimal)
- Typical nudge = 0.001-0.005 (a few actions away)
- Heuristic agent still works the same (uses find_best_placement for decisions)

**Why this structure:**
- Sparse line clears alone are insufficient for learning
- Dense nudges provide continuous gradient toward good play
- Scale separation ensures line clears remain primary objective

### Why Minimal Flags?
- Previous version had complex configuration and many unused features
- New version focuses on core functionality
- Easy to extend without breaking existing code

## File Structure

```
tetris-rl/
├── model.py                 # Neural network architectures
├── heuristic.py            # Shared heuristic reward function
├── base_agent.py           # Abstract agent base class
├── heuristic_agent.py      # Rule-based agent
├── policy_agent.py         # Policy network agent
├── value_agent.py          # Value network agent
├── hybrid_agent.py         # Agent mixer
├── train_supervised.py     # Supervised learning script
├── train_rl.py             # RL training script
├── main.py                 # Inference CLI
└── test_agents.py          # Quick test script
```

## Next Steps

1. Train supervised models to establish baseline
2. Experiment with RL fine-tuning
3. Test hybrid combinations for data collection
4. Adjust heuristic weights if needed
5. Consider adding more sophisticated RL algorithms (PPO, SAC, etc.)
