# Tetris RL

Reinforcement learning agents for Tetris using PufferLib.

## Agents

### 1. Heuristic Agent
A rule-based agent that uses heuristics to play Tetris:
- Evaluates all 4 rotations and horizontal positions for each piece
- Prioritizes line clears
- Uses heuristics (height, holes, bumpiness) to break ties
- Achieves ~4368 steps and 39.24 reward per episode

### 2. Q-Value Agent
A unified neural agent that predicts action values:
- Dual-input CNN architecture (piece as empty + piece as filled)
- Outputs 7 discounted line-clear Q-values (one per discrete action)
- Trained via supervised regression on teacher-labelled returns, then fine-tuned with TD learning

### 3. Hybrid Agent
A hybrid agent that mixes Q-value and heuristic policies:
- Randomly chooses between the learned Q-value policy, heuristic policy, and optional random actions
- Useful for exploring hybrid strategies and comparing agent behaviors
- Tracks usage statistics to show how often each policy is selected

## Training

Train the Q-value agent using on-policy data collection with teacher supervision:

```bash
# Train from scratch (default settings)
python train.py

# Quick test run
python train.py --iterations 3 --episodes 10 --epochs 5

# Full training with CUDA
python train.py --iterations 20 --episodes 50 --epochs 15 --device cuda

# Resume from checkpoint
python train.py --checkpoint checkpoints/checkpoint_iter005.pt
```

### How It Works

The training addresses **distribution shift** by generating fresh experience every iteration:

1. **Data Collection**: Teacher agent (with occasional random actions) plays episodes and labels every state
2. **Training**: Q-value network regresses teacher-labelled discounted (γ≈0.99) line-clear returns on the aggregate dataset
3. **Evaluation**: Agent performance measured in actual gameplay
4. **Checkpointing**: Regular checkpoints saved for resuming training

During collection, the teacher drives almost every move while a configurable fraction of actions are random (`--random-action-prob`) to expose the model to diverse board states.

### Key Arguments

- `--teacher`: Teacher agent type (default: `heuristic`)
- `--checkpoint`: Resume from checkpoint file
- `--iterations`: Number of data collection iterations (default: `10`)
- `--episodes`: Episodes per iteration (default: `20`)
- `--epochs`: Training epochs per iteration (default: `10`)
- `--device`: `cpu`, `cuda`, or `mps` (default: `cpu`)
- `--random-action-prob`: Probability of forcing random actions during data collection (default: `0.1`)
- `--discount`: Discount factor for the supervised Q-value targets (default: `0.99`)
- `--save-frequency`: Save checkpoint every N iterations (default: `1`)

See [TRAINING.md](TRAINING.md) for full documentation.

### Outputs

- `checkpoints/best_val.pt`: Best validation loss model
- `checkpoints/best_performance.pt`: Best evaluation reward model
- `checkpoints/checkpoint_iterXXX.pt`: Periodic iteration checkpoints
- `models/q_value_agent.pt`: Final supervised model (weights only)
- `models/q_value_agent_rl.pt`: RL fine-tuned weights

## Running Agents

### Heuristic Agent

```bash
python main.py --agent heuristic --episodes 3
```

### Q-Value Agent (aliases: `cnn`, `value`)

```bash
# Random initialization
python main.py --agent q --episodes 3

# With trained model
python main.py --agent q --model-path models/q_value_agent.pt --episodes 3
```

### Hybrid Agent

```bash
# With random Q-value initialization
python main.py --agent hybrid --episodes 3

# With trained Q-value model
python main.py --agent hybrid --model-path models/q_value_agent.pt --episodes 3
```

The hybrid agent will display usage statistics showing how many times each policy (Q-value vs Heuristic) was selected.

Options:
- `--agent`: Agent type (heuristic, q/cnn/value, or hybrid)
- `--episodes`: Number of episodes to run
- `--model-path`: Path to Q-value model weights (for q/cnn/value or hybrid agent)
- `--render`: Render the game (warning: slow)

## Q-Value CNN Architecture

```
Input: Two 20x10 boards
├── Board 1: Piece treated as empty (0)
└── Board 2: Piece treated as filled (1)

Shared CNN Backbone:
├── Conv2d(1, 32, 3x3) + ReLU
├── Conv2d(32, 64, 3x3) + ReLU
└── Conv2d(64, 64, 3x3) + ReLU

Feature Concatenation: [features_empty, features_filled]

MLP Head:
├── Linear(25600, 256) + ReLU + Dropout(0.3)
├── Linear(256, 128) + ReLU + Dropout(0.3)
└── Linear(128, 7) -> Action values

Output: 7 discounted line-clear Q-values
```

Total parameters: ~6.7M

## Training Hyperparameters

### Supervised Learning
- Learning rate: 1e-3
- Batch size: 128
- Optimizer: Adam
- Loss: MSE between predicted and target Q-values
- Scheduler: ReduceLROnPlateau (factor=0.5, patience=3)
- Dropout: 0.3

### Reinforcement Learning
- Algorithm: TD learning with replay + target network (Q-learning)
- Learning rate: 1e-4
- Discount factor (γ): 0.99
- Replay buffer: 100k transitions, 2k warm-up
- Exploration: epsilon-greedy (0.2 → 0.01)
- Target update: every 1000 environment steps

## Directory Structure

```
tetris-rl/
├── agents/
│   ├── __init__.py
│   ├── heuristic_agent.py    # Rule-based agent
│   ├── q_agent.py            # Unified Q-value agent
│   └── hybrid_agent.py       # Hybrid Q/heuristic agent
├── models/                    # Saved model checkpoints
├── checkpoints/               # Training checkpoints
├── archive/                   # Old training scripts
├── models_archive/            # Old model files
├── base_agent.py              # Base agent class
├── main.py                    # Run agents
├── train.py                   # Supervised pretraining
├── train_rl.py                # Q-learning fine-tuning
├── TRAINING.md                # Training documentation
└── README.md
```

## Requirements

```bash
pip install torch numpy tqdm pufferlib
```
