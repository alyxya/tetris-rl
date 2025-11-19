# Tetris RL

Reinforcement learning agents for Tetris using PufferLib.

## Agents

### 1. Heuristic Agent
A rule-based agent that uses heuristics to play Tetris:
- Evaluates all 4 rotations and horizontal positions for each piece
- Prioritizes line clears
- Uses heuristics (height, holes, bumpiness) to break ties
- Achieves ~4368 steps and 39.24 reward per episode

### 2. CNN Agent
A learned agent using a convolutional neural network:
- Dual-input CNN architecture (piece as empty + piece as filled)
- Shared CNN backbone with MLP head
- Can be trained via supervised learning (imitation) then fine-tuned with RL

### 3. Hybrid Agent
A hybrid agent that combines both CNN and heuristic approaches:
- Randomly chooses between CNN and heuristic policies with 50/50 probability
- Useful for exploring hybrid strategies and comparing agent behaviors
- Tracks usage statistics to show how often each policy is selected

## Training

Train the CNN agent using on-policy data collection with teacher supervision:

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

The training addresses **distribution shift** by having the student (CNN) collect its own data:

1. **Data Collection**: Student agent plays episodes while teacher (heuristic agent) labels the states
2. **Training**: CNN trains on aggregate dataset for several epochs
3. **Evaluation**: Agent performance measured in actual gameplay
4. **Checkpointing**: Regular checkpoints saved for resuming training

The exploration probability (how often the teacher action is taken) decays from 0.9 to 0.1, allowing the student to gradually take control.

### Key Arguments

- `--teacher`: Teacher agent type (default: `heuristic`)
- `--checkpoint`: Resume from checkpoint file
- `--iterations`: Number of data collection iterations (default: `10`)
- `--episodes`: Episodes per iteration (default: `20`)
- `--epochs`: Training epochs per iteration (default: `10`)
- `--device`: `cpu` or `cuda` (default: `cpu`)
- `--save-frequency`: Save checkpoint every N iterations (default: `1`)

See [TRAINING.md](TRAINING.md) for full documentation.

### Outputs

- `checkpoints/best_val.pt`: Best validation accuracy model
- `checkpoints/best_performance.pt`: Best evaluation reward model
- `checkpoints/checkpoint_iterXXX.pt`: Periodic iteration checkpoints
- `models/cnn_agent.pt`: Final trained model (weights only)

## Running Agents

### Heuristic Agent

```bash
python main.py --agent heuristic --episodes 3
```

### CNN Agent

```bash
# Random initialization
python main.py --agent cnn --episodes 3

# With trained model
python main.py --agent cnn --model-path models/cnn_agent_best.pt --episodes 3
```

### Hybrid Agent

```bash
# With random CNN initialization
python main.py --agent hybrid --episodes 3

# With trained CNN model
python main.py --agent hybrid --model-path models/cnn_agent_best.pt --episodes 3
```

The hybrid agent will display usage statistics showing how many times each policy (CNN vs Heuristic) was selected.

Options:
- `--agent`: Agent type (heuristic, cnn, or hybrid)
- `--episodes`: Number of episodes to run
- `--model-path`: Path to CNN model weights (for cnn or hybrid agent)
- `--render`: Render the game (warning: slow)

## CNN Architecture

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
└── Linear(128, 7) -> Action logits

Output: Softmax over 7 actions
```

Total parameters: ~6.7M

## Training Hyperparameters

### Supervised Learning
- Learning rate: 1e-3
- Batch size: 128
- Optimizer: Adam
- Scheduler: ReduceLROnPlateau (factor=0.5, patience=3)
- Dropout: 0.3

### Reinforcement Learning
- Algorithm: REINFORCE (policy gradient)
- Learning rate: 1e-4
- Discount factor (γ): 0.99
- Entropy coefficient: 0.01
- Gradient clipping: max_norm=1.0
- Optimizer: Adam

## Directory Structure

```
tetris-rl/
├── agents/
│   ├── __init__.py
│   ├── heuristic_agent.py    # Rule-based agent
│   ├── cnn_agent.py           # CNN-based agent
│   └── hybrid_agent.py        # Hybrid CNN/heuristic agent
├── models/                    # Saved model checkpoints
├── checkpoints/               # Training checkpoints
├── archive/                   # Old training scripts
├── models_archive/            # Old model files
├── base_agent.py              # Base agent class
├── main.py                    # Run agents
├── train.py                   # Training script
├── TRAINING.md                # Training documentation
└── README.md
```

## Requirements

```bash
pip install torch numpy tqdm pufferlib
```
