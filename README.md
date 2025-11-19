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

## Training Pipeline

### Step 1: Supervised Learning (Imitation)

Train CNN to imitate the heuristic agent:

```bash
python train_supervised.py --episodes 100 --epochs 20 --batch-size 128 --device cpu
```

Options:
- `--episodes`: Number of episodes to collect from expert (default: 100)
- `--epochs`: Number of training epochs (default: 20)
- `--batch-size`: Batch size (default: 128)
- `--lr`: Learning rate (default: 1e-3)
- `--device`: cpu or cuda (default: cpu)
- `--val-split`: Validation split ratio (default: 0.2)

This will:
1. Collect ~400k state-action pairs from heuristic agent
2. Train CNN to predict heuristic actions
3. Save best model to `models/cnn_agent_best.pt`

### Step 2: Reinforcement Learning (Fine-tuning)

Fine-tune CNN with policy gradient (REINFORCE):

```bash
python train_rl.py --model models/cnn_agent_best.pt --episodes 1000 --device cpu
```

Options:
- `--model`: Path to pretrained model (default: models/cnn_agent_best.pt)
- `--episodes`: Number of training episodes (default: 1000)
- `--temperature`: Sampling temperature (default: 1.0)
- `--device`: cpu or cuda (default: cpu)
- `--save-interval`: Save every N episodes (default: 100)
- `--eval-interval`: Evaluate every N episodes (default: 50)

This will:
1. Load supervised pretrained model
2. Fine-tune with REINFORCE algorithm
3. Save checkpoints to `models/cnn_agent_rl_epN.pt`

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

Options:
- `--agent`: Agent type (heuristic or cnn)
- `--episodes`: Number of episodes to run
- `--model-path`: Path to CNN model weights (for cnn agent)
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
│   └── cnn_agent.py           # CNN-based agent
├── models/                    # Saved model checkpoints
├── base_agent.py              # Base agent class
├── main.py                    # Run agents
├── train_supervised.py        # Supervised learning
├── train_rl.py               # RL training
└── README.md
```

## Requirements

```bash
pip install torch numpy tqdm pufferlib
```
