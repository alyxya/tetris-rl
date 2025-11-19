import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TetrisCNNPolicy(nn.Module):
    """
    CNN-based policy network for Tetris that processes the grid in two ways:
    1. With falling pieces treated as filled cells (value 2 -> 1)
    2. With falling pieces treated as empty cells (value 2 -> 0)

    Both grids are processed through the same CNN, then concatenated and fed
    through an MLP to produce action logits.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        cnn_channels=[32, 64, 64],  # Number of channels in each conv layer
        cnn_kernel_sizes=[3, 3, 3],  # Kernel sizes for each conv layer
        cnn_strides=[1, 1, 1],       # Strides for each conv layer
        mlp_hidden_dims=[256, 128],  # Hidden dimensions for the MLP
        grid_height=20,
        grid_width=10,
    ):
        super().__init__()

        self.observation_space = observation_space
        self.action_space = action_space
        self.grid_height = grid_height
        self.grid_width = grid_width

        # Handle both Discrete and MultiDiscrete action spaces
        if hasattr(action_space, 'nvec'):
            self.num_actions = action_space.nvec[0]  # MultiDiscrete
        else:
            self.num_actions = action_space.n  # Discrete

        # Build CNN layers (shared between both grid representations)
        self.conv_layers = nn.ModuleList()
        in_channels = 1  # Single channel input (the grid)

        for i, (out_channels, kernel_size, stride) in enumerate(
            zip(cnn_channels, cnn_kernel_sizes, cnn_strides)
        ):
            # Padding='same' equivalent: we want to maintain spatial dimensions
            # For padding, we use value=1.0 (filled cell) as specified
            self.conv_layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=kernel_size // 2,  # 'same' padding
                )
            )
            in_channels = out_channels

        # Calculate the output size after convolutions
        # Since we use 'same' padding with stride=1, spatial dims stay the same
        # The output will be: (batch, channels[-1], height, width)
        cnn_output_size = cnn_channels[-1] * grid_height * grid_width

        # We process two grids, so we concatenate their outputs
        mlp_input_size = cnn_output_size * 2

        # Build MLP layers
        self.mlp_layers = nn.ModuleList()
        prev_dim = mlp_input_size

        for hidden_dim in mlp_hidden_dims:
            self.mlp_layers.append(nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim

        # Final layer to produce action logits
        self.action_head = nn.Linear(prev_dim, self.num_actions)

        # Value head for actor-critic
        self.value_head = nn.Linear(prev_dim, 1)

    def extract_grid(self, obs):
        """
        Extract the 20x10 grid from the flattened observation.

        Args:
            obs: Tensor of shape (batch, 1, 234) or (batch, 234)

        Returns:
            grid: Tensor of shape (batch, 1, 20, 10)
        """
        # Flatten if needed
        if obs.dim() == 3:
            obs = obs.squeeze(1)  # (batch, 234)

        # Extract first 200 values (20x10 grid)
        grid_flat = obs[:, :200]

        # Reshape to (batch, 1, 20, 10)
        batch_size = obs.shape[0]
        grid = grid_flat.view(batch_size, 1, self.grid_height, self.grid_width)

        return grid

    def create_dual_grids(self, grid):
        """
        Create two versions of the grid:
        1. Falling pieces (value 2) treated as filled (converted to 1)
        2. Falling pieces (value 2) treated as empty (converted to 0)

        Args:
            grid: Tensor of shape (batch, 1, 20, 10)

        Returns:
            grid_filled: Grid with falling pieces as filled cells
            grid_empty: Grid with falling pieces as empty cells
        """
        # Clone the grid to avoid modifying the original
        grid_filled = grid.clone()
        grid_empty = grid.clone()

        # Falling pieces have value 2
        # For grid_filled: 2 -> 1 (filled)
        grid_filled[grid_filled == 2.0] = 1.0

        # For grid_empty: 2 -> 0 (empty)
        grid_empty[grid_empty == 2.0] = 0.0

        # Also handle the 0.5 values (possibly next piece preview) - treat as empty
        grid_filled[grid_filled == 0.5] = 0.0
        grid_empty[grid_empty == 0.5] = 0.0

        return grid_filled, grid_empty

    def apply_cnn(self, grid):
        """
        Apply CNN layers to the grid with proper padding.

        Args:
            grid: Tensor of shape (batch, 1, 20, 10)

        Returns:
            features: Flattened CNN output
        """
        x = grid

        # Apply each conv layer with ReLU activation
        for conv_layer in self.conv_layers:
            # Apply padding manually with value=1.0 (filled cell)
            # Calculate padding needed
            kernel_size = conv_layer.kernel_size[0]
            padding_size = kernel_size // 2

            # Pad with value 1.0 (filled cells outside the grid)
            x = F.pad(x, (padding_size, padding_size, padding_size, padding_size),
                     mode='constant', value=1.0)

            # Apply convolution (with padding=0 since we manually padded)
            conv_layer.padding = 0
            x = conv_layer(x)
            x = F.relu(x)

        # Flatten the output
        x = x.view(x.size(0), -1)

        return x

    def forward(self, obs):
        """
        Forward pass through the network.

        Args:
            obs: Observation tensor from the environment

        Returns:
            action_logits: Logits for action selection (batch, 7)
            value: State value estimate (batch, 1)
        """
        # Extract the grid from observation
        grid = self.extract_grid(obs)

        # Create two versions of the grid
        grid_filled, grid_empty = self.create_dual_grids(grid)

        # Apply CNN to both grids
        features_filled = self.apply_cnn(grid_filled)
        features_empty = self.apply_cnn(grid_empty)

        # Concatenate the features from both grids
        combined_features = torch.cat([features_filled, features_empty], dim=1)

        # Pass through MLP layers
        x = combined_features
        for mlp_layer in self.mlp_layers:
            x = F.relu(mlp_layer(x))

        # Get action logits and value
        action_logits = self.action_head(x)
        value = self.value_head(x)

        return action_logits, value

    def get_action_and_value(self, obs, action=None):
        """
        Get action distribution and value. Compatible with CleanRL/PufferLib interface.

        Args:
            obs: Observation tensor
            action: Optional action tensor for computing log_prob

        Returns:
            action: Sampled action (if action is None)
            log_prob: Log probability of the action
            entropy: Entropy of the action distribution
            value: State value estimate
        """
        action_logits, value = self.forward(obs)

        # Create categorical distribution
        probs = F.softmax(action_logits, dim=-1)
        dist = torch.distributions.Categorical(probs)

        # Sample action if not provided
        if action is None:
            action = dist.sample()

        # Get log probability and entropy
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, entropy, value.squeeze(-1)


# PufferLib wrapper to make it compatible with the framework
def make_policy(env, policy_args=None):
    """
    Factory function to create the policy network.
    This is the interface PufferLib expects.
    """
    if policy_args is None:
        policy_args = {}

    return TetrisCNNPolicy(
        observation_space=env.single_observation_space,
        action_space=env.single_action_space,
        **policy_args
    )
