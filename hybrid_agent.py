"""
Hybrid agent that mixes multiple agents.

Can combine any agents (heuristic, policy, value) plus random exploration
with configurable probabilities.
"""

import numpy as np
from base_agent import BaseAgent


class HybridAgent(BaseAgent):
    """Agent that probabilistically mixes actions from multiple agents."""

    def __init__(self, agents, probabilities, n_rows=20, n_cols=10):
        """
        Initialize hybrid agent.

        Args:
            agents: List of agent instances or 'random' string
                   e.g., [heuristic_agent, policy_agent, 'random']
            probabilities: List of selection probabilities (must sum to 1.0)
                          e.g., [0.5, 0.3, 0.2]
            n_rows: Board height
            n_cols: Board width
        """
        super().__init__(n_rows, n_cols)

        if len(agents) != len(probabilities):
            raise ValueError("agents and probabilities must have same length")

        if not np.isclose(sum(probabilities), 1.0):
            raise ValueError(f"probabilities must sum to 1.0, got {sum(probabilities)}")

        self.agents = agents
        self.probabilities = probabilities

        # Track usage statistics
        self.action_counts = [0] * len(agents)
        self.total_actions = 0

    def reset(self):
        """Reset agent state and all sub-agents."""
        for agent in self.agents:
            if agent != 'random' and hasattr(agent, 'reset'):
                agent.reset()

    def choose_action(self, obs):
        """
        Choose action by randomly selecting an agent based on probabilities.

        Args:
            obs: Flattened observation

        Returns:
            action: Selected action (0-6)
        """
        # Select which agent to use
        agent_idx = np.random.choice(len(self.agents), p=self.probabilities)
        selected_agent = self.agents[agent_idx]

        # Track usage
        self.action_counts[agent_idx] += 1
        self.total_actions += 1

        # Get action from selected agent
        if selected_agent == 'random':
            return np.random.randint(0, 7)
        else:
            return selected_agent.choose_action(obs)

    def get_usage_stats(self):
        """
        Get statistics on agent usage.

        Returns:
            stats: Dict with agent names and usage percentages
        """
        if self.total_actions == 0:
            return {f"agent_{i}": 0.0 for i in range(len(self.agents))}

        stats = {}
        for i, (agent, count) in enumerate(zip(self.agents, self.action_counts)):
            agent_name = 'random' if agent == 'random' else f"agent_{i}"
            stats[agent_name] = count / self.total_actions
        return stats

    def reset_stats(self):
        """Reset usage statistics."""
        self.action_counts = [0] * len(self.agents)
        self.total_actions = 0
