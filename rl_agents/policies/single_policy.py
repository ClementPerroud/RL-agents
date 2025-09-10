from rl_agents.policies.policy import AbstractPolicy
from rl_agents.agent import BaseAgent
import numpy as np
import torch


# Use for testing purpose
class DummyPolicy(
    AbstractPolicy,
):
    def __init__(self, action):
        super().__init__()
        self.action = action

    def pick_action(self, state: torch.Tensor, **kwargs) -> torch.Tensor:
        nb_env = state.size(0)
        if nb_env > 1:  # Return must tensor of shape [nb_env,]
            return torch.ones(state.shape[0], dtype=torch.long) * self.action
        # Else : nb_env = 1
        return torch.tensor(self.action, dtype=torch.long)
