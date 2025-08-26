from rl_agents.policies.policy import AbstractPolicy
from rl_agents.agent import AbstractAgent
from rl_agents.value_functions.q_function import AbstractQFunction
import numpy as np
import torch


# Use for testing purpose
class QValuePolicy(
    AbstractPolicy,
):
    def __init__(self, q_function : AbstractQFunction):
        super().__init__()
        self.q_function = q_function

    def pick_action(self, agent: AbstractAgent, state: torch.Tensor):
        return torch.argmax(
            self.q_function.Q(state = torch.as_tensor(state)), 
            dim=-1
        )