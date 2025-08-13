from rl_agents.policies.policy import AbstractPolicy
from rl_agents.agent import AbstractAgent
import numpy as np
import torch


# Use for testing purpose
class ValuePolicy(
    AbstractPolicy,
):
    def __init__(self, q_function):
        super().__init__()
        self.q_function = q_function

    def pick_action(self, agent: AbstractAgent, state: torch.Tensor, training : bool):
        return torch.argmax(
            self.q_function.Q(state = torch.as_tensor(state), training = training), 
            dim=-1
        )