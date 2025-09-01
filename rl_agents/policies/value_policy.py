from rl_agents.policies.policy import Policy
from rl_agents.value_functions.value import Q
import numpy as np
import torch



# Use for testing purpose
class DiscreteBestQValuePolicy(Policy):
    def __init__(self, q : Q):
        super().__init__()
        assert isinstance(q, Q), "q must implement Q. Please use QWrapper or create a custom Q"
        self.q_per_action = q.Q_per_action 

    def pick_action(self, state: torch.Tensor, **kwargs):
        return torch.argmax(
            self.q_per_action(state = state), 
            dim=-1
        ).long()