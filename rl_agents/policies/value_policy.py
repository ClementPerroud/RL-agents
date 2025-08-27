from rl_agents.policies.policy import AbstractPolicy
from rl_agents.agent import AbstractAgent
from rl_agents.value_functions.q_function import AbstractQFunction
import numpy as np
import torch


# Use for testing purpose
class QValueDiscretePolicy(
    AbstractPolicy,
):
    def __init__(self, net : AbstractQFunction):
        super().__init__()
        self.net = net
        # input [batch, state_shape], action [batch, action_shape] -> output [bath]

    def pick_action(self, state: torch.Tensor):
        return torch.argmax(
            self.net(state = state), 
            dim=-1
        )
    
class QValuePolicy(
    AbstractPolicy,
):
    def __init__(self, q_function : AbstractQFunction):
        super().__init__()
        self.q_function = q_function.Q

    def pick_action(self, state: torch.Tensor):
        return torch.argmax(
            self.q_function(state = torch.as_tensor(state)), 
            dim=-1
        )