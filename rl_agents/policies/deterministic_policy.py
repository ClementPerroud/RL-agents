from rl_agents.policies.policy import ContinuousPolicy, DiscretePolicy

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from rl_agents.agent import BaseAgent

import torch
import math
import gymnasium as gym
from abc import abstractmethod

class ContinuousDeterministicPolicy(ContinuousPolicy):
    def __init__(self, action_space : gym.spaces.Box, core_net : torch.nn.Module, *args, **kwargs):
        super().__init__(core_net=core_net, action_space=action_space, *args, **kwargs)

    def pick_action(self, state : torch.Tensor, **kwargs):
        # state : [batch, state_shape ...]
        raw_action = self.forward(state=state)
        action = self._squash(raw_action)
        return action
        # shape [batch]

