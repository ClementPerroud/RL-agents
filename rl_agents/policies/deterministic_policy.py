from rl_agents.policies.policy import AbstractPolicy

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from rl_agents.agent import AbstractAgent

import torch
import math
import gymnasium as gym
from abc import abstractmethod

class AbstractDeterministicPolicy(AbstractPolicy):
    def __init__(self, core_net : torch.nn.Module, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.core_net = core_net

class ContinuousDeterministicPolicy(AbstractDeterministicPolicy):
    def __init__(self, action_space : gym.spaces.Box, core_net, *args, **kwargs):
        super().__init__(core_net=core_net, *args, **kwargs)
        self.register_buffer("low",   torch.as_tensor(action_space.low, dtype = torch.float32))
        self.register_buffer("high",  torch.as_tensor(action_space.high, dtype = torch.float32))
        self.register_buffer("scale", (self.high - self.low) / 2)
        self.register_buffer("loc",   (self.high + self.low) / 2)

        self.head = torch.nn.LazyLinear(out_features=action_space.shape[0])


    def forward(self, state : torch.Tensor):
        x = self.core_net(state)
        return self.head(x) # ] -∞ ; +∞ [

    def pick_action(self, state : torch.Tensor, **kwargs):
        # state : [batch, state_shape ...]
        raw_action = self.forward(state=state)
        action, _ = self._squash(raw_action)
        return action
        # shape [batch]

    def _squash(self, u):
        a = torch.tanh(u)
        return self.loc + a * self.scale, a 
