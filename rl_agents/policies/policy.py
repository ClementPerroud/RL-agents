from rl_agents.service import AgentService

from abc import ABC, abstractmethod
import numpy as np
import torch
import gymnasium as gym

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from rl_agents.agent import BaseAgent

@runtime_checkable
class Policy(Protocol):
    def __init__(self, **kwargs): super().__init__(**kwargs)
    def pick_action(self, state: torch.Tensor) -> torch.Tensor: ...

@runtime_checkable
class StochasticPolicy(Policy, Protocol):
    def __init__(self, **kwargs): super().__init__(**kwargs)
    def action_distributions(self, state : torch.Tensor) -> torch.Tensor:...
    def evaluate_log_prob(self, action_distributions : torch.Tensor, action : torch.Tensor) -> torch.Tensor:...
    def entropy_loss(self, action_probs : torch.Tensor) -> torch.Tensor:...
    

class DiscretePolicy(AgentService, Policy, ABC):
    def __init__(self, core_net : torch.nn.Module, action_space : gym.spaces.Box, **kwargs):
        assert isinstance(action_space, gym.spaces.Discrete), "action_space only support gym.spaces.Discrete"
        super().__init__(**kwargs)
        self.action_space=action_space
        self.core_net = core_net
        self.head = torch.nn.Sequential(
            torch.nn.LazyLinear(action_space.n),
            torch.nn.Softmax(dim = -1)
        )
    
    def forward(self, state : torch.Tensor):
        x = self.core_net(state)
        return self.head(x)

class ContinuousPolicy(AgentService, Policy, ABC):
    LOG_STD_MIN, LOG_STD_MAX = -2, 5
    EPS = 1E-6
    def __init__(self, core_net : torch.nn.Module, action_space : gym.spaces.Box, **kwargs):
        assert isinstance(action_space, gym.spaces.Box), "action_space only support gym.spaces.Box"
        super().__init__(**kwargs)
        self.register_buffer("low",   torch.as_tensor(action_space.low, dtype = torch.float32))
        self.register_buffer("high",  torch.as_tensor(action_space.high, dtype = torch.float32))
        self.register_buffer("scale", (self.high - self.low) / 2)
        self.register_buffer("loc",   (self.high + self.low) / 2)
        self.action_space=action_space
        self.core_net = core_net
        self.head = torch.nn.LazyLinear(action_space.shape[0])
    
    def forward(self, state : torch.Tensor):
        x = self.core_net(state)
        return self.head(x) # #  ]-∞ ; +∞[

    @staticmethod
    def _atanh(x, eps=1e-6):
        x = x.clamp(-1 + eps, 1 - eps)
        return 0.5 * (torch.log1p(x) - torch.log1p(-x))
    
    def _squash(self, u):
        # u -> ] -∞ ; +∞ [
        a = torch.tanh(u) # ]-1 ; 1 [
        return self.loc + a * self.scale #] lower bound ; upper bound [

    def _unsquash(self, a):
        #a -> ] lower bound ; upper bound [
        u = (a - self.loc) / self.scale # ]-1 ; 1 [
        return self._atanh(u) # ] -∞ ; +∞ [
