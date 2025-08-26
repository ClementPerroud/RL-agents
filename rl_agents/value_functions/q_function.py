from rl_agents.agent import AbstractAgent
from rl_agents.service import AgentService
from rl_agents.value_functions.v_function import AbstractVFunction
from abc import ABC, abstractmethod
from typing import Callable
import torch

class AbstractQFunction(AbstractVFunction):
    @abstractmethod
    def Q(self, state: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def Q_a(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor: ...
