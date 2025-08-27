from rl_agents.agent import AbstractAgent
from rl_agents.service import AgentService
from rl_agents.trainers.trainable import Trainable

from abc import ABC, abstractmethod
from typing import Callable
import torch


class AbstractVFunction(AgentService, ABC):
    @abstractmethod
    def V(self, state: torch.Tensor, training : bool) -> torch.Tensor: ...