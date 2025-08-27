from rl_agents.agent import AbstractAgent
from rl_agents.service import AgentService
from rl_agents.trainers.trainable import Trainable

from abc import ABC, abstractmethod
from typing import Callable
import torch


class AbstractVFunction(AgentService, ABC):
    @abstractmethod
    def V(self, state: torch.Tensor) -> torch.Tensor: ...

    _gamma = None
    @property
    def gamma(self) -> int:
        if self._gamma is None: raise AttributeError(f"{self.__class__.__name__}.gamma is not set")
        return self._gamma
    @gamma.setter
    def gamma(self, val): self._gamma = val