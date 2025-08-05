from rl_agents.agent import AbstractAgent

import torch
from abc import ABC, abstractmethod


class AbstractQAgent(AbstractAgent, ABC):
    @abstractmethod
    def Q(self, state: torch.Tensor): ...

    @abstractmethod
    def Q_a(self, state: torch.Tensor, action: torch.Tensor): ...

    @abstractmethod
    def compute_td_errors(self): ...
