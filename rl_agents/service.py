from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
import torch
from enum import Enum

if TYPE_CHECKING:
    from rl_agents.agent import BaseAgent


class AgentService(torch.nn.Module, ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def update(self, **kwargs): ...
    def reset(self, **kwargs): ...
