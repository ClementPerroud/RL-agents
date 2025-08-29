from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
import torch
from enum import Enum

if TYPE_CHECKING:
    from rl_agents.agent import AbstractAgent


class AgentService(torch.nn.Module, ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def __update__(self, **kwargs):
        self.update(**kwargs)
        for element in self.children():
            if isinstance(element, AgentService):
                element.__update__(**kwargs)
        
    def update(self, **kwargs): ...
