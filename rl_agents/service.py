from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
import torch
from enum import Enum

if TYPE_CHECKING:
    from rl_agents.agent import AbstractAgent


class AgentService(torch.nn.Module, ABC):
    def __init__(self, *args, **kwargs):
        torch.nn.Module.__init__(self, *args, **kwargs)


    def __update__(self, agent: "AbstractAgent"):
        self.update(agent=agent)
        for element in self.children():
            if isinstance(element, AgentService):
                element.__update__(agent=self)
        
    def update(self, agent: "AbstractAgent"): ...
