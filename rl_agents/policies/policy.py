from rl_agents.service import AgentService

from abc import ABC, abstractmethod
import numpy as np
import torch

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rl_agents.agent import AbstractAgent
    
class AbstractPolicy(AgentService, ABC):
    @abstractmethod
    def pick_action(self, agent : 'AbstractAgent', state: torch.Tensor) -> torch.Tensor: ...