from rl_agents.service import AgentService

from abc import ABC, abstractmethod
import numpy as np
import torch

class AbstractActionStrategy(AgentService, ABC):
    @abstractmethod
    def pick_action(self, state : np.ndarray) -> np.ndarray:
        ...
    
