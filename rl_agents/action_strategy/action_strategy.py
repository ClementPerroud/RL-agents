from rl_agents.service import AgentService

from abc import ABC, abstractmethod
import numpy as np


class AbstractActionStrategy(AgentService, ABC):
    @abstractmethod
    def pick_action(self, state: np.ndarray) -> np.ndarray: ...
