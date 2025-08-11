from rl_agents.service import AgentService

from abc import ABC, abstractmethod
import numpy as np

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rl_agents.agent import AbstractAgent
    
class AbstractActionStrategy(AgentService, ABC):
    @abstractmethod
    def pick_action(self, agent : 'AbstractAgent', state: np.ndarray) -> np.ndarray: ...

class NoActionStrategy(AgentService, ABC):
    def pick_action(self, agent: 'AbstractAgent', state: np.ndarray):
        return agent._pick_deterministic_action(state)