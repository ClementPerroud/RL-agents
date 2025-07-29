from rl_agents.replay_memory import ReplayMemory
from rl_agents.element import AgentService

from  collections.abc import Callable
from abc import ABC, abstractmethod
import numpy as np

class AbstractSampler(ABC, AgentService):
    def __init__(self):
        self._replay_memory : ReplayMemory = None

    @abstractmethod
    def sample(self, batch_size : int) -> tuple[np.ndarray, np.ndarray]:
        ...
    
    def update(self, infos : dict):
        ...



class ClassicSampler(AbstractSampler):
    def __init__(self):
        pass
    
    def sample(self, batch_size):
        batch = np.random.choice(self.replay_memory.size(), size = batch_size,
                replace=False)
        return batch, 1


from rl_agents.utils.sumtree import SumTree
class PrioritizedReplaySampler(AbstractSampler):
        
    def __init__(self, length, alpha = 0.65):
        self.length = length
        self.alpha = alpha
        self.priorities = SumTree(size= self.length)
        self.last_batch = None
        self.step = 0
    
    def sample(self, batch_size : int):
        beta = min(1, 0.5 + 0.5*self.step/150_000)
        if beta >= 1: 
            self.replay_memory.sampler = ClassicSampler()

        batch = self.priorities.sample(batch_size)
        weights = (self.replay_memory.size() * self.priorities[batch]/self.priorities.sum() ) ** (- beta)
        weights = weights / np.amax(weights)

        self.last_batch = batch
        return batch, weights

    def update(self, infos : dict):

        td_errors = infos["td_errors"]
        self.priorities[self.last_batch] = (np.abs(td_errors)+ 1)**self.alpha
        self.step = infos["step"]