from rl_agents.service import AgentService
from rl_agents.utils.sumtree import SumTree

from abc import ABC, abstractmethod
import numpy as np

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from rl_agents.replay_memory import ReplayMemory


class AbstractSampler(AgentService, ABC):
    @abstractmethod
    def sample(self, batch_size : int, size : int) -> tuple[np.ndarray, np.ndarray]:
        ...
    
    def update(self, infos : dict):
        ...



class RandomSampler(AbstractSampler):
    def __init__(self):
        pass
    
    def sample(self, batch_size : int, size : int):
        batch = np.random.choice(size, size = batch_size,
                replace=False)
        return batch, 1



class PrioritizedReplaySampler(AbstractSampler):
        
    def __init__(self, length, alpha = 0.65):
        self.length = length
        self.alpha = alpha
        self.priorities = SumTree(size= self.length)
        self.last_batch = None
        self.step = 0
        self.random_sampler = RandomSampler()
    
    def sample(self, batch_size : int, size : int):
        beta = min(1, 0.5 + 0.5*self.step/150_000)
        if beta >= 1: 
            return self.random_sampler.sample(batch_size=batch_size)

        batch = self.priorities.sample(batch_size)
        weights = (size * self.priorities[batch]/self.priorities.sum() ) ** (- beta)
        weights = weights / np.amax(weights)

        self.last_batch = batch
        return batch, weights

    def update(self, infos : dict):
        td_errors = infos["td_errors"]
        self.priorities[self.last_batch] = (np.abs(td_errors)+ 1)**self.alpha
        self.step = infos["step"]